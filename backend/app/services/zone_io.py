"""Zone definition import/export format conversions.

Supports:
- GeoJSON / Shapefile export via GeoDataFrame
- MODFLOW zone file export/import (structured grids only)
- GeoJSON polygon overlay → zone assignments
"""

import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured-grid polygon builder (Python port of gridPolygons.ts)
# ---------------------------------------------------------------------------

def _build_structured_centroids(
    delr: list[float],
    delc: list[float],
    xoff: float = 0.0,
    yoff: float = 0.0,
    angrot: float = 0.0,
) -> list[tuple[float, float]]:
    """Compute cell centroids for a structured grid.

    Returns a flat list of (x, y) centroids in row-major order (nrow * ncol).
    """
    ncol = len(delr)
    nrow = len(delc)
    angrot_rad = math.radians(angrot)
    cos_a = math.cos(angrot_rad)
    sin_a = math.sin(angrot_rad)

    x_edges = [0.0]
    for c in range(ncol):
        x_edges.append(x_edges[c] + delr[c])

    y_edges = [0.0]
    for r in range(nrow):
        y_edges.append(y_edges[r] + delc[r])

    total_height = y_edges[nrow]

    centroids: list[tuple[float, float]] = []
    for r in range(nrow):
        y0 = total_height - y_edges[r + 1]
        y1 = total_height - y_edges[r]
        my = (y0 + y1) / 2.0
        for c in range(ncol):
            x0 = x_edges[c]
            x1 = x_edges[c + 1]
            mx = (x0 + x1) / 2.0
            if angrot_rad != 0:
                rx = mx * cos_a - my * sin_a
                ry = mx * sin_a + my * cos_a
                centroids.append((rx + xoff, ry + yoff))
            else:
                centroids.append((mx + xoff, my + yoff))
    return centroids


def _build_structured_polygons(
    delr: list[float],
    delc: list[float],
    xoff: float = 0.0,
    yoff: float = 0.0,
    angrot: float = 0.0,
) -> list[list[tuple[float, float]]]:
    """Build closed cell polygons for a structured grid.

    Returns a flat list (nrow * ncol) of 5-point closed polygons.
    """
    ncol = len(delr)
    nrow = len(delc)
    angrot_rad = math.radians(angrot)
    cos_a = math.cos(angrot_rad)
    sin_a = math.sin(angrot_rad)

    x_edges = [0.0]
    for c in range(ncol):
        x_edges.append(x_edges[c] + delr[c])
    y_edges = [0.0]
    for r in range(nrow):
        y_edges.append(y_edges[r] + delc[r])
    total_height = y_edges[nrow]

    def _tp(lx: float, ly: float) -> tuple[float, float]:
        if angrot_rad != 0:
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            return (rx + xoff, ry + yoff)
        return (lx + xoff, ly + yoff)

    polys: list[list[tuple[float, float]]] = []
    for r in range(nrow):
        y_bot = total_height - y_edges[r + 1]
        y_top = total_height - y_edges[r]
        for c in range(ncol):
            x0 = x_edges[c]
            x1 = x_edges[c + 1]
            polys.append([
                _tp(x0, y_bot),
                _tp(x1, y_bot),
                _tp(x1, y_top),
                _tp(x0, y_top),
                _tp(x0, y_bot),
            ])
    return polys


# ---------------------------------------------------------------------------
# Polygon-grid helpers (DISV / DISU)
# ---------------------------------------------------------------------------

def _load_polygon_grid_centroids(grid_geometry: dict) -> list[tuple[float, float]]:
    """Extract cell centroids from grid_geometry JSON (polygon grids).

    grid_geometry format: {layers: {"0": {polygons: [...]}}, ncpl, ...}
    Only loads layer 0 polygons (cell centroids are the same across layers).
    """
    centroids: list[tuple[float, float]] = []
    layer_data = grid_geometry.get("layers", {}).get("0", {})
    polygons = layer_data.get("polygons", [])
    for poly in polygons:
        if not poly or len(poly) < 3:
            centroids.append((0.0, 0.0))
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        centroids.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return centroids


# ---------------------------------------------------------------------------
# Export: zone_layers → GeoDataFrame
# ---------------------------------------------------------------------------

def zone_layers_to_geodataframe(
    zone_layers: dict[str, dict[str, list[int]]],
    *,
    nrow: int,
    ncol: int,
    nlay: int,
    delr: list[float] | None = None,
    delc: list[float] | None = None,
    xoff: float = 0.0,
    yoff: float = 0.0,
    angrot: float = 0.0,
    epsg: int | None = None,
    grid_type: str = "structured",
    grid_geometry: dict | None = None,
):
    """Convert zone_layers to a GeoDataFrame with merged geometries per (layer, zone).

    Returns a GeoDataFrame with columns: zone_name, zone_number, layer, cell_count, geometry.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
    except ImportError:
        raise RuntimeError("geopandas and shapely are required for GIS export")

    is_structured = grid_type == "structured"

    if is_structured:
        if not delr or not delc:
            raise ValueError("delr and delc required for structured grid export")
        polys = _build_structured_polygons(delr, delc, xoff, yoff, angrot)
        total_cells = nrow * ncol
    else:
        if not grid_geometry:
            raise ValueError("grid_geometry required for polygon grid export")
        layer_data = grid_geometry.get("layers", {}).get("0", {})
        raw_polys = layer_data.get("polygons", [])
        polys = [[(p[0], p[1]) for p in poly] + [(poly[0][0], poly[0][1])]
                 for poly in raw_polys if poly and len(poly) >= 3]
        total_cells = len(polys)

    rows = []
    for layer_str, layer_zones in zone_layers.items():
        lay = int(layer_str)
        # Group cell indices by zone name
        for zone_name, cell_indices in layer_zones.items():
            cell_polys = []
            for ci in cell_indices:
                if 0 <= ci < total_cells:
                    try:
                        cell_polys.append(Polygon(polys[ci]))
                    except Exception:
                        continue
            if not cell_polys:
                continue
            merged = unary_union(cell_polys)
            if not isinstance(merged, (Polygon, MultiPolygon)):
                continue
            # Parse zone number from name
            zone_num = 0
            parts = zone_name.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                zone_num = int(parts[-1])
            rows.append({
                "zone_name": zone_name,
                "zone_number": zone_num,
                "layer": lay,
                "cell_count": len(cell_indices),
                "geometry": merged,
            })

    if not rows:
        gdf = gpd.GeoDataFrame(
            columns=["zone_name", "zone_number", "layer", "cell_count", "geometry"],
            geometry="geometry",
        )
    else:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry")

    if epsg:
        gdf = gdf.set_crs(epsg=epsg, allow_override=True)

    return gdf


# ---------------------------------------------------------------------------
# Export: zone_layers → MODFLOW zone file (structured grids only)
# ---------------------------------------------------------------------------

def zone_layers_to_modflow_zone_file(
    zone_layers: dict[str, dict[str, list[int]]],
    *,
    nlay: int,
    nrow: int,
    ncol: int,
) -> str:
    """Convert zone_layers to MODFLOW zone-file text format.

    Works only for structured grids.
    Returns the file content as a string.
    """
    # Collect zone names
    all_zone_names: list[str] = []
    for layer_zones in zone_layers.values():
        for zn in layer_zones:
            if zn not in all_zone_names:
                all_zone_names.append(zn)
    zone_name_to_num = {name: idx + 1 for idx, name in enumerate(all_zone_names)}

    # Build zone array
    zone_array = np.zeros((nlay, nrow, ncol), dtype=int)
    for layer_str, layer_zones in zone_layers.items():
        lay = int(layer_str)
        if lay < 0 or lay >= nlay:
            continue
        for zone_name, cell_indices in layer_zones.items():
            znum = zone_name_to_num[zone_name]
            for ci in cell_indices:
                r = ci // ncol
                c = ci % ncol
                if 0 <= r < nrow and 0 <= c < ncol:
                    zone_array[lay, r, c] = znum

    lines: list[str] = []
    lines.append("# Zone Budget Definition File")
    if all_zone_names:
        mapping = ", ".join(f"{name} = {zone_name_to_num[name]}" for name in all_zone_names)
        lines.append(f"# {mapping}")

    for lay in range(nlay):
        lines.append(f"# Layer {lay + 1}")
        for r in range(nrow):
            lines.append(" ".join(str(zone_array[lay, r, c]) for c in range(ncol)))

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Import: MODFLOW zone file → zone_layers
# ---------------------------------------------------------------------------

def parse_modflow_zone_file(
    content: str,
    *,
    nlay: int,
    nrow: int,
    ncol: int,
) -> tuple[dict[str, dict[str, list[int]]], int]:
    """Parse a MODFLOW zone file back into zone_layers format.

    Returns (zone_layers, num_zones).
    """
    # Try to extract zone name mapping from comment header
    name_map: dict[int, str] = {}
    data_lines: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("!"):
            # Try to parse "Zone N = M" mappings from comment
            import re
            for match in re.finditer(r'(Zone\s*\d+)\s*=\s*(\d+)', stripped, re.IGNORECASE):
                zname = match.group(1).strip()
                znum = int(match.group(2))
                name_map[znum] = zname
            continue
        upper = stripped.upper()
        if upper.startswith("INTERNAL"):
            continue
        if upper.startswith("CONSTANT"):
            # CONSTANT <value> — fill current layer with value
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    val = int(parts[1])
                    row_text = " ".join([str(val)] * ncol)
                    for _ in range(nrow):
                        data_lines.append(row_text)
                except ValueError:
                    pass
            continue
        data_lines.append(stripped)

    # Parse integers from data lines
    zone_array = np.zeros((nlay, nrow, ncol), dtype=int)
    line_idx = 0
    for lay in range(nlay):
        for r in range(nrow):
            if line_idx >= len(data_lines):
                break
            values = []
            while len(values) < ncol and line_idx < len(data_lines):
                parts = data_lines[line_idx].replace(",", " ").replace("\t", " ").split()
                for p in parts:
                    try:
                        values.append(int(p))
                    except ValueError:
                        pass
                line_idx += 1
            for c in range(min(ncol, len(values))):
                zone_array[lay, r, c] = values[c]

    # Convert array to zone_layers
    unique_zones = set(int(v) for v in np.unique(zone_array) if v != 0)
    num_zones = len(unique_zones)

    # Build name map for missing names
    for znum in unique_zones:
        if znum not in name_map:
            name_map[znum] = f"Zone {znum}"

    zone_layers: dict[str, dict[str, list[int]]] = {}
    for lay in range(nlay):
        groups: dict[str, list[int]] = {}
        for r in range(nrow):
            for c in range(ncol):
                val = int(zone_array[lay, r, c])
                if val == 0:
                    continue
                zname = name_map[val]
                if zname not in groups:
                    groups[zname] = []
                groups[zname].append(r * ncol + c)
        if groups:
            zone_layers[str(lay)] = groups

    return zone_layers, num_zones


# ---------------------------------------------------------------------------
# Import: GeoJSON → zone_layers (overlay on model grid)
# ---------------------------------------------------------------------------

# Fields to auto-detect as zone identifier (case-insensitive)
_ZONE_FIELD_CANDIDATES = [
    "zone", "zone_id", "zone_num", "zone_name", "zoneid", "zonenum",
    "zonename", "fid", "id", "name", "class", "category",
]


def geojson_to_zone_layers(
    geojson_data: dict,
    *,
    nrow: int,
    ncol: int,
    nlay: int,
    delr: list[float] | None = None,
    delc: list[float] | None = None,
    xoff: float = 0.0,
    yoff: float = 0.0,
    angrot: float = 0.0,
    grid_type: str = "structured",
    grid_geometry: dict | None = None,
    zone_field: str | None = None,
    target_layer: int = 0,
    apply_all_layers: bool = False,
) -> tuple[dict[str, dict[str, list[int]]], int]:
    """Overlay GeoJSON features on the model grid to assign cells to zones.

    Returns (zone_layers, num_zones).
    Raises ValueError with available_fields if zone_field auto-detection fails.
    """
    try:
        from shapely.geometry import Point, shape
        from shapely.strtree import STRtree
    except ImportError:
        raise RuntimeError("shapely is required for GeoJSON import")

    # Extract features
    features = geojson_data.get("features", [])
    if not features:
        if geojson_data.get("type") == "Feature":
            features = [geojson_data]
        elif geojson_data.get("type") in ("Polygon", "MultiPolygon"):
            features = [{"type": "Feature", "geometry": geojson_data, "properties": {}}]

    if not features:
        return {}, 0

    # Auto-detect zone field
    available_fields: list[str] = []
    if features[0].get("properties"):
        available_fields = list(features[0]["properties"].keys())

    if not zone_field:
        for candidate in _ZONE_FIELD_CANDIDATES:
            for field in available_fields:
                if field.lower() == candidate.lower():
                    zone_field = field
                    break
            if zone_field:
                break

    if not zone_field and available_fields:
        raise ValueError(json.dumps({
            "detail": "Could not auto-detect zone field from GeoJSON properties",
            "available_fields": available_fields,
        }))

    # Build cell centroids
    is_structured = grid_type == "structured"
    if is_structured:
        if not delr or not delc:
            raise ValueError("delr and delc required for structured grid")
        centroids = _build_structured_centroids(delr, delc, xoff, yoff, angrot)
        total_cells = nrow * ncol
    else:
        if not grid_geometry:
            raise ValueError("grid_geometry required for polygon grid")
        centroids = _load_polygon_grid_centroids(grid_geometry)
        total_cells = len(centroids)

    # Build shapely Points and STRtree for spatial indexing
    cell_points = [Point(cx, cy) for cx, cy in centroids]
    tree = STRtree(cell_points)

    # Process features: assign cells to zones
    zone_assignments: dict[int, int] = {}  # cell_index → zone_number
    zone_names: dict[int, str] = {}        # zone_number → zone_name
    zone_counter = 0
    zone_value_to_num: dict[str, int] = {}

    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            poly = shape(geom)
        except Exception:
            continue
        if poly.is_empty:
            continue

        # Determine zone value from feature properties
        if zone_field and feat.get("properties"):
            zone_val = str(feat["properties"].get(zone_field, ""))
        else:
            zone_val = f"Feature_{features.index(feat)}"

        if zone_val not in zone_value_to_num:
            zone_counter += 1
            zone_value_to_num[zone_val] = zone_counter
            # Try to create a nice zone name
            try:
                num = int(zone_val)
                zone_names[zone_counter] = f"Zone {num}"
            except (ValueError, TypeError):
                zone_names[zone_counter] = f"Zone {zone_counter}"

        znum = zone_value_to_num[zone_val]

        # Use STRtree to find candidate cells whose centroids fall in the polygon
        try:
            from shapely.prepared import prep
            prepared_poly = prep(poly)
        except Exception:
            prepared_poly = None

        candidate_indices = tree.query(poly)
        for idx in candidate_indices:
            if idx >= total_cells:
                continue
            pt = cell_points[idx]
            if prepared_poly:
                if prepared_poly.contains(pt):
                    zone_assignments[idx] = znum
            elif poly.contains(pt):
                zone_assignments[idx] = znum

    num_zones = zone_counter

    # Convert to zone_layers format
    target_layers = list(range(nlay)) if apply_all_layers else [target_layer]
    zone_layers: dict[str, dict[str, list[int]]] = {}

    for lay in target_layers:
        groups: dict[str, list[int]] = {}
        for ci, znum in zone_assignments.items():
            zname = zone_names.get(znum, f"Zone {znum}")
            if zname not in groups:
                groups[zname] = []
            groups[zname].append(ci)
        if groups:
            zone_layers[str(lay)] = groups

    return zone_layers, num_zones
