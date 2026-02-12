"""3D mesh generation service for MODFLOW grids."""

import struct
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import flopy
import numpy as np

from app.config import get_settings
from app.services.path_normalizer import normalize_all_model_files
from app.services.storage import get_storage_service

settings = get_settings()

# Per-project lock to prevent concurrent model loads that cause OOM
_model_load_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_project_lock(project_id: str) -> threading.Lock:
    """Get or create a per-project lock for model loading."""
    with _locks_lock:
        if project_id not in _model_load_locks:
            _model_load_locks[project_id] = threading.Lock()
        return _model_load_locks[project_id]


def _get_mf6_grid_array(mf_array, shape: Union[int, tuple]) -> np.ndarray:
    """
    Extract array data from MF6 array object.

    Handles CONSTANT values by expanding them to full arrays.
    Uses get_data() which properly handles CONSTANT arrays in FloPy.
    """
    try:
        # The most reliable way to get MF6 array data
        # get_data() should expand CONSTANT values
        if hasattr(mf_array, 'get_data'):
            data = mf_array.get_data()
            if data is not None:
                arr = np.array(data, dtype=np.float64)
                # Ensure correct shape
                if isinstance(shape, int) and arr.size >= shape:
                    return arr.flatten()[:shape]
                elif isinstance(shape, tuple):
                    if arr.shape == shape:
                        return arr
                    # Try to reshape
                    try:
                        return arr.reshape(shape)
                    except ValueError:
                        pass
                return arr

        # Try direct data access
        if hasattr(mf_array, 'data') and mf_array.data is not None:
            arr = np.array(mf_array.data, dtype=np.float64)
            return arr

        # Try array property
        if hasattr(mf_array, 'array') and mf_array.array is not None:
            return np.array(mf_array.array, dtype=np.float64)

    except Exception as e:
        print(f"Warning extracting array: {e}")

    # Return default array filled with a recognizable default
    if isinstance(shape, int):
        return np.ones(shape, dtype=np.float64) * 100.0  # Default value
    return np.ones(shape, dtype=np.float64) * 100.0


@dataclass
class GridMesh:
    """3D mesh data for a MODFLOW grid."""

    # Grid type: 0=structured, 1=unstructured
    grid_type: int

    # Grid dimensions
    nlay: int
    nrow: int
    ncol: int

    # Cell center coordinates (nlay * nrow * ncol, 3)
    centers: np.ndarray

    # Top and bottom elevations
    top: np.ndarray  # (nrow, ncol)
    botm: np.ndarray  # (nlay, nrow, ncol)

    # Structured grids only: delr/delc for lossless vertex reconstruction
    delr: Optional[np.ndarray] = None  # (ncol,)
    delc: Optional[np.ndarray] = None  # (nrow,)

    # Unstructured grids only: actual vertices (ncells, 8, 3)
    vertices: Optional[np.ndarray] = None

    def to_binary(self) -> bytes:
        """
        Serialize mesh data to binary format for efficient transfer.

        Structured (grid_type=0):
          grid_type(i4) + nlay(i4) + nrow(i4) + ncol(i4) +
          centers(ncells*3 f32) + delr(ncol f32) + delc(nrow f32) +
          top(nrow*ncol f32) + botm(nlay*nrow*ncol f32)

        Unstructured (grid_type=1):
          grid_type(i4) + nlay(i4) + nrow(i4) + ncol(i4) +
          centers(ncells*3 f32) + vertices(ncells*8*3 f32) +
          top(nrow*ncol f32) + botm(nlay*nrow*ncol f32)
        """
        # Header: grid_type + dimensions
        header = struct.pack("<iiii", self.grid_type, self.nlay, self.nrow, self.ncol)

        # Centers (x, y, z for each cell)
        centers_flat = self.centers.astype(np.float32).tobytes()

        if self.grid_type == 0:
            # Structured: send delr + delc
            delr = self.delr.astype(np.float32).tobytes()
            delc = self.delc.astype(np.float32).tobytes()
            grid_data = delr + delc
        else:
            # Unstructured: send actual vertices
            verts = self.vertices.astype(np.float32).tobytes()
            grid_data = verts

        # Elevations
        top = self.top.astype(np.float32).tobytes()
        botm = self.botm.astype(np.float32).tobytes()

        return header + centers_flat + grid_data + top + botm


def load_model_from_storage(project_id: str, storage_path: str) -> Optional[object]:
    """
    Load a MODFLOW model from MinIO storage.

    Downloads files to a temp directory and loads with FloPy.
    Uses a per-project lock to prevent concurrent loads that could cause OOM.
    """
    lock = _get_project_lock(project_id)
    with lock:
        storage = get_storage_service()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download all model files
            files = storage.list_objects(
                settings.minio_bucket_models,
                prefix=storage_path,
                recursive=True,
            )

            for obj_name in files:
                # Get relative path within model directory
                rel_path = obj_name[len(storage_path):].lstrip("/")
                if not rel_path:
                    continue

                local_path = temp_path / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                file_data = storage.download_file(settings.minio_bucket_models, obj_name)
                local_path.write_bytes(file_data)

            # Detect and load model
            return load_model_from_directory(temp_path)


def load_and_extract_array(
    project_id: str, storage_path: str, array_name: str
) -> Optional[np.ndarray]:
    """
    Load model from MinIO and extract an array within the same temp directory.

    FloPy uses lazy loading for OPEN/CLOSE external arrays, so the model
    files must remain on disk while get_data() is called.
    """
    lock = _get_project_lock(project_id)
    with lock:
        storage = get_storage_service()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            files = storage.list_objects(
                settings.minio_bucket_models,
                prefix=storage_path,
                recursive=True,
            )

            for obj_name in files:
                rel_path = obj_name[len(storage_path):].lstrip("/")
                if not rel_path:
                    continue
                local_path = temp_path / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                file_data = storage.download_file(settings.minio_bucket_models, obj_name)
                local_path.write_bytes(file_data)

            model = load_model_from_directory(temp_path)
            if model is None:
                return None

            return get_array_data(model, array_name)


def load_model_from_directory(model_dir: Path) -> Optional[object]:
    """Load a MODFLOW model from a directory."""
    # Normalize paths in model files (backslash → forward slash)
    normalize_all_model_files(model_dir)

    # Check for MF6
    if (model_dir / "mfsim.nam").exists():
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=str(model_dir),
            verbosity_level=0,
        )
        model_names = sim.model_names
        if model_names:
            return sim.get_model(model_names[0])
        return None

    # Check for MODFLOW-USG first (look for DISU in nam file)
    for f in model_dir.iterdir():
        if f.suffix.lower() == ".nam" and f.name.lower() != "mfsim.nam":
            try:
                content = f.read_text().upper()
                if "DISU" in content or "CLN" in content or "GNC" in content:
                    # This is a USG model - load with flopy.mfusg
                    return _load_usg_model(model_dir, f)
            except Exception:
                pass

    # Check for MF2005/NWT
    for f in model_dir.iterdir():
        if f.suffix.lower() == ".nam" and f.name.lower() != "mfsim.nam":
            return flopy.modflow.Modflow.load(
                f.name,
                model_ws=str(model_dir),
                check=False,
                verbose=False,
            )

    return None


def _load_usg_model(model_dir: Path, nam_file: Path) -> Optional[object]:
    """Load a MODFLOW-USG model with gridspec support."""
    import flopy.mfusg as mfusg

    # Look for gridspec file (.gsf extension or similar)
    gridspec_file = None
    for ext in ['.gsf', '.gridspec', '.gspec']:
        for f in model_dir.iterdir():
            if f.suffix.lower() == ext:
                gridspec_file = f
                break
        if gridspec_file:
            break

    try:
        # Load USG model
        model = mfusg.MfUsg.load(
            nam_file.name,
            model_ws=str(model_dir),
            check=False,
            verbose=False,
        )

        # If gridspec exists, try to load it for grid geometry
        if gridspec_file and model.disu is not None:
            try:
                # Read gridspec to get node coordinates
                grid_data = _parse_gridspec(gridspec_file)
                if grid_data:
                    # Attach grid data to model for later use
                    model._gridspec_data = grid_data
            except Exception as e:
                print(f"Warning: Could not parse gridspec file: {e}")

        return model

    except Exception as e:
        print(f"Error loading USG model: {e}")
        return None


def _parse_gridspec(gridspec_file: Path) -> Optional[dict]:
    """
    Parse a MODFLOW-USG gridspec (.gsf) file.

    Format:
      Line 0: "UNSTRUCTURED" (or grid type string)
      Line 1: nnodes nlay ...
      Line 2: nvert (number of 2D polygon vertices)
      Lines 3..3+nvert-1: vertex_x vertex_y value (only x,y used for geometry)
      Lines 3+nvert..end: cell connectivity, one per node:
        cell_id center_x center_y center_z layer nverts v1 v2 ... vn
    """
    try:
        lines = gridspec_file.read_text().strip().split('\n')
        if len(lines) < 4:
            return None

        # Line 0: grid type
        grid_type_str = lines[0].strip()

        # Line 1: nnodes, nlay, ...
        header_parts = lines[1].split()
        nnodes = int(header_parts[0])
        nlay = int(header_parts[1])

        # Line 2: number of vertices
        nvert = int(lines[2].strip())

        # Lines 3..3+nvert-1: vertex coordinates (x, y; third column ignored)
        vertices_2d = []
        for i in range(3, 3 + nvert):
            parts = lines[i].split()
            vertices_2d.append((float(parts[0]), float(parts[1])))

        # Lines 3+nvert..end: cell connectivity
        cell_start = 3 + nvert
        cells = []
        for i in range(cell_start, min(cell_start + nnodes, len(lines))):
            parts = lines[i].split()
            if len(parts) < 6:
                continue
            cell_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            cz = float(parts[3])
            layer = int(parts[4])
            nverts = int(parts[5])
            vert_indices = [int(v) for v in parts[6:6 + nverts]]  # 1-based
            cells.append({
                'id': cell_id,
                'x': cx,
                'y': cy,
                'z': cz,
                'layer': layer,
                'vertex_indices': vert_indices,
            })

        return {
            'file': str(gridspec_file),
            'grid_type': grid_type_str,
            'nnodes': nnodes,
            'nlay': nlay,
            'nvert': nvert,
            'vertices_2d': vertices_2d,
            'cells': cells,
        }

    except Exception as e:
        print(f"Error parsing gridspec: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_usg_mesh(model) -> Optional[GridMesh]:
    """
    Generate 3D mesh data from a MODFLOW-USG model.

    USG models use unstructured grids (DISU package).
    Sends actual cell vertices so the frontend can render exact geometry.
    """
    try:
        disu = model.disu
        if disu is None:
            if model.dis is not None:
                return None  # Will fall through to standard mesh generation
            return None

        nlay = disu.nlay
        nodes_per_layer = disu.nodes // nlay if nlay > 0 else disu.nodes
        nrow = 1
        ncol = nodes_per_layer

        # Get top and bottom elevations from DISU
        try:
            top_arr = np.array(disu.top.array).flatten()
            botm_arr = np.array(disu.bot.array).flatten()
        except Exception:
            try:
                grid = model.modelgrid
                top_arr = np.array(grid.top).flatten()
                botm_arr = np.array(grid.botm).flatten()
            except Exception:
                top_arr = np.zeros(nodes_per_layer)
                botm_arr = np.ones(nodes_per_layer) * -10

        # Reshape top and botm for the binary format
        top_2d = top_arr[:nodes_per_layer].reshape(1, -1)
        if len(botm_arr) >= nlay * nodes_per_layer:
            botm_3d = botm_arr[:nlay * nodes_per_layer].reshape(nlay, 1, nodes_per_layer)
        else:
            botm_3d = np.zeros((nlay, 1, nodes_per_layer))
            botm_3d[0, 0, :] = botm_arr[:nodes_per_layer] if len(botm_arr) >= nodes_per_layer else -10

        # Fix inverted cells (top < botm) — these are typically inactive cells
        # with placeholder elevation values that create giant visual spikes.
        # Set top = botm so they render as zero-thickness (invisible).
        for k in range(nlay):
            cell_top = top_2d[0] if k == 0 else botm_3d[k - 1, 0]
            cell_bot = botm_3d[k, 0]
            inverted = cell_top < cell_bot
            if k == 0:
                top_2d[0, inverted] = cell_bot[inverted]
            else:
                botm_3d[k - 1, 0, inverted] = cell_bot[inverted]

        # Try to get cell geometry from gridspec data
        gsf = getattr(model, '_gridspec_data', None)
        cell_bboxes = None  # per-cell (x0, x1, y0, y1) from GSF polygons
        cell_centers_xy = None  # per-cell (cx, cy) from GSF

        if gsf and 'cells' in gsf and 'vertices_2d' in gsf:
            gsf_verts = gsf['vertices_2d']  # list of (x, y)
            gsf_cells = gsf['cells']  # list of dicts with vertex_indices

            # Only use layer-1 cells for geometry (same x/y for all layers),
            # sorted by cell ID to match node ordering in head output
            layer1_cells = sorted(
                [c for c in gsf_cells if c['layer'] == 1],
                key=lambda c: c['id'],
            )
            if len(layer1_cells) == nodes_per_layer:
                cell_bboxes = []
                cell_centers_xy = []
                for cell in layer1_cells:
                    cx, cy = cell['x'], cell['y']
                    cell_centers_xy.append((cx, cy))

                    # Get polygon bounding box from vertex indices (1-based into vertices_2d)
                    vis = cell['vertex_indices']
                    xs = [gsf_verts[vi - 1][0] for vi in vis if 0 < vi <= len(gsf_verts)]
                    ys = [gsf_verts[vi - 1][1] for vi in vis if 0 < vi <= len(gsf_verts)]
                    if xs and ys:
                        cell_bboxes.append((min(xs), max(xs), min(ys), max(ys)))
                    else:
                        # Fallback for this cell
                        cell_bboxes.append((cx - 50, cx + 50, cy - 50, cy + 50))

        # If no GSF data, try FloPy grid
        if cell_centers_xy is None:
            try:
                grid = model.modelgrid
                xcenters = np.array(grid.xcellcenters).flatten()[:nodes_per_layer]
                ycenters = np.array(grid.ycellcenters).flatten()[:nodes_per_layer]
                cell_centers_xy = list(zip(xcenters, ycenters))
            except Exception:
                # Generate placeholder centers (grid pattern)
                side = int(np.ceil(np.sqrt(nodes_per_layer)))
                xs = np.tile(np.arange(side), side)[:nodes_per_layer].astype(float) * 100
                ys = np.repeat(np.arange(side), side)[:nodes_per_layer].astype(float) * 100
                cell_centers_xy = list(zip(xs, ys))

        # If no bounding boxes from GSF, estimate from nearest neighbors
        if cell_bboxes is None:
            xc = np.array([c[0] for c in cell_centers_xy])
            yc = np.array([c[1] for c in cell_centers_xy])
            # Compute per-cell size from nearest-neighbor distance
            if nodes_per_layer > 1:
                coords_2d = np.column_stack([xc, yc])
                # Simple O(n) nearest-neighbor via sorted x-coordinates
                sort_idx = np.argsort(xc)
                nn_dist = np.full(nodes_per_layer, np.inf)
                for ii in range(len(sort_idx)):
                    for jj in range(max(0, ii - 10), min(len(sort_idx), ii + 11)):
                        if ii == jj:
                            continue
                        a, b = sort_idx[ii], sort_idx[jj]
                        d = np.sqrt((xc[a] - xc[b])**2 + (yc[a] - yc[b])**2)
                        nn_dist[a] = min(nn_dist[a], d)
                cell_bboxes = []
                for j in range(nodes_per_layer):
                    half = nn_dist[j] / 2 if np.isfinite(nn_dist[j]) else 50.0
                    cell_bboxes.append((xc[j] - half, xc[j] + half, yc[j] - half, yc[j] + half))
            else:
                cell_bboxes = [(cell_centers_xy[0][0] - 50, cell_centers_xy[0][0] + 50,
                                cell_centers_xy[0][1] - 50, cell_centers_xy[0][1] + 50)]

        # Build centers array for all cells
        ncells = nlay * nodes_per_layer
        centers = np.zeros((ncells, 3), dtype=np.float32)
        idx = 0
        for k in range(nlay):
            for j in range(nodes_per_layer):
                cx, cy = cell_centers_xy[j]
                centers[idx, 0] = cx
                centers[idx, 1] = cy
                cell_top = top_2d[0, j] if k == 0 else botm_3d[k - 1, 0, j]
                cell_bot = botm_3d[k, 0, j]
                centers[idx, 2] = (cell_top + cell_bot) / 2
                idx += 1

        # Build vertices (8 per cell) using bounding boxes for x/y and DISU for z
        vertices = np.zeros((ncells, 8, 3), dtype=np.float32)
        idx = 0
        for k in range(nlay):
            for j in range(nodes_per_layer):
                z_top = top_2d[0, j] if k == 0 else botm_3d[k - 1, 0, j]
                z_bot = botm_3d[k, 0, j]
                x0, x1, y0, y1 = cell_bboxes[j]

                # 8 vertices: bottom face (0-3), top face (4-7)
                vertices[idx, 0] = [x0, y0, z_bot]
                vertices[idx, 1] = [x1, y0, z_bot]
                vertices[idx, 2] = [x1, y1, z_bot]
                vertices[idx, 3] = [x0, y1, z_bot]
                vertices[idx, 4] = [x0, y0, z_top]
                vertices[idx, 5] = [x1, y0, z_top]
                vertices[idx, 6] = [x1, y1, z_top]
                vertices[idx, 7] = [x0, y1, z_top]
                idx += 1

        return GridMesh(
            grid_type=1,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            centers=centers,
            vertices=vertices,
            top=top_2d,
            botm=botm_3d,
        )

    except Exception as e:
        print(f"Error generating USG mesh: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_mf6_vertex_mesh(model, dis_pkg) -> Optional[GridMesh]:
    """
    Generate 3D mesh for MF6 DISV or DISU grids.

    Uses bounding-box vertices per cell (same approach as USG mesh)
    so the frontend can render using gridType=1 path.
    """
    try:
        grid = model.modelgrid

        # Get dimensions
        if hasattr(dis_pkg, 'nlay'):
            nlay = dis_pkg.nlay.data
        else:
            nlay = 1

        if hasattr(dis_pkg, 'ncpl'):
            ncpl = dis_pkg.ncpl.data
        elif hasattr(dis_pkg, 'nodes'):
            ncpl = dis_pkg.nodes.data // nlay if nlay > 0 else dis_pkg.nodes.data
        else:
            ncpl = grid.ncpl if hasattr(grid, 'ncpl') else 1

        nrow = 1
        ncol = ncpl

        # Get top/botm arrays
        try:
            top_data = dis_pkg.top.get_data()
            top_arr = np.array(top_data, dtype=np.float64).flatten()[:ncpl]
        except Exception:
            top_arr = np.zeros(ncpl)

        try:
            botm_data = dis_pkg.botm.get_data()
            botm_arr = np.array(botm_data, dtype=np.float64).flatten()
        except Exception:
            botm_arr = np.ones(nlay * ncpl) * -10

        top_2d = top_arr.reshape(1, -1)
        if len(botm_arr) >= nlay * ncpl:
            botm_3d = botm_arr[:nlay * ncpl].reshape(nlay, 1, ncpl)
        else:
            botm_3d = np.zeros((nlay, 1, ncpl))
            botm_3d[0, 0, :] = botm_arr[:ncpl] if len(botm_arr) >= ncpl else -10

        # Fix inverted cells
        for k in range(nlay):
            cell_top = top_2d[0] if k == 0 else botm_3d[k - 1, 0]
            cell_bot = botm_3d[k, 0]
            inverted = cell_top < cell_bot
            if k == 0:
                top_2d[0, inverted] = cell_bot[inverted]
            else:
                botm_3d[k - 1, 0, inverted] = cell_bot[inverted]

        # Get cell centers and bounding boxes from modelgrid
        cell_centers_xy = None
        cell_bboxes = None

        try:
            # FloPy modelgrid provides cell vertices for DISV grids
            xcenters = np.array(grid.xcellcenters).flatten()[:ncpl]
            ycenters = np.array(grid.ycellcenters).flatten()[:ncpl]
            cell_centers_xy = list(zip(xcenters, ycenters))

            # Try to get actual cell vertices for bounding boxes
            if hasattr(grid, 'get_cell_vertices'):
                cell_bboxes = []
                for j in range(ncpl):
                    try:
                        verts = grid.get_cell_vertices(j)
                        xs = [v[0] for v in verts]
                        ys = [v[1] for v in verts]
                        cell_bboxes.append((min(xs), max(xs), min(ys), max(ys)))
                    except Exception:
                        # Fallback for this cell
                        cx, cy = cell_centers_xy[j]
                        cell_bboxes.append((cx - 50, cx + 50, cy - 50, cy + 50))
        except Exception:
            pass

        if cell_centers_xy is None:
            # Generate placeholder centers
            side = int(np.ceil(np.sqrt(ncpl)))
            xs = np.tile(np.arange(side), side)[:ncpl].astype(float) * 100
            ys = np.repeat(np.arange(side), side)[:ncpl].astype(float) * 100
            cell_centers_xy = list(zip(xs, ys))

        if cell_bboxes is None:
            # Estimate from nearest neighbors
            xc = np.array([c[0] for c in cell_centers_xy])
            yc = np.array([c[1] for c in cell_centers_xy])
            if ncpl > 1:
                sort_idx = np.argsort(xc)
                nn_dist = np.full(ncpl, np.inf)
                for ii in range(len(sort_idx)):
                    for jj in range(max(0, ii - 10), min(len(sort_idx), ii + 11)):
                        if ii == jj:
                            continue
                        a, b = sort_idx[ii], sort_idx[jj]
                        d = np.sqrt((xc[a] - xc[b])**2 + (yc[a] - yc[b])**2)
                        nn_dist[a] = min(nn_dist[a], d)
                cell_bboxes = []
                for j in range(ncpl):
                    half = nn_dist[j] / 2 if np.isfinite(nn_dist[j]) else 50.0
                    cell_bboxes.append((xc[j] - half, xc[j] + half, yc[j] - half, yc[j] + half))
            else:
                cell_bboxes = [(cell_centers_xy[0][0] - 50, cell_centers_xy[0][0] + 50,
                                cell_centers_xy[0][1] - 50, cell_centers_xy[0][1] + 50)]

        # Build centers array
        ncells = nlay * ncpl
        centers = np.zeros((ncells, 3), dtype=np.float32)
        idx = 0
        for k in range(nlay):
            for j in range(ncpl):
                cx, cy = cell_centers_xy[j]
                centers[idx, 0] = cx
                centers[idx, 1] = cy
                cell_top = top_2d[0, j] if k == 0 else botm_3d[k - 1, 0, j]
                cell_bot = botm_3d[k, 0, j]
                centers[idx, 2] = (cell_top + cell_bot) / 2
                idx += 1

        # Build vertices (8 per cell)
        vertices = np.zeros((ncells, 8, 3), dtype=np.float32)
        idx = 0
        for k in range(nlay):
            for j in range(ncpl):
                z_top = top_2d[0, j] if k == 0 else botm_3d[k - 1, 0, j]
                z_bot = botm_3d[k, 0, j]
                x0, x1, y0, y1 = cell_bboxes[j]

                vertices[idx, 0] = [x0, y0, z_bot]
                vertices[idx, 1] = [x1, y0, z_bot]
                vertices[idx, 2] = [x1, y1, z_bot]
                vertices[idx, 3] = [x0, y1, z_bot]
                vertices[idx, 4] = [x0, y0, z_top]
                vertices[idx, 5] = [x1, y0, z_top]
                vertices[idx, 6] = [x1, y1, z_top]
                vertices[idx, 7] = [x0, y1, z_top]
                idx += 1

        return GridMesh(
            grid_type=1,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            centers=centers,
            vertices=vertices,
            top=top_2d,
            botm=botm_3d,
        )

    except Exception as e:
        print(f"Error generating MF6 vertex mesh: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_grid_mesh(model) -> Optional[GridMesh]:
    """
    Generate 3D mesh data from a FloPy model.

    Works with MF6, MF2005/NWT, and MODFLOW-USG models.
    """
    try:
        # Check model type
        is_mf6 = model.__class__.__module__.startswith('flopy.mf6')
        is_usg = model.__class__.__module__.startswith('flopy.mfusg')

        # Handle MODFLOW-USG models
        if is_usg:
            return _generate_usg_mesh(model)

        if is_mf6:
            # Get discretization package — get_package("DIS") returns
            # whatever dis package exists (DIS, DISV, or DISU)
            dis_pkg = model.get_package("DIS")
            pkg_type = getattr(dis_pkg, 'package_type', '').upper() if dis_pkg else ''

            if pkg_type == "DIS":
                nlay = dis_pkg.nlay.data
                nrow = dis_pkg.nrow.data
                ncol = dis_pkg.ncol.data

                # Get arrays - handle CONSTANT values
                delr = _get_mf6_grid_array(dis_pkg.delr, ncol)
                delc = _get_mf6_grid_array(dis_pkg.delc, nrow)
                top = _get_mf6_grid_array(dis_pkg.top, (nrow, ncol))
                botm = _get_mf6_grid_array(dis_pkg.botm, (nlay, nrow, ncol))
            elif pkg_type in ("DISV", "DISU"):
                return _generate_mf6_vertex_mesh(model, dis_pkg)
            else:
                return None
        else:
            # Classic MODFLOW (MF2005/NWT)
            grid = model.modelgrid
            nlay = grid.nlay
            nrow = grid.nrow
            ncol = grid.ncol
            delr = np.array(grid.delr)
            delc = np.array(grid.delc)
            top = np.array(grid.top)
            botm = np.array(grid.botm)

        # Calculate cell centers
        # X coordinates (column centers)
        x_edges = np.concatenate([[0], np.cumsum(delr)])
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2

        # Y coordinates (row centers) - note: rows go from top to bottom
        y_edges = np.concatenate([[0], np.cumsum(delc)])
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        # Flip so that row 0 is at the top (highest y)
        total_y = y_edges[-1]
        y_centers = total_y - y_centers

        # Calculate cell centers for all cells
        ncells = nlay * nrow * ncol
        centers = np.zeros((ncells, 3), dtype=np.float32)

        idx = 0
        for k in range(nlay):
            for i in range(nrow):
                for j in range(ncol):
                    centers[idx, 0] = x_centers[j]
                    centers[idx, 1] = y_centers[i]
                    cell_top = top[i, j] if k == 0 else botm[k - 1, i, j]
                    cell_bot = botm[k, i, j]
                    centers[idx, 2] = (cell_top + cell_bot) / 2
                    idx += 1

        return GridMesh(
            grid_type=0,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            centers=centers,
            delr=delr,
            delc=delc,
            top=top,
            botm=botm,
        )

    except Exception as e:
        print(f"Error generating grid mesh: {e}")
        return None


def get_array_data(model, array_name: str) -> Optional[np.ndarray]:
    """
    Extract a named array from the model.

    Supported arrays:
    - ibound/idomain: Active cell indicator
    - top: Top elevation
    - botm: Bottom elevation
    - hk/k: Horizontal hydraulic conductivity
    - vka/k33: Vertical hydraulic conductivity
    - ss: Specific storage
    - sy: Specific yield
    - strt/ic: Starting heads
    """
    array_name = array_name.lower()

    try:
        # Check model type
        is_mf6 = model.__class__.__module__.startswith('flopy.mf6')
        is_usg = model.__class__.__module__.startswith('flopy.mfusg')

        if is_mf6:
            return _get_mf6_array(model, array_name)
        elif is_usg:
            return _get_usg_array(model, array_name)
        else:
            return _get_mf2005_array(model, array_name)

    except Exception as e:
        print(f"Error getting array {array_name}: {e}")
        return None


def _get_mf6_array(model, array_name: str) -> Optional[np.ndarray]:
    """Get array from MF6 model."""
    # Get DIS package for grid info — get_package("DIS") returns
    # whatever dis package exists (DIS, DISV, or DISU)
    dis = model.get_package("DIS")
    if dis is None:
        return None
    pkg_type = getattr(dis, 'package_type', '').upper()
    if pkg_type == "DIS":
        nlay = dis.nlay.data
        nrow = dis.nrow.data
        ncol = dis.ncol.data
    elif pkg_type == "DISV":
        nlay = dis.nlay.data
        nrow = 1
        ncol = dis.ncpl.data
    elif pkg_type == "DISU":
        nlay = 1
        nrow = 1
        ncol = dis.nodes.data
    else:
        return None

    if array_name in ["ibound", "idomain"]:
        # IDOMAIN from DIS package
        if hasattr(dis, "idomain") and dis.idomain is not None:
            data = dis.idomain.get_data()
            if data is not None:
                arr = np.array(data, dtype=np.int32)
                target_shape = (nlay, nrow, ncol)
                if arr.shape == target_shape and np.any(arr != 0):
                    return arr
                # DISV/DISU may return (nlay, ncpl) instead of (nlay, 1, ncpl)
                if arr.size == nlay * nrow * ncol:
                    try:
                        arr = arr.reshape(target_shape)
                        if np.any(arr != 0):
                            return arr
                    except ValueError:
                        pass
        # Default: all active
        return np.ones((nlay, nrow, ncol), dtype=np.int32)

    elif array_name == "top":
        return _get_mf6_grid_array(dis.top, (nrow, ncol))

    elif array_name == "botm":
        return _get_mf6_grid_array(dis.botm, (nlay, nrow, ncol))

    elif array_name in ["hk", "k", "k11"]:
        npf = model.get_package("NPF")
        if npf is not None and hasattr(npf, "k"):
            return _get_mf6_grid_array(npf.k, (nlay, nrow, ncol))
        return None

    elif array_name in ["vka", "k33"]:
        npf = model.get_package("NPF")
        if npf is not None and hasattr(npf, "k33"):
            return _get_mf6_grid_array(npf.k33, (nlay, nrow, ncol))
        return None

    elif array_name == "ss":
        sto = model.get_package("STO")
        if sto is not None and hasattr(sto, "ss"):
            return _get_mf6_grid_array(sto.ss, (nlay, nrow, ncol))
        return None

    elif array_name == "sy":
        sto = model.get_package("STO")
        if sto is not None and hasattr(sto, "sy"):
            return _get_mf6_grid_array(sto.sy, (nlay, nrow, ncol))
        return None

    elif array_name in ["strt", "ic"]:
        ic = model.get_package("IC")
        if ic is not None and hasattr(ic, "strt"):
            return _get_mf6_grid_array(ic.strt, (nlay, nrow, ncol))
        return None

    return None


def _get_mf2005_array(model, array_name: str) -> Optional[np.ndarray]:
    """Get array from MF2005/NWT model."""
    if array_name == "ibound":
        if model.bas6 is not None:
            return np.array(model.bas6.ibound.array)
        return None

    elif array_name in ["idomain"]:
        # MF2005 uses IBOUND
        if model.bas6 is not None:
            return np.array(model.bas6.ibound.array)
        return None

    elif array_name == "top":
        if model.dis is not None:
            return np.array(model.dis.top.array)
        return None

    elif array_name == "botm":
        if model.dis is not None:
            return np.array(model.dis.botm.array)
        return None

    elif array_name in ["hk", "k"]:
        # Check LPF, UPW first (they use 'hk')
        for pkg_name in ["lpf", "upw"]:
            pkg = getattr(model, pkg_name, None)
            if pkg is not None and hasattr(pkg, "hk"):
                return np.array(pkg.hk.array)
        # BCF6 uses 'hy' for horizontal hydraulic conductivity
        bcf = getattr(model, "bcf6", None)
        if bcf is not None and hasattr(bcf, "hy"):
            arr = bcf.hy.array
            if arr is not None:
                return np.array(arr)
        return None

    elif array_name == "vka":
        # Check LPF, UPW first (they use 'vka')
        for pkg_name in ["lpf", "upw"]:
            pkg = getattr(model, pkg_name, None)
            if pkg is not None and hasattr(pkg, "vka"):
                return np.array(pkg.vka.array)
        # BCF6 uses 'vcont' for vertical leakance
        bcf = getattr(model, "bcf6", None)
        if bcf is not None and hasattr(bcf, "vcont"):
            arr = bcf.vcont.array
            if arr is not None:
                return np.array(arr)
        return None

    elif array_name == "ss":
        # Check LPF, UPW first (they use 'ss')
        for pkg_name in ["lpf", "upw"]:
            pkg = getattr(model, pkg_name, None)
            if pkg is not None and hasattr(pkg, "ss"):
                return np.array(pkg.ss.array)
        # BCF6 uses 'sf1' for specific storage
        bcf = getattr(model, "bcf6", None)
        if bcf is not None and hasattr(bcf, "sf1"):
            arr = bcf.sf1.array
            if arr is not None:
                return np.array(arr)
        return None

    elif array_name == "sy":
        # Check LPF, UPW first (they use 'sy')
        for pkg_name in ["lpf", "upw"]:
            pkg = getattr(model, pkg_name, None)
            if pkg is not None and hasattr(pkg, "sy"):
                return np.array(pkg.sy.array)
        # BCF6 uses 'sf2' for specific yield
        bcf = getattr(model, "bcf6", None)
        if bcf is not None and hasattr(bcf, "sf2"):
            arr = bcf.sf2.array
            if arr is not None:
                return np.array(arr)
        return None

    elif array_name in ["strt", "ic"]:
        if model.bas6 is not None:
            return np.array(model.bas6.strt.array)
        return None

    return None


def _get_usg_array(model, array_name: str) -> Optional[np.ndarray]:
    """Get array from MODFLOW-USG model."""
    # Get dimensions from DISU
    disu = model.disu
    if disu is None:
        # Fall back to MF2005 style if no DISU
        return _get_mf2005_array(model, array_name)

    nlay = disu.nlay
    nodes = disu.nodes
    nodes_per_layer = nodes // nlay if nlay > 0 else nodes

    if array_name in ["ibound", "idomain"]:
        # IBOUND from BAS package
        if model.bas6 is not None and hasattr(model.bas6, 'ibound'):
            arr = np.array(model.bas6.ibound.array).flatten()
            # Reshape for visualization: (nlay, 1, nodes_per_layer)
            if len(arr) >= nodes:
                return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
        # Default: all active
        return np.ones((nlay, 1, nodes_per_layer), dtype=np.int32)

    elif array_name == "top":
        try:
            arr = np.array(disu.top.array).flatten()
            return arr[:nodes_per_layer].reshape(1, nodes_per_layer)
        except Exception:
            return None

    elif array_name == "botm":
        try:
            arr = np.array(disu.bot.array).flatten()
            if len(arr) >= nodes:
                return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
            return arr.reshape(-1, 1, nodes_per_layer)
        except Exception:
            return None

    elif array_name in ["hk", "k"]:
        # Check LPF, UPW first (they use 'hk')
        for pkg_name in ["lpf", "upw"]:
            pkg = getattr(model, pkg_name, None)
            if pkg is not None and hasattr(pkg, "hk"):
                arr = np.array(pkg.hk.array).flatten()
                if len(arr) >= nodes:
                    return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
                return arr.reshape(-1, 1, nodes_per_layer)
        # BCF6 uses 'hy' for horizontal hydraulic conductivity
        bcf = getattr(model, "bcf6", None)
        if bcf is not None and hasattr(bcf, "hy"):
            data = bcf.hy.array
            if data is not None:
                arr = np.array(data).flatten()
                if len(arr) >= nodes:
                    return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
                return arr.reshape(-1, 1, nodes_per_layer)
        return None

    elif array_name == "vka":
        # Check LPF, UPW first (they use 'vka')
        for pkg_name in ["lpf", "upw"]:
            pkg = getattr(model, pkg_name, None)
            if pkg is not None and hasattr(pkg, "vka"):
                arr = np.array(pkg.vka.array).flatten()
                if len(arr) >= nodes:
                    return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
                return arr.reshape(-1, 1, nodes_per_layer)
        # BCF6 uses 'vcont' for vertical leakance
        bcf = getattr(model, "bcf6", None)
        if bcf is not None and hasattr(bcf, "vcont"):
            data = bcf.vcont.array
            if data is not None:
                arr = np.array(data).flatten()
                if len(arr) >= nodes:
                    return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
                return arr.reshape(-1, 1, nodes_per_layer)
        return None

    elif array_name == "ss":
        # Check LPF, UPW first (they use 'ss')
        for pkg_name in ["lpf", "upw"]:
            pkg = getattr(model, pkg_name, None)
            if pkg is not None and hasattr(pkg, "ss"):
                arr = np.array(pkg.ss.array).flatten()
                if len(arr) >= nodes:
                    return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
                return arr.reshape(-1, 1, nodes_per_layer)
        # BCF6 uses 'sf1' for specific storage
        bcf = getattr(model, "bcf6", None)
        if bcf is not None and hasattr(bcf, "sf1"):
            data = bcf.sf1.array
            if data is not None:
                arr = np.array(data).flatten()
                if len(arr) >= nodes:
                    return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
                return arr.reshape(-1, 1, nodes_per_layer)
        return None

    elif array_name == "sy":
        # Check LPF, UPW first (they use 'sy')
        for pkg_name in ["lpf", "upw"]:
            pkg = getattr(model, pkg_name, None)
            if pkg is not None and hasattr(pkg, "sy"):
                arr = np.array(pkg.sy.array).flatten()
                if len(arr) >= nodes:
                    return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
                return arr.reshape(-1, 1, nodes_per_layer)
        # BCF6 uses 'sf2' for specific yield
        bcf = getattr(model, "bcf6", None)
        if bcf is not None and hasattr(bcf, "sf2"):
            data = bcf.sf2.array
            if data is not None:
                arr = np.array(data).flatten()
                if len(arr) >= nodes:
                    return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
                return arr.reshape(-1, 1, nodes_per_layer)
        return None

    elif array_name in ["strt", "ic"]:
        if model.bas6 is not None and hasattr(model.bas6, 'strt'):
            arr = np.array(model.bas6.strt.array).flatten()
            if len(arr) >= nodes:
                return arr[:nodes].reshape(nlay, 1, nodes_per_layer)
            return arr.reshape(-1, 1, nodes_per_layer)
        return None

    return None


def get_list_package_data(model, package_name: str) -> Optional[dict]:
    """
    Extract data from list-based packages (HFB, SFR).

    Returns a dict with package statistics and base values that can be
    multiplied during calibration.

    Args:
        model: FloPy model object.
        package_name: Name of the package ('hfb' or 'sfr_cond').

    Returns:
        Dict with count, base_values, stats, or None if package not found.
    """
    try:
        is_mf6 = model.__class__.__module__.startswith('flopy.mf6')
        is_usg = model.__class__.__module__.startswith('flopy.mfusg')

        if package_name == "hfb":
            return _get_hfb_data(model, is_mf6, is_usg)
        elif package_name == "sfr_cond":
            return _get_sfr_cond_data(model, is_mf6, is_usg)
        else:
            return None

    except Exception as e:
        print(f"Error getting list package data {package_name}: {e}")
        return None


def _get_hfb_data(model, is_mf6: bool, is_usg: bool) -> Optional[dict]:
    """
    Extract HFB (Horizontal Flow Barrier) hydraulic characteristic values.

    MF6: model.get_package("HFB").stress_period_data['hydchr']
    MF2005: model.hfb6.hfb_data column 5 (hydchr)

    Returns dict with count, base_values (array), stats.
    """
    values = []

    if is_mf6:
        try:
            hfb = model.get_package("HFB")
            if hfb is None:
                return None

            # MF6 HFB stress_period_data contains the barrier data
            spd = hfb.stress_period_data
            if spd is None:
                return None

            # Get data for stress period 0 (or all periods)
            if hasattr(spd, 'get_data'):
                data = spd.get_data(0)
            else:
                data = spd.data.get(0, None)

            if data is None:
                return None

            # Extract hydchr values from the recarray
            if hasattr(data, 'hydchr'):
                values = data['hydchr'].astype(float).tolist()
            elif isinstance(data, np.ndarray) and 'hydchr' in data.dtype.names:
                values = data['hydchr'].astype(float).tolist()

        except Exception as e:
            print(f"Error extracting MF6 HFB data: {e}")
            return None

    elif is_usg:
        # MODFLOW-USG HFB is similar to MF2005
        try:
            hfb = getattr(model, 'hfb6', None)
            if hfb is None:
                return None

            hfb_data = hfb.hfb_data
            if hfb_data is None:
                return None

            # hfb_data is typically (nper, n_barriers, 6) or list of arrays
            # Column indices: layer1, row1, col1, layer2, row2, col2, hydchr
            # For USG: node1, node2, hydchr
            if isinstance(hfb_data, list):
                for period_data in hfb_data:
                    if period_data is not None and len(period_data) > 0:
                        # Get hydchr column (last column typically)
                        if hasattr(period_data, 'hydchr'):
                            values.extend(period_data['hydchr'].astype(float).tolist())
                        else:
                            arr = np.array(period_data)
                            if arr.ndim == 2:
                                values.extend(arr[:, -1].astype(float).tolist())
                        break
            elif isinstance(hfb_data, np.ndarray):
                if hfb_data.ndim == 2:
                    values = hfb_data[:, -1].astype(float).tolist()

        except Exception as e:
            print(f"Error extracting USG HFB data: {e}")
            return None

    else:
        # MF2005/NWT
        try:
            hfb = getattr(model, 'hfb6', None)
            if hfb is None:
                return None

            hfb_data = hfb.hfb_data
            if hfb_data is None:
                return None

            # hfb_data structure: list of arrays per stress period
            # Each array: [layer1, row1, col1, layer2, row2, col2, hydchr]
            if isinstance(hfb_data, list):
                for period_data in hfb_data:
                    if period_data is not None and len(period_data) > 0:
                        arr = np.array(period_data)
                        if arr.ndim == 2 and arr.shape[1] >= 7:
                            # Column 6 (0-indexed) is hydchr
                            values.extend(arr[:, 6].astype(float).tolist())
                        break
            elif isinstance(hfb_data, np.ndarray):
                if hfb_data.ndim == 2 and hfb_data.shape[1] >= 7:
                    values = hfb_data[:, 6].astype(float).tolist()

        except Exception as e:
            print(f"Error extracting MF2005 HFB data: {e}")
            return None

    if not values:
        return None

    values_arr = np.array(values)
    valid = values_arr[np.isfinite(values_arr) & (values_arr > 0)]

    return {
        "count": len(values),
        "base_values": values_arr,
        "stats": {
            "mean": round(float(np.mean(valid)), 6) if valid.size > 0 else None,
            "min": round(float(np.min(valid)), 6) if valid.size > 0 else None,
            "max": round(float(np.max(valid)), 6) if valid.size > 0 else None,
        },
    }


def _get_sfr_cond_data(model, is_mf6: bool, is_usg: bool) -> Optional[dict]:
    """
    Extract SFR (Streamflow Routing) streambed conductance values.

    MF6: model.get_package("SFR").packagedata - look for 'hk' or 'rhk'
    MF2005: model.sfr.reach_data['strhc1'] (streambed hydraulic conductivity)

    Returns dict with count, base_values (array), stats.
    """
    values = []

    if is_mf6:
        try:
            sfr = model.get_package("SFR")
            if sfr is None:
                return None

            # packagedata contains reach properties
            pkgdata = sfr.packagedata
            if pkgdata is None:
                return None

            if hasattr(pkgdata, 'get_data'):
                data = pkgdata.get_data()
            else:
                data = pkgdata.data

            if data is None:
                return None

            # Look for hydraulic conductivity field
            # MF6 SFR can have 'rhk' (hydraulic conductivity) or similar
            for field in ['rhk', 'hk', 'rbed_k', 'rbdk']:
                if hasattr(data, field):
                    values = data[field].astype(float).tolist()
                    break
                elif isinstance(data, np.ndarray) and field in data.dtype.names:
                    values = data[field].astype(float).tolist()
                    break

        except Exception as e:
            print(f"Error extracting MF6 SFR data: {e}")
            return None

    else:
        # MF2005/NWT/USG
        try:
            sfr = getattr(model, 'sfr', None)
            if sfr is None:
                return None

            # reach_data contains per-reach properties
            reach_data = getattr(sfr, 'reach_data', None)
            if reach_data is None:
                return None

            # strhc1 is streambed hydraulic conductivity
            if hasattr(reach_data, 'strhc1'):
                values = reach_data['strhc1'].astype(float).tolist()
            elif isinstance(reach_data, np.ndarray) and 'strhc1' in reach_data.dtype.names:
                values = reach_data['strhc1'].astype(float).tolist()

        except Exception as e:
            print(f"Error extracting MF2005 SFR data: {e}")
            return None

    if not values:
        return None

    values_arr = np.array(values)
    valid = values_arr[np.isfinite(values_arr) & (values_arr > 0)]

    return {
        "count": len(values),
        "base_values": values_arr,
        "stats": {
            "mean": round(float(np.mean(valid)), 6) if valid.size > 0 else None,
            "min": round(float(np.min(valid)), 6) if valid.size > 0 else None,
            "max": round(float(np.max(valid)), 6) if valid.size > 0 else None,
        },
    }
