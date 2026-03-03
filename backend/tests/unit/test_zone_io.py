"""Unit tests for zone_io service (zone import/export format conversions)."""

import json
import pytest

from app.services.zone_io import (
    zone_layers_to_modflow_zone_file,
    parse_modflow_zone_file,
    geojson_to_zone_layers,
    zone_layers_to_geodataframe,
    _build_structured_centroids,
)


# ---------------------------------------------------------------------------
# MODFLOW zone file roundtrip
# ---------------------------------------------------------------------------

class TestModflowZoneFileRoundtrip:
    """Test export → import roundtrip for MODFLOW zone files."""

    def test_basic_roundtrip(self):
        """3x4 grid, 2 layers — export and re-import should produce same data."""
        nlay, nrow, ncol = 2, 3, 4

        zone_layers = {
            "0": {
                "Zone 1": [0, 1, 4, 5],
                "Zone 2": [2, 3, 6, 7],
            },
            "1": {
                "Zone 1": [8, 9, 10, 11],
            },
        }

        content = zone_layers_to_modflow_zone_file(
            zone_layers, nlay=nlay, nrow=nrow, ncol=ncol,
        )

        # Verify format
        assert "# Zone Budget Definition File" in content
        assert "Zone 1" in content
        assert "Zone 2" in content

        # Parse back
        parsed_layers, num_zones = parse_modflow_zone_file(
            content, nlay=nlay, nrow=nrow, ncol=ncol,
        )

        assert num_zones == 2

        # Verify layer 0
        assert "0" in parsed_layers
        assert sorted(parsed_layers["0"]["Zone 1"]) == sorted(zone_layers["0"]["Zone 1"])
        assert sorted(parsed_layers["0"]["Zone 2"]) == sorted(zone_layers["0"]["Zone 2"])

        # Verify layer 1
        assert "1" in parsed_layers
        assert sorted(parsed_layers["1"]["Zone 1"]) == sorted(zone_layers["1"]["Zone 1"])

    def test_single_layer(self):
        """Single layer with 3 zones."""
        nlay, nrow, ncol = 1, 2, 3

        zone_layers = {
            "0": {
                "Zone 1": [0],
                "Zone 2": [1, 2],
                "Zone 3": [3, 4, 5],
            },
        }

        content = zone_layers_to_modflow_zone_file(
            zone_layers, nlay=nlay, nrow=nrow, ncol=ncol,
        )
        parsed_layers, num_zones = parse_modflow_zone_file(
            content, nlay=nlay, nrow=nrow, ncol=ncol,
        )

        assert num_zones == 3
        assert sorted(parsed_layers["0"]["Zone 3"]) == [3, 4, 5]


# ---------------------------------------------------------------------------
# parse_modflow_zone_file edge cases
# ---------------------------------------------------------------------------

class TestParseModflowZoneFile:

    def test_with_comments_and_internal(self):
        """Parse zone file with comments, INTERNAL keyword."""
        content = """\
# Zone definitions
# Zone 1 = 1, Zone 2 = 2
INTERNAL
1 1 2
2 2 1
"""
        layers, num = parse_modflow_zone_file(content, nlay=1, nrow=2, ncol=3)
        assert num == 2
        assert "0" in layers
        assert 0 in layers["0"]["Zone 1"]
        assert 2 in layers["0"]["Zone 2"]

    def test_constant_keyword(self):
        """Parse zone file with CONSTANT keyword."""
        content = """\
# Layer 1
CONSTANT 3
"""
        layers, num = parse_modflow_zone_file(content, nlay=1, nrow=2, ncol=3)
        assert num == 1
        assert "0" in layers
        # All 6 cells should be zone 3
        assert len(layers["0"]["Zone 3"]) == 6

    def test_empty_zones(self):
        """All-zero zone file produces empty zone_layers."""
        content = "0 0 0\n0 0 0\n"
        layers, num = parse_modflow_zone_file(content, nlay=1, nrow=2, ncol=3)
        assert num == 0
        assert len(layers) == 0

    def test_comma_separated(self):
        """Parse comma-separated values."""
        content = "1,2,3\n0,0,1\n"
        layers, num = parse_modflow_zone_file(content, nlay=1, nrow=2, ncol=3)
        assert num == 3
        assert 0 in layers["0"]["Zone 1"]


# ---------------------------------------------------------------------------
# GeoJSON to zone_layers
# ---------------------------------------------------------------------------

class TestGeojsonToZoneLayers:

    def test_simple_rectangle(self):
        """A single rectangle polygon covering part of a 4x4 grid."""
        # 4x4 grid with uniform 10m cells, no offset/rotation
        delr = [10.0] * 4
        delc = [10.0] * 4

        # Rectangle covering cells in rows 0-1, cols 0-1
        # Centroids: col 0 x=5, col 1 x=15 (in range); col 2 x=25, col 3 x=35 (out)
        # Row 0 y=35 (in), Row 1 y=25 (in), Row 2 y=15 (out), Row 3 y=5 (out)
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"zone": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 20], [20, 20], [20, 40], [0, 40], [0, 20]]],
                },
            }],
        }

        layers, num = geojson_to_zone_layers(
            geojson,
            nrow=4, ncol=4, nlay=1,
            delr=delr, delc=delc,
            zone_field="zone",
        )

        assert num == 1
        assert "0" in layers
        # 2 rows × 2 cols = 4 cells have centroids inside the polygon
        assert len(layers["0"]["Zone 1"]) == 4

    def test_auto_detect_zone_field(self):
        """Auto-detect zone field from properties."""
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"zone_id": "A", "other_field": 42},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]],
                },
            }],
        }

        layers, num = geojson_to_zone_layers(
            geojson,
            nrow=2, ncol=2, nlay=1,
            delr=[50.0, 50.0], delc=[50.0, 50.0],
        )
        assert num == 1
        assert "0" in layers

    def test_no_zone_field_raises(self):
        """Should raise ValueError with available_fields when no zone field detected."""
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"x_val": 1, "y_val": 2},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
            }],
        }

        with pytest.raises(ValueError) as exc_info:
            geojson_to_zone_layers(
                geojson,
                nrow=1, ncol=1, nlay=1,
                delr=[10.0], delc=[10.0],
            )

        err_data = json.loads(str(exc_info.value))
        assert "available_fields" in err_data
        assert "x_val" in err_data["available_fields"]


# ---------------------------------------------------------------------------
# GeoDataFrame export
# ---------------------------------------------------------------------------

class TestZoneLayersToGeoDataFrame:

    def test_produces_valid_gdf(self):
        """Export produces GeoDataFrame with correct row count."""
        try:
            import geopandas  # noqa: F401
        except ImportError:
            pytest.skip("geopandas not installed")

        zone_layers = {
            "0": {
                "Zone 1": [0, 1, 2],
                "Zone 2": [3, 4, 5],
            },
        }

        gdf = zone_layers_to_geodataframe(
            zone_layers,
            nrow=2, ncol=3, nlay=1,
            delr=[10.0, 10.0, 10.0],
            delc=[10.0, 10.0],
        )

        # Should have 2 rows: Zone 1 and Zone 2
        assert len(gdf) == 2
        assert "zone_name" in gdf.columns
        assert "geometry" in gdf.columns
        assert gdf.iloc[0]["cell_count"] == 3

    def test_with_epsg(self):
        """Export with EPSG sets CRS."""
        try:
            import geopandas  # noqa: F401
        except ImportError:
            pytest.skip("geopandas not installed")

        zone_layers = {"0": {"Zone 1": [0]}}
        gdf = zone_layers_to_geodataframe(
            zone_layers,
            nrow=1, ncol=1, nlay=1,
            delr=[10.0], delc=[10.0],
            epsg=4326,
        )

        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# Structured grid centroids
# ---------------------------------------------------------------------------

class TestStructuredCentroids:

    def test_uniform_grid(self):
        """2x2 uniform grid: centroids at expected positions."""
        centroids = _build_structured_centroids(
            delr=[10.0, 10.0],
            delc=[10.0, 10.0],
        )
        assert len(centroids) == 4
        # Row 0, Col 0: x=5, y=15 (top row, left col)
        assert abs(centroids[0][0] - 5.0) < 0.001
        assert abs(centroids[0][1] - 15.0) < 0.001
        # Row 1, Col 1: x=15, y=5 (bottom row, right col)
        assert abs(centroids[3][0] - 15.0) < 0.001
        assert abs(centroids[3][1] - 5.0) < 0.001

    def test_with_offset(self):
        """Grid with offset shifts centroids."""
        centroids = _build_structured_centroids(
            delr=[10.0],
            delc=[10.0],
            xoff=100.0,
            yoff=200.0,
        )
        assert abs(centroids[0][0] - 105.0) < 0.001
        assert abs(centroids[0][1] - 205.0) < 0.001
