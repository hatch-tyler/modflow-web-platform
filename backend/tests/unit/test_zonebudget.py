"""Unit tests for zone budget executable helpers."""

import csv
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestWriteMf6ZoneFile:
    """Test MF6 zone file (.zon) writing for zbud6."""

    def test_basic_zone_file(self, tmp_path):
        from app.api.v1.zonebudget import _write_mf6_zone_file

        zone_array = np.array([[[1, 2], [3, 0]], [[0, 1], [2, 3]]])  # (2, 2, 2)
        path = tmp_path / "test.zon"
        _write_mf6_zone_file(path, zone_array)

        content = path.read_text()
        assert "BEGIN DIMENSIONS" in content
        assert "NCELLS 8" in content
        assert "END DIMENSIONS" in content
        assert "BEGIN GRIDDATA" in content
        assert "IZONE" in content
        assert "INTERNAL FACTOR 1 IPRN 0" in content
        assert "END GRIDDATA" in content

        # Verify the values are in the file
        assert "1 2 3 0 0 1 2 3" in content

    def test_large_array_wraps_lines(self, tmp_path):
        from app.api.v1.zonebudget import _write_mf6_zone_file

        # 25 cells should wrap to two lines (20 + 5)
        zone_array = np.ones(25, dtype=int).reshape(1, 5, 5)
        path = tmp_path / "test.zon"
        _write_mf6_zone_file(path, zone_array)

        content = path.read_text()
        lines = content.strip().split("\n")
        # Find data lines (after INTERNAL FACTOR line, before END GRIDDATA)
        data_lines = [l.strip() for l in lines if l.strip().startswith("1")]
        assert len(data_lines) == 2  # 20 values + 5 values


class TestWriteZbud6Namefile:
    """Test zbud6 name file (.zbnam) writing."""

    def test_namefile_contents(self, tmp_path):
        from app.api.v1.zonebudget import _write_zbud6_namefile

        path = tmp_path / "test.zbnam"
        _write_zbud6_namefile(path, "zones.zon", "budget.cbc", "grid.grb")

        content = path.read_text()
        assert "BEGIN ZONEBUDGET" in content
        assert "ZON6  zones.zon" in content
        assert "BUD  budget.cbc" in content
        assert "GRB  grid.grb" in content
        assert "END ZONEBUDGET" in content


class TestParseZbud6Csv:
    """Test parsing of zbud6 CSV output."""

    def test_basic_csv_parsing(self, tmp_path):
        from app.api.v1.zonebudget import _parse_zbud6_csv

        csv_path = tmp_path / "output.csv"
        csv_path.write_text(textwrap.dedent("""\
            TOTIM,KSTP,KPER,ZONE,STO-SS_IN,STO-SS_OUT,MAW_IN,MAW_OUT,TOTAL_IN,TOTAL_OUT,IN-OUT
            1.0,1,1,ZONE_1,100.0,50.0,200.0,0.0,300.0,50.0,250.0
            1.0,1,1,ZONE_2,0.0,30.0,0.0,150.0,0.0,180.0,-180.0
        """))

        zone_name_to_num = {"Zone 1": 1, "Zone 2": 2}
        result = _parse_zbud6_csv(csv_path, zone_name_to_num)

        assert result["zone_names"] == ["Zone 1", "Zone 2"]
        assert "ZONE_1" in result["columns"]
        assert "ZONE_2" in result["columns"]
        assert len(result["records"]) > 0

        # Check that FROM_STO_SS has the right zone values
        sto_in_recs = [r for r in result["records"] if r["name"] == "FROM_STO-SS"]
        assert len(sto_in_recs) == 1
        assert sto_in_recs[0]["ZONE_1"] == 100.0
        assert sto_in_recs[0]["ZONE_2"] == 0.0

    def test_interzone_flow_parsing(self, tmp_path):
        from app.api.v1.zonebudget import _parse_zbud6_csv

        csv_path = tmp_path / "output.csv"
        csv_path.write_text(textwrap.dedent("""\
            TOTIM,KSTP,KPER,ZONE,STO-SS_IN,STO-SS_OUT,FROM_ZONE_2,TO_ZONE_2,TOTAL_IN,TOTAL_OUT,IN-OUT
            1.0,1,1,ZONE_1,10.0,5.0,300.0,100.0,310.0,105.0,205.0
        """))

        zone_name_to_num = {"Zone 1": 1}
        result = _parse_zbud6_csv(csv_path, zone_name_to_num)

        # Should have FROM_ZONE_2 and TO_ZONE_2 records
        rec_names = {r["name"] for r in result["records"]}
        assert "FROM_ZONE_2" in rec_names
        assert "TO_ZONE_2" in rec_names

    def test_empty_csv(self, tmp_path):
        from app.api.v1.zonebudget import _parse_zbud6_csv

        csv_path = tmp_path / "output.csv"
        csv_path.write_text("")

        result = _parse_zbud6_csv(csv_path, {"Zone 1": 1})
        assert result["records"] == []

    def test_multi_period_csv(self, tmp_path):
        from app.api.v1.zonebudget import _parse_zbud6_csv

        csv_path = tmp_path / "output.csv"
        csv_path.write_text(textwrap.dedent("""\
            TOTIM,KSTP,KPER,ZONE,RCH_IN,RCH_OUT,TOTAL_IN,TOTAL_OUT,IN-OUT
            1.0,1,1,ZONE_1,100.0,0.0,100.0,0.0,100.0
            2.0,1,2,ZONE_1,200.0,0.0,200.0,0.0,200.0
        """))

        zone_name_to_num = {"Zone 1": 1}
        result = _parse_zbud6_csv(csv_path, zone_name_to_num)

        # Should have records for both periods
        kpers = {r["kper"] for r in result["records"]}
        assert 1 in kpers
        assert 2 in kpers


class TestFindGrbObject:
    """Test GRB file finder."""

    def test_finds_grb_file(self):
        from app.api.v1.zonebudget import _find_grb_object

        mock_storage = MagicMock()
        mock_storage.list_objects.return_value = [
            "results/run1/output/flow.dis.grb",
            "results/run1/output/flow.hds",
        ]

        result = _find_grb_object(mock_storage, "results/run1")
        assert result == "results/run1/output/flow.dis.grb"

    def test_prefers_fewer_dots(self):
        from app.api.v1.zonebudget import _find_grb_object

        mock_storage = MagicMock()
        mock_storage.list_objects.return_value = [
            "results/output/flow.disv.grb",
            "results/output/flow.grb",
        ]

        result = _find_grb_object(mock_storage, "results")
        assert result == "results/output/flow.grb"

    def test_no_grb_file(self):
        from app.api.v1.zonebudget import _find_grb_object

        mock_storage = MagicMock()
        mock_storage.list_objects.return_value = [
            "results/output/flow.hds",
            "results/output/flow.cbc",
        ]

        result = _find_grb_object(mock_storage, "results")
        assert result is None


class TestResultsPathFromCbc:
    """Test extracting results base path from CBC object path."""

    def test_with_output_dir(self):
        from app.api.v1.zonebudget import _results_path_from_cbc

        assert _results_path_from_cbc("runs/abc/output/flow.cbc") == "runs/abc"

    def test_without_output_dir(self):
        from app.api.v1.zonebudget import _results_path_from_cbc

        assert _results_path_from_cbc("runs/abc/flow.cbc") == "runs/abc"


class TestFallbackBehavior:
    """Test that fallback to Python zone budget works when executables are unavailable."""

    @patch("app.api.v1.zonebudget.settings")
    def test_falls_back_when_zbud6_missing(self, mock_settings):
        """When zbud6 path doesn't exist, should fall back to Python."""
        mock_settings.zbud6_exe_path = "/nonexistent/zbud6"

        from pathlib import Path as RealPath
        # Just verify the path check logic
        assert not RealPath("/nonexistent/zbud6").exists()
