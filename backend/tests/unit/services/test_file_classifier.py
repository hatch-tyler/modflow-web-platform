"""Tests for the file classifier service.

Tests file classification, security blocking, and observation detection.
"""

import pytest
from pathlib import Path

from app.services.file_classifier import (
    is_blocked_file,
    get_blocked_files,
    classify_file,
    classify_files,
    classify_directory,
    detect_observation_csv,
    get_file_description,
    get_categorized_summary,
    BLOCKED_EXTENSIONS,
)


class TestIsBlockedFile:
    """Tests for is_blocked_file function."""

    @pytest.mark.parametrize("filename,expected", [
        # Executables - should be blocked
        ("model.exe", True),
        ("library.dll", True),
        ("shared.so", True),
        ("plugin.dylib", True),
        ("program.bin", True),
        ("setup.msi", True),
        # Scripts - should be blocked
        ("script.py", True),
        ("batch.bat", True),
        ("shell.sh", True),
        ("powershell.ps1", True),
        ("vbscript.vbs", True),
        ("node.js", True),
        # Archives - should be blocked (nested archive risk)
        ("archive.zip", True),
        ("backup.tar", True),
        ("compressed.gz", True),
        ("compressed.7z", True),
        # Model files - should NOT be blocked
        ("model.nam", False),
        ("model.dis", False),
        ("model.hds", False),
        ("data.txt", False),
        ("array.arr", False),
        ("observations.csv", False),
        # PEST files - should NOT be blocked
        ("pest.pst", False),
        ("template.tpl", False),
        ("instructions.ins", False),
        # Edge cases
        ("", False),  # Empty filename
        ("noextension", False),  # No extension
        (".hidden", False),  # Hidden file (Unix-style)
    ])
    def test_is_blocked_file(self, filename: str, expected: bool):
        """Test file blocking based on extension."""
        assert is_blocked_file(filename) == expected

    def test_is_blocked_file_case_insensitive(self):
        """Test that extension check is case-insensitive."""
        assert is_blocked_file("model.EXE") == True
        assert is_blocked_file("script.PY") == True
        assert is_blocked_file("model.NAM") == False

    def test_is_blocked_file_with_path(self):
        """Test blocking with full file paths."""
        assert is_blocked_file("path/to/model.exe") == True
        assert is_blocked_file("path/to/model.nam") == False
        assert is_blocked_file("C:/projects/script.py") == True


class TestGetBlockedFiles:
    """Tests for get_blocked_files function."""

    def test_get_blocked_files_returns_only_blocked(self):
        """Test that only blocked files are returned."""
        files = [
            "model.nam",
            "model.dis",
            "malware.exe",
            "script.py",
            "data.txt",
            "archive.zip",
        ]
        blocked = get_blocked_files(files)
        assert set(blocked) == {"malware.exe", "script.py", "archive.zip"}

    def test_get_blocked_files_empty_list(self):
        """Test with empty file list."""
        assert get_blocked_files([]) == []

    def test_get_blocked_files_no_blocked(self):
        """Test when no files are blocked."""
        files = ["model.nam", "model.dis", "data.txt"]
        assert get_blocked_files(files) == []


class TestClassifyFile:
    """Tests for classify_file function."""

    @pytest.mark.parametrize("filepath,expected_category", [
        # Model core files
        ("mfsim.nam", "model_core"),
        ("model.nam", "model_core"),
        ("model.dis", "model_core"),
        ("model.disu", "model_core"),
        ("model.npf", "model_core"),
        ("model.ims", "model_core"),
        ("model.ic", "model_core"),
        # Model input files
        ("model.wel", "model_input"),
        ("model.drn", "model_input"),
        ("model.riv", "model_input"),
        ("model.rch", "model_input"),
        ("model.sfr", "model_input"),
        ("data.txt", "model_input"),
        ("array.arr", "model_input"),
        # Model output files
        ("model.hds", "model_output"),
        ("model.cbc", "model_output"),
        ("model.lst", "model_output"),
        # PEST files
        ("pest.pst", "pest"),
        ("template.tpl", "pest"),
        ("instructions.ins", "pest"),
        ("jacobian.jco", "pest"),
        # Blocked files
        ("malware.exe", "blocked"),
        ("script.py", "blocked"),
        ("archive.zip", "blocked"),
        # Other files
        ("readme.md", "other"),
        ("config.json", "other"),
        ("image.png", "other"),
    ])
    def test_classify_file_categories(self, filepath: str, expected_category: str):
        """Test file classification into categories."""
        assert classify_file(filepath) == expected_category

    def test_classify_file_observation_patterns(self):
        """Test observation file detection by filename pattern."""
        assert classify_file("observations.csv") == "observation"
        assert classify_file("head_obs.csv") == "observation"
        assert classify_file("field_data.csv") == "observation"
        assert classify_file("calibration_data.csv") == "observation"
        assert classify_file("measured_heads.csv") == "other"  # Doesn't match pattern

    def test_classify_file_observation_folder(self):
        """Test observation file detection by folder."""
        assert classify_file("observations/wells.csv") == "observation"
        assert classify_file("obs/data.csv") == "observation"
        assert classify_file("field_data/measurements.csv") == "observation"

    def test_classify_file_with_required_files(self):
        """Test that files in required_files set are classified as model_core."""
        required = {"custom_data.dat", "special_array.ref"}
        assert classify_file("custom_data.dat", required) == "model_core"
        assert classify_file("special_array.ref", required) == "model_core"
        # Files not in required set use normal classification
        assert classify_file("other.dat", required) == "model_input"

    def test_classify_file_pest_patterns(self):
        """Test PEST-related filename patterns."""
        assert classify_file("pest_run.dat") == "pest"
        assert classify_file("model_pest.cfg") == "pest"
        assert classify_file("pest_config.json") == "pest"


class TestClassifyFiles:
    """Tests for classify_files function."""

    def test_classify_files_returns_all_categories(self):
        """Test that all category keys are present in result."""
        files = ["model.nam"]
        result = classify_files(files)

        expected_keys = {
            'model_core', 'model_input', 'model_output',
            'pest', 'observation', 'blocked', 'other'
        }
        assert set(result.keys()) == expected_keys

    def test_classify_files_correct_categorization(self):
        """Test that files are placed in correct categories."""
        files = [
            "model.nam",
            "model.wel",
            "model.hds",
            "pest.pst",
            "observations.csv",
            "malware.exe",
            "readme.md",
        ]
        result = classify_files(files)

        assert len(result["model_core"]) == 1
        assert result["model_core"][0]["name"] == "model.nam"

        assert len(result["model_input"]) == 1
        assert result["model_input"][0]["name"] == "model.wel"

        assert len(result["model_output"]) == 1
        assert len(result["pest"]) == 1
        assert len(result["observation"]) == 1
        assert len(result["blocked"]) == 1
        assert len(result["other"]) == 1

    def test_classify_files_file_info_structure(self):
        """Test that file info contains expected fields."""
        files = ["model.nam"]
        result = classify_files(files)

        file_info = result["model_core"][0]
        assert "path" in file_info
        assert "name" in file_info
        assert "extension" in file_info
        assert "description" in file_info


class TestClassifyDirectory:
    """Tests for classify_directory function."""

    def test_classify_directory_includes_sizes(self, tmp_path: Path):
        """Test that directory classification includes file sizes."""
        # Create test files
        (tmp_path / "model.nam").write_text("NAME FILE")
        (tmp_path / "data.txt").write_text("DATA")

        result = classify_directory(tmp_path)

        # Check that sizes are included
        for category in result.values():
            for file_info in category:
                assert "size" in file_info
                assert isinstance(file_info["size"], int)

    def test_classify_directory_recursive(self, tmp_path: Path):
        """Test that directory classification is recursive."""
        # Create nested structure
        subdir = tmp_path / "nested"
        subdir.mkdir()
        (tmp_path / "model.nam").write_text("TOP")
        (subdir / "model.wel").write_text("NESTED")

        result = classify_directory(tmp_path)

        # Should find files in subdirectories
        all_paths = []
        for category in result.values():
            all_paths.extend(f["path"] for f in category)

        assert "model.nam" in all_paths
        assert "nested/model.wel" in all_paths


class TestDetectObservationCsv:
    """Tests for detect_observation_csv function."""

    def test_detect_long_format(self, tmp_path: Path):
        """Test detection of long format observation CSV."""
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text(
            "WellName,Layer,Row,Col,Time,Head\n"
            "MW-01,1,10,10,1.0,45.2\n"
            "MW-01,1,10,10,30.0,44.8\n"
        )

        result = detect_observation_csv(csv_file)

        assert result is not None
        assert result["format"] == "long"
        assert result["n_observations"] == 2
        assert result["column_mapping"]["well_name"] == "wellname"
        assert result["column_mapping"]["time"] == "time"

    def test_detect_wide_format(self, tmp_path: Path):
        """Test detection of wide format observation CSV."""
        csv_file = tmp_path / "obs.csv"
        csv_file.write_text(
            "Time,MW-01,MW-02,MW-03\n"
            "1.0,45.2,42.1,48.3\n"
            "30.0,44.8,41.5,47.9\n"
        )

        result = detect_observation_csv(csv_file)

        assert result is not None
        assert result["format"] == "wide"
        # Headers are lowercased by the parser
        assert "mw-01" in result["wells"]
        assert "mw-02" in result["wells"]
        assert "mw-03" in result["wells"]

    def test_detect_unrecognized_format(self, tmp_path: Path):
        """Test that unrecognized formats return None."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "Col1,Col2,Col3\n"
            "1,2,3\n"
        )

        result = detect_observation_csv(csv_file)
        assert result is None

    def test_detect_empty_file(self, tmp_path: Path):
        """Test handling of empty file."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        result = detect_observation_csv(csv_file)
        assert result is None

    def test_detect_nonexistent_file(self, tmp_path: Path):
        """Test handling of nonexistent file."""
        result = detect_observation_csv(tmp_path / "nonexistent.csv")
        assert result is None


class TestGetFileDescription:
    """Tests for get_file_description function."""

    @pytest.mark.parametrize("filename,extension,expected_desc", [
        ("mfsim.nam", ".nam", "MODFLOW 6 simulation name file"),
        ("model.nam", ".nam", "Model name file"),
        ("model.dis", ".dis", "Discretization package"),
        ("model.npf", ".npf", "Node property flow package"),
        ("model.wel", ".wel", "Well package"),
        ("model.hds", ".hds", "Head output file"),
        ("pest.pst", ".pst", "PEST control file"),
        ("unknown.xyz", ".xyz", ""),
    ])
    def test_get_file_description(self, filename: str, extension: str, expected_desc: str):
        """Test file description generation."""
        desc = get_file_description(filename, extension)
        assert desc == expected_desc


class TestGetCategorizedSummary:
    """Tests for get_categorized_summary function."""

    def test_summary_structure(self):
        """Test summary contains expected fields."""
        categories = {
            'model_core': [{'path': 'a.nam', 'size': 100}],
            'model_input': [{'path': 'b.wel', 'size': 200}],
            'model_output': [],
            'pest': [],
            'observation': [],
            'blocked': [],
            'other': [],
        }

        summary = get_categorized_summary(categories)

        assert "total_files" in summary
        assert "total_size_bytes" in summary
        assert "total_size_mb" in summary
        assert "categories" in summary

    def test_summary_counts(self):
        """Test summary counts files correctly."""
        categories = {
            'model_core': [{'path': 'a.nam', 'size': 100}, {'path': 'b.dis', 'size': 200}],
            'model_input': [{'path': 'c.wel', 'size': 300}],
            'model_output': [],
            'pest': [],
            'observation': [],
            'blocked': [],
            'other': [],
        }

        summary = get_categorized_summary(categories)

        assert summary["total_files"] == 3
        assert summary["total_size_bytes"] == 600
        assert summary["categories"]["model_core"]["count"] == 2
        assert summary["categories"]["model_core"]["size"] == 300
        assert summary["categories"]["model_input"]["count"] == 1

    def test_summary_empty_categories(self):
        """Test summary with all empty categories."""
        categories = {
            'model_core': [],
            'model_input': [],
            'model_output': [],
            'pest': [],
            'observation': [],
            'blocked': [],
            'other': [],
        }

        summary = get_categorized_summary(categories)

        assert summary["total_files"] == 0
        assert summary["total_size_bytes"] == 0
