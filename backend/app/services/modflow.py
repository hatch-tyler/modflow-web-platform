"""MODFLOW model parsing and validation service using FloPy."""

import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Apply FloPy SFR2 patches before importing flopy
# This must be done early to ensure patches are applied
try:
    import flopy_patches.apply_patches  # noqa: F401
except ImportError:
    pass  # Patches not available, use standard FloPy

import flopy
import numpy as np

from app.services.path_normalizer import (
    extract_zip_with_normalized_paths,
    normalize_all_model_files,
)


@dataclass
class ValidationResult:
    """Result of model validation."""

    is_valid: bool
    model_type: Optional[str] = None
    nlay: Optional[int] = None
    nrow: Optional[int] = None
    ncol: Optional[int] = None
    nper: Optional[int] = None
    packages: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    xoff: Optional[float] = None
    yoff: Optional[float] = None
    angrot: Optional[float] = None
    epsg: Optional[int] = None
    length_unit: Optional[str] = None
    stress_period_data: Optional[list] = None
    delr: Optional[list] = None
    delc: Optional[list] = None
    time_unit: Optional[str] = None
    start_date: Optional[str] = None
    grid_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "model_type": self.model_type,
            "grid_info": {
                "nlay": self.nlay,
                "nrow": self.nrow,
                "ncol": self.ncol,
                "nper": self.nper,
                "grid_type": self.grid_type,
            }
            if self.nlay
            else None,
            "packages_found": list(self.packages.keys()),
            "packages_missing": [],
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _parse_nam_spatial_reference(nam_path: Path) -> dict:
    """
    Parse spatial reference info from NAM file comments.

    Looks for comment lines like:
    - # xoff:123.45 yoff:678.90 angrot:0.0 epsg:32615
    - # start_date:2000-01-01
    - # xul:123 yul:456 rotation:0  (FloPy style)
    """
    import re

    result = {}
    try:
        content = nam_path.read_text(errors="replace")
    except Exception:
        return result

    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        text = stripped.lstrip("#").strip()

        # Parse key:value pairs
        for match in re.finditer(
            r"(xoff|yoff|angrot|epsg|start_date|xul|yul|rotation)\s*[:=]\s*(\S+)",
            text,
            re.IGNORECASE,
        ):
            key = match.group(1).lower()
            val = match.group(2)
            try:
                if key == "epsg":
                    result["epsg"] = int(val)
                elif key == "start_date":
                    result["start_date"] = val
                elif key == "xoff":
                    result["xoff"] = float(val)
                elif key == "yoff":
                    result["yoff"] = float(val)
                elif key == "angrot":
                    result["angrot"] = float(val)
                elif key == "xul":
                    result["xoff"] = float(val)
                elif key == "yul":
                    result["yoff"] = float(val)
                elif key == "rotation":
                    result["angrot"] = float(val)
            except (ValueError, TypeError):
                pass

    return result


def _parse_dis_options_spatial_reference(model_dir: Path) -> dict:
    """
    Directly parse MF6 DIS file OPTIONS block for XORIGIN/YORIGIN/ANGROT.

    Fallback when FloPy doesn't capture these from the DIS package.
    """
    import re

    result = {}
    # Look for DIS files in model directory
    for f in model_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".dis", ".dis6", ".disv", ".disv6", ".disu", ".disu6"):
            continue
        try:
            content = f.read_text(errors="replace")
        except Exception:
            continue

        in_options = False
        for line in content.split("\n"):
            stripped = line.strip().upper()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped.startswith("BEGIN OPTIONS"):
                in_options = True
                continue
            if stripped.startswith("END OPTIONS"):
                break
            if not in_options:
                continue

            # Parse XORIGIN, YORIGIN, ANGROT values
            for key in ("XORIGIN", "YORIGIN", "ANGROT"):
                m = re.match(rf"{key}\s+(\S+)", stripped)
                if m:
                    try:
                        val = float(m.group(1))
                        if key == "XORIGIN":
                            result["xoff"] = val
                        elif key == "YORIGIN":
                            result["yoff"] = val
                        elif key == "ANGROT":
                            result["angrot"] = val
                    except (ValueError, TypeError):
                        pass
        if result:
            break  # Found data in this file
    return result


def _parse_model_reference_file(model_dir: Path) -> dict:
    """
    Parse usgs.model.reference file for spatial reference.

    Common FloPy convention — file contains key value pairs like:
      xll 123456.0
      yll 654321.0
      rotation 0.0
      epsg 32611
      start_datetime 1/1/2000
    """
    result = {}
    ref_file = model_dir / "usgs.model.reference"
    if not ref_file.exists():
        return result

    try:
        content = ref_file.read_text(errors="replace")
    except Exception:
        return result

    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) < 2:
            continue
        key, val = parts[0].lower(), parts[1].strip()
        try:
            if key in ("xll", "xul", "xoff"):
                result["xoff"] = float(val)
            elif key in ("yll", "yul", "yoff"):
                result["yoff"] = float(val)
            elif key in ("rotation", "angrot"):
                result["angrot"] = float(val)
            elif key == "epsg":
                result["epsg"] = int(val)
            elif key in ("start_datetime", "start_date"):
                result["start_date"] = val
        except (ValueError, TypeError):
            pass

    return result


# ITMUNI mapping for MF2005/NWT/USG
_ITMUNI_MAP = {
    0: "undefined",
    1: "seconds",
    2: "minutes",
    3: "hours",
    4: "days",
    5: "years",
}


def detect_model_type(model_dir: Path) -> Optional[str]:
    """
    Detect MODFLOW model type based on files present.

    Returns:
        'mf6' for MODFLOW 6
        'mfusg' for MODFLOW-USG
        'mfnwt' for MODFLOW-NWT
        'mf2005' for MODFLOW-2005
        None if unknown
    """
    # Get actual file objects to preserve case
    all_files = [f for f in model_dir.iterdir() if f.is_file()]
    file_names_lower = [f.name.lower() for f in all_files]

    # MODFLOW 6: look for mfsim.nam or .tdis files
    if any(f == "mfsim.nam" for f in file_names_lower):
        return "mf6"
    if any(f.endswith(".tdis") for f in file_names_lower):
        return "mf6"

    # MODFLOW-NWT/2005/USG: look for .nam files (case-insensitive)
    nam_files = [f for f in all_files if f.suffix.lower() == ".nam"]
    if nam_files:
        for nam_file in nam_files:
            try:
                content = nam_file.read_text().upper()
                # Check for USG-specific packages
                if "DISU" in content or "CLN" in content or "GNC" in content:
                    return "mfusg"
                # Check if it's NWT by looking for UPW package
                if "UPW" in content or "NWT" in content:
                    return "mfnwt"
            except Exception:
                pass
        return "mf2005"

    return None


def find_nam_file(model_dir: Path) -> Optional[Path]:
    """Find the main NAM file in a directory."""
    for f in model_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".nam":
            # Skip mfsim.nam for MF6
            if f.name.lower() != "mfsim.nam":
                return f
    return None


def load_mf6_model(model_dir: Path) -> ValidationResult:
    """Load and validate a MODFLOW 6 model."""
    result = ValidationResult(is_valid=False, model_type="mf6")

    try:
        # Load the simulation
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=str(model_dir),
            verbosity_level=0,
            load_only=None,
        )

        # Get the groundwater flow model(s)
        model_names = sim.model_names
        if not model_names:
            result.errors.append("No groundwater flow models found in simulation")
            return result

        # Use the first GWF model
        gwf = sim.get_model(model_names[0])

        # Extract grid information from discretization package
        # Note: gwf.get_package("DIS") returns whatever dis package exists
        # (DIS, DISV, or DISU), so we must check the actual package type.
        dis_pkg = gwf.get_package("DIS")
        pkg_type = getattr(dis_pkg, 'package_type', '').upper() if dis_pkg else ''

        if pkg_type == "DIS":
            result.grid_type = "structured"
            result.nlay = dis_pkg.nlay.data
            result.nrow = dis_pkg.nrow.data
            result.ncol = dis_pkg.ncol.data
            dis = dis_pkg  # Keep reference for delr/delc extraction
        elif pkg_type == "DISV":
            result.grid_type = "vertex"
            result.nlay = dis_pkg.nlay.data
            result.ncol = dis_pkg.ncpl.data  # cells per layer
            result.nrow = 1  # DISV doesn't have rows
            dis = None  # No delr/delc for vertex grids
        elif pkg_type == "DISU":
            result.grid_type = "unstructured"
            result.nlay = 1
            result.nrow = 1
            result.ncol = dis_pkg.nodes.data
            dis = None  # No delr/delc for unstructured grids
        else:
            dis = None

        # Get number of stress periods from TDIS
        if sim.tdis is not None:
            result.nper = sim.tdis.nper.data

            # Extract stress period data
            try:
                period_data = sim.tdis.perioddata.get_data()
                spd = []
                for row in period_data:
                    spd.append({
                        "perlen": float(row[0]),
                        "nstp": int(row[1]),
                        "tsmult": float(row[2]),
                    })
                result.stress_period_data = spd
            except Exception:
                pass

            # Extract time unit
            try:
                tu = sim.tdis.time_units.get_data()
                if tu:
                    result.time_unit = str(tu).lower()
            except Exception:
                pass

        # Extract delr/delc for structured grids
        if dis is not None:
            try:
                delr_data = dis.delr.get_data()
                delc_data = dis.delc.get_data()
                if delr_data is not None:
                    result.delr = [float(v) for v in np.asarray(delr_data).flatten()]
                if delc_data is not None:
                    result.delc = [float(v) for v in np.asarray(delc_data).flatten()]
            except Exception:
                pass

        # Collect package information
        for pkg_name in gwf.package_names:
            pkg = gwf.get_package(pkg_name)
            pkg_type = pkg.package_type.upper() if hasattr(pkg, "package_type") else pkg_name.upper()
            result.packages[pkg_type] = {
                "name": pkg_name,
                "type": pkg_type,
            }

        # Extract coordinate reference from modelgrid
        try:
            grid = gwf.modelgrid
            if grid.xoffset is not None and grid.xoffset != 0:
                result.xoff = float(grid.xoffset)
            if grid.yoffset is not None and grid.yoffset != 0:
                result.yoff = float(grid.yoffset)
            if grid.angrot is not None and grid.angrot != 0:
                result.angrot = float(grid.angrot)
            if hasattr(grid, 'epsg') and grid.epsg:
                result.epsg = int(grid.epsg)
            # Also check DIS/DISV/DISU xorigin/yorigin via FloPy
            # dis_pkg was set during grid extraction above
            if dis_pkg is not None:
                try:
                    if hasattr(dis_pkg, 'xorigin') and dis_pkg.xorigin.data is not None:
                        val = float(dis_pkg.xorigin.data)
                        if val != 0:
                            result.xoff = val
                except Exception:
                    pass
                try:
                    if hasattr(dis_pkg, 'yorigin') and dis_pkg.yorigin.data is not None:
                        val = float(dis_pkg.yorigin.data)
                        if val != 0:
                            result.yoff = val
                except Exception:
                    pass
            # Length unit
            if hasattr(grid, 'length_unit') and grid.length_unit:
                result.length_unit = str(grid.length_unit)
        except Exception as e:
            logger.debug(f"MF6 coordinate extraction from modelgrid failed: {e}")

        # Fallback: directly parse DIS file OPTIONS block for XORIGIN/YORIGIN
        if not result.xoff and not result.yoff:
            try:
                dis_info = _parse_dis_options_spatial_reference(model_dir)
                if dis_info.get("xoff") and not result.xoff:
                    result.xoff = dis_info["xoff"]
                if dis_info.get("yoff") and not result.yoff:
                    result.yoff = dis_info["yoff"]
                if dis_info.get("angrot") and not result.angrot:
                    result.angrot = dis_info["angrot"]
            except Exception as e:
                logger.debug(f"MF6 DIS OPTIONS spatial reference parse failed: {e}")

        # Fallback: usgs.model.reference file
        try:
            ref_info = _parse_model_reference_file(model_dir)
            if ref_info.get("xoff") and not result.xoff:
                result.xoff = ref_info["xoff"]
            if ref_info.get("yoff") and not result.yoff:
                result.yoff = ref_info["yoff"]
            if ref_info.get("angrot") and not result.angrot:
                result.angrot = ref_info["angrot"]
            if ref_info.get("epsg") and not result.epsg:
                result.epsg = ref_info["epsg"]
            if ref_info.get("start_date") and not result.start_date:
                result.start_date = ref_info["start_date"]
        except Exception:
            pass

        # NAM fallback: parse GWF model NAM file for spatial reference comments
        try:
            # Find the GWF NAM file via mfsim.nam MODELS block
            for f in model_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".nam", ".gwf"):
                    if f.name.lower() != "mfsim.nam":
                        nam_info = _parse_nam_spatial_reference(f)
                        if nam_info.get("xoff") and not result.xoff:
                            result.xoff = nam_info["xoff"]
                        if nam_info.get("yoff") and not result.yoff:
                            result.yoff = nam_info["yoff"]
                        if nam_info.get("angrot") and not result.angrot:
                            result.angrot = nam_info["angrot"]
                        if nam_info.get("epsg") and not result.epsg:
                            result.epsg = nam_info["epsg"]
                        if nam_info.get("start_date"):
                            result.start_date = nam_info["start_date"]
                        break
            # Also check mfsim.nam itself for comments
            mfsim_path = model_dir / "mfsim.nam"
            if mfsim_path.exists():
                nam_info = _parse_nam_spatial_reference(mfsim_path)
                if nam_info.get("xoff") and not result.xoff:
                    result.xoff = nam_info["xoff"]
                if nam_info.get("yoff") and not result.yoff:
                    result.yoff = nam_info["yoff"]
                if nam_info.get("angrot") and not result.angrot:
                    result.angrot = nam_info["angrot"]
                if nam_info.get("epsg") and not result.epsg:
                    result.epsg = nam_info["epsg"]
                if nam_info.get("start_date") and not result.start_date:
                    result.start_date = nam_info["start_date"]
        except Exception as e:
            logger.debug(f"MF6 NAM spatial reference parse failed: {e}")

        result.is_valid = True

    except Exception as e:
        result.errors.append(f"Failed to load MODFLOW 6 model: {str(e)}")

    return result


def load_mf2005_model(model_dir: Path, model_type: str = "mf2005") -> ValidationResult:
    """Load and validate a MODFLOW-2005 or MODFLOW-NWT model."""
    result = ValidationResult(is_valid=False, model_type=model_type)

    nam_file = find_nam_file(model_dir)
    if not nam_file:
        result.errors.append("No NAM file found in model directory")
        return result

    try:
        # Determine which loader to use
        if model_type == "mfnwt":
            model = flopy.modflow.Modflow.load(
                nam_file.name,
                model_ws=str(model_dir),
                check=False,
                verbose=False,
                load_only=None,
            )
        else:
            model = flopy.modflow.Modflow.load(
                nam_file.name,
                model_ws=str(model_dir),
                check=False,
                verbose=False,
                load_only=None,
            )

        # Extract grid information from DIS package
        result.grid_type = "structured"
        if model.dis is not None:
            result.nlay = model.dis.nlay
            result.nrow = model.dis.nrow
            result.ncol = model.dis.ncol
            result.nper = model.dis.nper

            # Extract stress period data
            try:
                perlen = np.asarray(model.dis.perlen.array).flatten()
                nstp = np.asarray(model.dis.nstp.array).flatten()
                tsmult = np.asarray(model.dis.tsmult.array).flatten()
                spd = []
                for i in range(len(perlen)):
                    spd.append({
                        "perlen": float(perlen[i]),
                        "nstp": int(nstp[i]),
                        "tsmult": float(tsmult[i]),
                    })
                result.stress_period_data = spd
            except Exception:
                pass

            # Extract time unit
            try:
                itmuni = model.dis.itmuni
                result.time_unit = _ITMUNI_MAP.get(itmuni, "undefined")
            except Exception:
                pass

            # Extract delr/delc
            try:
                result.delr = [float(v) for v in np.asarray(model.dis.delr.array).flatten()]
                result.delc = [float(v) for v in np.asarray(model.dis.delc.array).flatten()]
            except Exception:
                pass

        # Collect package information
        for pkg in model.packagelist:
            pkg_name = pkg.name[0].upper() if isinstance(pkg.name, list) else pkg.name.upper()
            result.packages[pkg_name] = {
                "name": pkg_name,
                "type": pkg_name,
            }

        # Extract coordinate reference from modelgrid
        try:
            grid = model.modelgrid
            if grid.xoffset is not None and grid.xoffset != 0:
                result.xoff = float(grid.xoffset)
            if grid.yoffset is not None and grid.yoffset != 0:
                result.yoff = float(grid.yoffset)
            if grid.angrot is not None and grid.angrot != 0:
                result.angrot = float(grid.angrot)
            if hasattr(grid, 'epsg') and grid.epsg:
                result.epsg = int(grid.epsg)
            if hasattr(grid, 'length_unit') and grid.length_unit:
                result.length_unit = str(grid.length_unit)
        except Exception as e:
            logger.debug(f"MF2005 coordinate extraction from modelgrid failed: {e}")

        # Fallback: usgs.model.reference file
        try:
            ref_info = _parse_model_reference_file(model_dir)
            if ref_info.get("xoff") and not result.xoff:
                result.xoff = ref_info["xoff"]
            if ref_info.get("yoff") and not result.yoff:
                result.yoff = ref_info["yoff"]
            if ref_info.get("angrot") and not result.angrot:
                result.angrot = ref_info["angrot"]
            if ref_info.get("epsg") and not result.epsg:
                result.epsg = ref_info["epsg"]
            if ref_info.get("start_date") and not result.start_date:
                result.start_date = ref_info["start_date"]
        except Exception as e:
            logger.debug(f"MF2005 model reference file parse failed: {e}")

        # NAM fallback: parse NAM file comments for spatial reference
        if nam_file:
            try:
                nam_info = _parse_nam_spatial_reference(nam_file)
                if nam_info.get("xoff") and not result.xoff:
                    result.xoff = nam_info["xoff"]
                if nam_info.get("yoff") and not result.yoff:
                    result.yoff = nam_info["yoff"]
                if nam_info.get("angrot") and not result.angrot:
                    result.angrot = nam_info["angrot"]
                if nam_info.get("epsg") and not result.epsg:
                    result.epsg = nam_info["epsg"]
                if nam_info.get("start_date"):
                    result.start_date = nam_info["start_date"]
            except Exception as e:
                logger.debug(f"MF2005 NAM spatial reference parse failed: {e}")

        result.is_valid = True

    except Exception as e:
        result.errors.append(f"Failed to load {model_type.upper()} model: {str(e)}")

    return result


def load_mfusg_model(model_dir: Path) -> ValidationResult:
    """Load and validate a MODFLOW-USG model."""
    result = ValidationResult(is_valid=False, model_type="mfusg")

    nam_file = find_nam_file(model_dir)
    if not nam_file:
        result.errors.append("No NAM file found in model directory")
        return result

    try:
        # Use flopy.mfusg for USG models
        model = flopy.mfusg.MfUsg.load(
            nam_file.name,
            model_ws=str(model_dir),
            check=False,
            verbose=False,
        )

        # Extract grid information from DISU package
        dis_pkg = None
        if model.disu is not None:
            result.grid_type = "unstructured"
            result.nlay = model.disu.nlay
            result.nrow = 1  # USG doesn't have traditional rows
            result.ncol = model.disu.nodes  # Total nodes
            result.nper = model.disu.nper
            dis_pkg = model.disu
        elif model.dis is not None:
            # Some USG models may still use DIS
            result.grid_type = "structured"
            result.nlay = model.dis.nlay
            result.nrow = model.dis.nrow
            result.ncol = model.dis.ncol
            result.nper = model.dis.nper
            dis_pkg = model.dis

        # Extract stress period data and time unit from dis/disu
        if dis_pkg is not None:
            try:
                perlen = np.asarray(dis_pkg.perlen.array).flatten()
                nstp = np.asarray(dis_pkg.nstp.array).flatten()
                tsmult = np.asarray(dis_pkg.tsmult.array).flatten()
                spd = []
                for i in range(len(perlen)):
                    spd.append({
                        "perlen": float(perlen[i]),
                        "nstp": int(nstp[i]),
                        "tsmult": float(tsmult[i]),
                    })
                result.stress_period_data = spd
            except Exception:
                pass

            try:
                itmuni = dis_pkg.itmuni
                result.time_unit = _ITMUNI_MAP.get(itmuni, "undefined")
            except Exception:
                pass

        # Collect package information
        for pkg in model.packagelist:
            pkg_name = pkg.name[0].upper() if isinstance(pkg.name, list) else pkg.name.upper()
            result.packages[pkg_name] = {
                "name": pkg_name,
                "type": pkg_name,
            }

        # Extract coordinate reference from modelgrid
        try:
            grid = model.modelgrid
            if grid.xoffset is not None and grid.xoffset != 0:
                result.xoff = float(grid.xoffset)
            if grid.yoffset is not None and grid.yoffset != 0:
                result.yoff = float(grid.yoffset)
            if grid.angrot is not None and grid.angrot != 0:
                result.angrot = float(grid.angrot)
            if hasattr(grid, 'epsg') and grid.epsg:
                result.epsg = int(grid.epsg)
            if hasattr(grid, 'length_unit') and grid.length_unit:
                result.length_unit = str(grid.length_unit)
        except Exception as e:
            logger.debug(f"MFUSG coordinate extraction from modelgrid failed: {e}")

        # Fallback: usgs.model.reference file
        try:
            ref_info = _parse_model_reference_file(model_dir)
            if ref_info.get("xoff") and not result.xoff:
                result.xoff = ref_info["xoff"]
            if ref_info.get("yoff") and not result.yoff:
                result.yoff = ref_info["yoff"]
            if ref_info.get("angrot") and not result.angrot:
                result.angrot = ref_info["angrot"]
            if ref_info.get("epsg") and not result.epsg:
                result.epsg = ref_info["epsg"]
            if ref_info.get("start_date") and not result.start_date:
                result.start_date = ref_info["start_date"]
        except Exception as e:
            logger.debug(f"MFUSG model reference file parse failed: {e}")

        # NAM fallback: parse NAM file comments for spatial reference
        if nam_file:
            try:
                nam_info = _parse_nam_spatial_reference(nam_file)
                if nam_info.get("xoff") and not result.xoff:
                    result.xoff = nam_info["xoff"]
                if nam_info.get("yoff") and not result.yoff:
                    result.yoff = nam_info["yoff"]
                if nam_info.get("angrot") and not result.angrot:
                    result.angrot = nam_info["angrot"]
                if nam_info.get("epsg") and not result.epsg:
                    result.epsg = nam_info["epsg"]
                if nam_info.get("start_date"):
                    result.start_date = nam_info["start_date"]
            except Exception as e:
                logger.debug(f"MFUSG NAM spatial reference parse failed: {e}")

        result.is_valid = True

    except Exception as e:
        result.errors.append(f"Failed to load MODFLOW-USG model: {str(e)}")

    return result


def validate_model(model_dir: Path) -> ValidationResult:
    """
    Validate a MODFLOW model in the given directory.

    Args:
        model_dir: Path to directory containing model files

    Returns:
        ValidationResult with model information or errors
    """
    # First, detect model type
    model_type = detect_model_type(model_dir)

    if model_type is None:
        return ValidationResult(
            is_valid=False,
            errors=["Could not detect MODFLOW model type. No NAM or mfsim.nam file found."],
        )

    # Load based on model type
    if model_type == "mf6":
        return load_mf6_model(model_dir)
    elif model_type == "mfusg":
        return load_mfusg_model(model_dir)
    else:
        return load_mf2005_model(model_dir, model_type)


def extract_and_validate_zip(zip_data: bytes) -> tuple[ValidationResult, Optional[str]]:
    """
    Extract a ZIP file and validate the MODFLOW model inside.

    Handles both Windows (backslash) and Linux (forward slash) paths in:
    - ZIP file entry names
    - Path references inside MODFLOW input files (OPEN/CLOSE, etc.)

    Args:
        zip_data: Raw bytes of the ZIP file

    Returns:
        Tuple of (ValidationResult, extracted_dir_name or None)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extracted = temp_path / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)

        # Extract ZIP with normalized paths (handles Windows backslashes)
        try:
            file_count, _ = extract_zip_with_normalized_paths(zip_data, extracted)
        except zipfile.BadZipFile:
            return (
                ValidationResult(is_valid=False, errors=["Invalid or corrupted ZIP file"]),
                None,
            )
        except Exception as e:
            return (
                ValidationResult(is_valid=False, errors=[f"Failed to extract ZIP: {str(e)}"]),
                None,
            )

        # Find the model directory (might be in a subdirectory)
        model_dir = find_model_directory(extracted)

        if model_dir is None:
            return (
                ValidationResult(
                    is_valid=False,
                    errors=["No MODFLOW model files found in ZIP"],
                ),
                None,
            )

        # Normalize paths inside MODFLOW input files
        # (converts backslashes in OPEN/CLOSE references, external file paths, etc.)
        try:
            normalize_all_model_files(model_dir)
        except Exception as e:
            logger.debug(f"Path normalization failed (continuing): {e}")

        # Validate the model
        result = validate_model(model_dir)

        return result, str(model_dir.relative_to(extracted)) if model_dir != extracted else ""


def find_model_directory(base_dir: Path) -> Optional[Path]:
    """
    Find the directory containing MODFLOW model files.

    The model might be in the root of the ZIP or in a subdirectory.
    """
    # Check if model files are in the base directory
    if detect_model_type(base_dir) is not None:
        return base_dir

    # Check immediate subdirectories
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            if detect_model_type(subdir) is not None:
                return subdir

    # Check one more level deep
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir():
                    if detect_model_type(subsubdir) is not None:
                        return subsubdir

    return None
