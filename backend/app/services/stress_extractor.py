"""
Extract per-stress-period package rates from model input files.

Correlates with convergence difficulty by summarizing stress magnitudes
per package per stress period.
"""

import tempfile
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np

from app.config import get_settings
from app.services.storage import get_storage_service

settings = get_settings()

# Stress packages to extract rates from
MF6_STRESS_PACKAGES = {"WEL", "MAW", "RCH", "EVT", "SFR", "GHB", "CHD", "DRN", "RIV", "LAK", "UZF"}
CLASSIC_STRESS_PACKAGES = {"WEL", "RCH", "EVT", "SFR", "GHB", "CHD", "DRN", "RIV", "LAK", "MNW2", "UZF"}


def extract_stress_summary(
    project_id: str,
    storage_path: str,
    model_type: str,
    nper: int,
    stress_period_data: Optional[list] = None,
) -> dict:
    """
    Extract per-stress-period package rate summaries from model files.

    Downloads model from MinIO, loads with FloPy, and extracts rates.
    Keeps temp dir alive during extraction to handle FloPy lazy loading.

    Args:
        project_id: Project UUID string
        storage_path: MinIO prefix for model files
        model_type: One of mf6, mf2005, mfnwt, mfusg
        nper: Number of stress periods
        stress_period_data: List of {perlen, nstp, tsmult} per SP

    Returns:
        Dict with packages list and per-period rate data
    """
    storage = get_storage_service()

    # Download model files to temp dir (keep alive for FloPy lazy loading)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            objects = storage.list_objects(
                settings.minio_bucket_models,
                prefix=storage_path,
                recursive=True,
            )
            for obj_name in objects:
                rel_path = obj_name[len(storage_path):].lstrip("/")
                local_path = temp_path / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                storage.download_to_file(
                    settings.minio_bucket_models, obj_name, local_path
                )
        except Exception as e:
            print(f"Warning: Failed to download model files: {e}")
            return {"packages": [], "periods": []}

        if model_type == "mf6":
            return _extract_mf6_stress(temp_path, nper, stress_period_data)
        else:
            return _extract_classic_stress(temp_path, model_type, nper, stress_period_data)


def _extract_mf6_stress(
    model_dir: Path,
    nper: int,
    stress_period_data: Optional[list],
) -> dict:
    """Extract stress data from MF6 model."""
    try:
        import flopy
        sim = flopy.mf6.MFSimulation.load(
            sim_ws=str(model_dir),
            verbosity_level=0,
        )
    except Exception as e:
        print(f"Warning: Could not load MF6 model for stress extraction: {e}")
        return {"packages": [], "periods": []}

    # Get the first (usually only) groundwater flow model
    model_names = list(sim.model_names)
    if not model_names:
        return {"packages": [], "periods": []}

    gwf = sim.get_model(model_names[0])
    found_packages = set()
    package_data: dict[str, list] = {}  # pkg_name -> list of per-SP dicts

    for pkg in gwf.packagelist:
        pkg_type = getattr(pkg, "package_type", "").upper()
        if pkg_type not in MF6_STRESS_PACKAGES:
            continue

        found_packages.add(pkg_type)
        sp_rates = []

        for kper in range(nper):
            sp_info = _extract_mf6_package_sp(pkg, pkg_type, kper)
            sp_rates.append(sp_info)

        package_data[pkg_type] = sp_rates

    # Build periods list
    periods = []
    for kper in range(nper):
        period_info: dict = {"kper": kper}

        # Add stress period timing info
        if stress_period_data and kper < len(stress_period_data):
            spd = stress_period_data[kper]
            if isinstance(spd, dict):
                period_info["perlen"] = spd.get("perlen", 0)
                period_info["nstp"] = spd.get("nstp", 1)
                period_info["tsmult"] = spd.get("tsmult", 1.0)

        for pkg_name, sp_rates in package_data.items():
            if kper < len(sp_rates):
                period_info[pkg_name] = sp_rates[kper]

        periods.append(period_info)

    return {
        "packages": sorted(found_packages),
        "periods": periods,
    }


def _extract_mf6_package_sp(pkg, pkg_type: str, kper: int) -> dict:
    """Extract rate summary for a single MF6 package stress period."""
    try:
        spd = pkg.stress_period_data.get_data(kper)
    except Exception:
        return {"total_rate": 0.0, "n_active": 0}

    if spd is None:
        return {"total_rate": 0.0, "n_active": 0}

    result: dict = {"n_active": 0, "total_rate": 0.0}

    if isinstance(spd, np.recarray) and len(spd) > 0:
        result["n_active"] = len(spd)

        # Find rate/flux column
        rate_col = None
        for col in ("q", "rate", "flux", "recharge", "surf_rate", "pet"):
            if col in spd.dtype.names:
                rate_col = col
                break

        if rate_col:
            rates = spd[rate_col]
            valid = rates[np.isfinite(rates)]
            if valid.size > 0:
                result["total_rate"] = float(np.sum(valid))
                result["mean_rate"] = float(np.mean(valid))

    elif isinstance(spd, dict):
        # Array-based package (RCH, EVT)
        for key, arr in spd.items():
            if isinstance(arr, np.ndarray):
                valid = arr[np.isfinite(arr)]
                if valid.size > 0:
                    result["total_rate"] = float(np.sum(valid))
                    result["mean_rate"] = float(np.mean(valid))
                    result["n_active"] = int(valid.size)

    return result


def _extract_classic_stress(
    model_dir: Path,
    model_type: str,
    nper: int,
    stress_period_data: Optional[list],
) -> dict:
    """Extract stress data from MF2005/NWT/USG models."""
    try:
        import flopy
        from app.services.modflow import find_nam_file

        nam_file = find_nam_file(model_dir)
        if not nam_file:
            return {"packages": [], "periods": []}

        if model_type == "mfusg":
            model = flopy.modflow.Modflow.load(
                nam_file.name,
                model_ws=str(model_dir),
                check=False,
                forgive=True,
                load_only=None,
            )
        elif model_type == "mfnwt":
            model = flopy.modflow.Modflow.load(
                nam_file.name,
                model_ws=str(model_dir),
                version="mfnwt",
                check=False,
                forgive=True,
            )
        else:
            model = flopy.modflow.Modflow.load(
                nam_file.name,
                model_ws=str(model_dir),
                check=False,
                forgive=True,
            )
    except Exception as e:
        print(f"Warning: Could not load classic model: {e}")
        return {"packages": [], "periods": []}

    found_packages = set()
    package_data: dict[str, list] = {}

    for pkg in model.packagelist:
        pkg_type = type(pkg).__name__.upper()
        # Map FloPy class names to standard names
        pkg_map = {
            "MODFLOWWEL": "WEL", "MODFLOWRCH": "RCH", "MODFLOWEVT": "EVT",
            "MODFLOWGHB": "GHB", "MODFLOWCHD": "CHD", "MODFLOWDRN": "DRN",
            "MODFLOWRIV": "RIV", "MODFLOWSFR2": "SFR", "MODFLOWLAK": "LAK",
            "MODFLOWMNW2": "MNW2", "MODFLOWUZF1": "UZF",
        }
        short_name = pkg_map.get(pkg_type, "")
        if short_name not in CLASSIC_STRESS_PACKAGES:
            continue

        found_packages.add(short_name)
        sp_rates = []

        for kper in range(nper):
            sp_info = _extract_classic_package_sp(pkg, short_name, kper)
            sp_rates.append(sp_info)

        package_data[short_name] = sp_rates

    periods = []
    for kper in range(nper):
        period_info: dict = {"kper": kper}
        if stress_period_data and kper < len(stress_period_data):
            spd = stress_period_data[kper]
            if isinstance(spd, dict):
                period_info["perlen"] = spd.get("perlen", 0)
                period_info["nstp"] = spd.get("nstp", 1)
                period_info["tsmult"] = spd.get("tsmult", 1.0)

        for pkg_name, sp_rates in package_data.items():
            if kper < len(sp_rates):
                period_info[pkg_name] = sp_rates[kper]

        periods.append(period_info)

    return {
        "packages": sorted(found_packages),
        "periods": periods,
    }


def _extract_classic_package_sp(pkg, pkg_type: str, kper: int) -> dict:
    """Extract rate summary for a classic MODFLOW package stress period."""
    result: dict = {"n_active": 0, "total_rate": 0.0}

    try:
        if hasattr(pkg, "stress_period_data") and pkg.stress_period_data is not None:
            spd = pkg.stress_period_data.get(kper)
            if spd is None:
                # Try getting from previous period (MODFLOW reuses)
                for prev in range(kper - 1, -1, -1):
                    spd = pkg.stress_period_data.get(prev)
                    if spd is not None:
                        break

            if spd is not None and isinstance(spd, np.recarray) and len(spd) > 0:
                result["n_active"] = len(spd)
                rate_col = None
                for col in ("flux", "q", "cond", "bhead", "stage", "rech", "surf"):
                    if col in spd.dtype.names:
                        rate_col = col
                        break
                if rate_col:
                    rates = spd[rate_col]
                    valid = rates[np.isfinite(rates)]
                    if valid.size > 0:
                        result["total_rate"] = float(np.sum(valid))
                        result["mean_rate"] = float(np.mean(valid))

        elif pkg_type in ("RCH", "EVT") and hasattr(pkg, "rech"):
            # Array-based recharge
            arr = pkg.rech[kper].array if hasattr(pkg.rech[kper], "array") else pkg.rech[kper]
            if isinstance(arr, np.ndarray):
                valid = arr[np.isfinite(arr)]
                if valid.size > 0:
                    result["total_rate"] = float(np.sum(valid))
                    result["mean_rate"] = float(np.mean(valid))
                    result["n_active"] = int(valid.size)

    except Exception:
        pass

    return result
