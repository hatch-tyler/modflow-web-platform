"""PEST++ workspace setup service.

Handles parameter discovery, PEST workspace construction
(forward_run.py, template/instruction/PST files), and result parsing.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from app.services.mesh import get_array_data, get_list_package_data, load_model_from_directory

# Adjustable parameter property definitions
# Array-based properties (per-layer parameters)
PARAM_PROPERTIES = [
    {
        "id": "hk",
        "name": "Horizontal Hydraulic Conductivity",
        "short": "HK",
        "array_key": "hk",
        "transform": "log",
        "default_lower": 0.01,
        "default_upper": 100.0,
        "package_type": "array",
    },
    {
        "id": "vka",
        "name": "Vertical Hydraulic Conductivity",
        "short": "VKA",
        "array_key": "vka",
        "transform": "log",
        "default_lower": 0.01,
        "default_upper": 100.0,
        "package_type": "array",
    },
    {
        "id": "ss",
        "name": "Specific Storage",
        "short": "SS",
        "array_key": "ss",
        "transform": "log",
        "default_lower": 0.01,
        "default_upper": 100.0,
        "package_type": "array",
    },
    {
        "id": "sy",
        "name": "Specific Yield",
        "short": "SY",
        "array_key": "sy",
        "transform": "none",
        "default_lower": 0.5,
        "default_upper": 2.0,
        "package_type": "array",
    },
]

# List-based properties (single multiplier for all package elements)
LIST_PARAM_PROPERTIES = [
    {
        "id": "hfb",
        "name": "HFB Hydraulic Characteristic",
        "short": "HFB",
        "array_key": "hfb",
        "transform": "log",
        "default_lower": 0.01,
        "default_upper": 100.0,
        "package_type": "list",
        "multiplier_mode": "base_values",
    },
    {
        "id": "sfr_cond",
        "name": "SFR Streambed Conductance",
        "short": "SFRC",
        "array_key": "sfr_cond",
        "transform": "log",
        "default_lower": 0.01,
        "default_upper": 100.0,
        "package_type": "list",
        "multiplier_mode": "base_values",
    },
    {
        "id": "ghb_cond",
        "name": "GHB Conductance",
        "short": "GHBC",
        "array_key": "ghb_cond",
        "transform": "log",
        "default_lower": 0.01,
        "default_upper": 100.0,
        "package_type": "list",
        "multiplier_mode": "stress_period",
    },
    {
        "id": "riv_cond",
        "name": "River Conductance",
        "short": "RIVC",
        "array_key": "riv_cond",
        "transform": "log",
        "default_lower": 0.01,
        "default_upper": 100.0,
        "package_type": "list",
        "multiplier_mode": "stress_period",
    },
    {
        "id": "drn_cond",
        "name": "Drain Conductance",
        "short": "DRNC",
        "array_key": "drn_cond",
        "transform": "log",
        "default_lower": 0.01,
        "default_upper": 100.0,
        "package_type": "list",
        "multiplier_mode": "stress_period",
    },
    {
        "id": "rch",
        "name": "Recharge Rate Multiplier",
        "short": "RCH",
        "array_key": "rch",
        "transform": "none",
        "default_lower": 0.5,
        "default_upper": 2.0,
        "package_type": "list",
        "multiplier_mode": "stress_period",
    },
    {
        "id": "evt",
        "name": "ET Rate Multiplier",
        "short": "EVT",
        "array_key": "evt",
        "transform": "none",
        "default_lower": 0.5,
        "default_upper": 2.0,
        "package_type": "list",
        "multiplier_mode": "stress_period",
    },
]


def discover_parameters(model_dir: Path) -> list[dict]:
    """
    Discover adjustable parameters from a MODFLOW model.

    Loads the model with FloPy and checks for available parameter arrays
    (HK, VKA, SS, SY) and list-based packages (HFB, SFR).
    Returns per-layer statistics and suggested bounds.

    Args:
        model_dir: Directory containing model files.

    Returns:
        List of parameter descriptors with package_type indicator.
    """
    model = load_model_from_directory(model_dir)
    if model is None:
        return []

    params = []

    # Discover array-based parameters (per-layer)
    for prop_def in PARAM_PROPERTIES:
        arr = get_array_data(model, prop_def["array_key"])
        if arr is None:
            continue

        nlay = arr.shape[0]
        for lay in range(nlay):
            layer_arr = arr[lay].flatten().astype(float)
            valid = layer_arr[np.isfinite(layer_arr) & (layer_arr > 0)]

            params.append(
                {
                    "property": prop_def["id"],
                    "property_name": prop_def["name"],
                    "short_name": prop_def["short"],
                    "layer": lay,
                    "package_type": prop_def.get("package_type", "array"),
                    "stats": {
                        "mean": round(float(np.mean(valid)), 6)
                        if valid.size > 0
                        else None,
                        "min": round(float(np.min(valid)), 6)
                        if valid.size > 0
                        else None,
                        "max": round(float(np.max(valid)), 6)
                        if valid.size > 0
                        else None,
                    },
                    "suggested_transform": prop_def["transform"],
                    "suggested_lower": prop_def["default_lower"],
                    "suggested_upper": prop_def["default_upper"],
                }
            )

    # Discover list-based parameters (single multiplier for whole package)
    for prop_def in LIST_PARAM_PROPERTIES:
        pkg_data = get_list_package_data(model, prop_def["array_key"])
        if pkg_data is None:
            continue

        params.append(
            {
                "property": prop_def["id"],
                "property_name": prop_def["name"],
                "short_name": prop_def["short"],
                "layer": None,  # List packages have no layer
                "package_type": "list",
                "count": pkg_data["count"],
                "stats": pkg_data["stats"],
                "suggested_transform": prop_def["transform"],
                "suggested_lower": prop_def["default_lower"],
                "suggested_upper": prop_def["default_upper"],
            }
        )

    return params


def build_pest_workspace(
    workspace_dir: Path,
    model_dir: Path,
    model_type: str,
    nam_file: str,
    executable: str,
    parameters: list[dict],
    observations: list[dict],
    pest_settings: dict,
) -> Path:
    """
    Build a complete PEST++ workspace with all required files.

    Creates: forward_run.py, mult_pars.dat.tpl, sim_obs.dat.ins,
    pest_run.pst, base_arrays/, base_model/, and pest_config.json.

    Supports three parameter types:
    - Array-based multipliers (hk, vka, ss, sy per layer)
    - List-based multipliers (hfb, sfr_cond for entire package)
    - Pilot points (spatial heterogeneity via kriging)

    Args:
        workspace_dir: Target directory for PEST workspace.
        model_dir: Directory with original model files.
        model_type: MODFLOW model type string (mf6, mf2005, etc.).
        nam_file: Model name file.
        executable: Path to MODFLOW executable.
        parameters: List of parameter configurations.
        observations: List of observation configurations.
        pest_settings: PEST++ control settings.

    Returns:
        Path to the generated .pst file.
    """
    from app.services.pilot_points import (
        create_pilot_point_grid,
        setup_kriging,
        get_default_variogram,
    )

    # Copy model files to workspace
    for item in model_dir.iterdir():
        dest = workspace_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest)

    # Create base_model/ with original files for restoration during runs
    base_dir = workspace_dir / "base_model"
    base_dir.mkdir()
    for item in model_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, base_dir / item.name)

    # Load model to extract base arrays
    model = load_model_from_directory(workspace_dir)
    if model is None:
        raise ValueError("Failed to load model for PEST setup")

    base_arrays_dir = workspace_dir / "base_arrays"
    base_arrays_dir.mkdir()

    # Get grid info for pilot points
    ibound = get_array_data(model, "ibound")
    if ibound is None:
        ibound = get_array_data(model, "idomain")
    if ibound is None:
        # Assume all active
        dis = get_array_data(model, "top")
        if dis is not None:
            nrow, ncol = dis.shape[-2:]
            nlay = 1
            ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)

    # Get grid coordinates for kriging
    delr = get_array_data(model, "delr") if hasattr(model, "dis") or hasattr(model, "modelgrid") else None
    delc = get_array_data(model, "delc") if hasattr(model, "dis") or hasattr(model, "modelgrid") else None

    # Try to get delr/delc from model grid
    try:
        if hasattr(model, "modelgrid"):
            mg = model.modelgrid
            if hasattr(mg, "delr"):
                delr = np.array(mg.delr)
            if hasattr(mg, "delc"):
                delc = np.array(mg.delc)
        elif hasattr(model, "dis"):
            if hasattr(model.dis, "delr"):
                delr = np.array(model.dis.delr.array)
            if hasattr(model.dis, "delc"):
                delc = np.array(model.dis.delc.array)
    except Exception:
        pass

    param_info: dict[str, dict] = {}
    param_names: list[str] = []
    pilot_point_info: dict[str, dict] = {}
    list_param_info: dict[str, dict] = {}

    for param in parameters:
        package_type = param.get("package_type", "array")
        approach = param.get("approach", "multiplier")

        if package_type == "list":
            # List-based parameters (HFB, SFR, GHB, RIV, DRN, RCH, EVT)
            pname = param["property"]  # No layer suffix
            pkg_data = get_list_package_data(model, param["property"])
            if pkg_data is None:
                continue

            # Look up multiplier_mode from LIST_PARAM_PROPERTIES
            mult_mode = "base_values"
            for lpp in LIST_PARAM_PROPERTIES:
                if lpp["id"] == param["property"]:
                    mult_mode = lpp.get("multiplier_mode", "base_values")
                    break

            if mult_mode == "base_values":
                # Save base values as .npy for HFB/SFR
                np.save(base_arrays_dir / f"{pname}_base.npy", pkg_data["base_values"])

            list_param_info[pname] = {
                "property": param["property"],
                "count": pkg_data["count"],
                "multiplier_mode": mult_mode,
            }
            param_info[pname] = {
                "property": param["property"],
                "layer": None,
                "package_type": "list",
            }
            param_names.append(pname)

        elif approach == "pilotpoints":
            # Pilot points approach
            layer = param["layer"]
            pp_space = param.get("pp_space", 10)
            variogram = param.get("variogram", {})

            # Create pilot point grid
            pp_df = create_pilot_point_grid(
                ibound=ibound,
                pp_space=pp_space,
                param_name=param["property"],
                layer=layer,
                output_dir=base_arrays_dir,
                delr=delr,
                delc=delc,
            )

            if len(pp_df) == 0:
                continue

            # Get grid coordinates for kriging
            nrow, ncol = ibound.shape[-2:]
            if delr is None:
                delr = np.ones(ncol)
            if delc is None:
                delc = np.ones(nrow)

            x_edges = np.concatenate([[0], np.cumsum(delr)])
            x_centers = ((x_edges[:-1] + x_edges[1:]) / 2).tolist()

            y_edges = np.concatenate([[0], np.cumsum(delc)])
            y_centers = ((y_edges[:-1] + y_edges[1:]) / 2).tolist()
            y_centers = [np.sum(delc) - y for y in y_centers]  # Flip

            # Get default variogram if not specified
            if not variogram:
                grid_extent = max(np.sum(delr), np.sum(delc))
                variogram = get_default_variogram(param["property"], grid_extent)

            # Setup kriging
            pp_file = base_arrays_dir / f"{param['property']}_l{layer}.pp"
            factors_file = setup_kriging(
                pp_file=pp_file,
                variogram_config=variogram,
                grid_coords={
                    "x_centers": x_centers,
                    "y_centers": y_centers,
                    "nrow": nrow,
                    "ncol": ncol,
                },
                output_dir=base_arrays_dir,
            )

            # Save base array for this layer
            base_pname = f"{param['property']}_l{layer}"
            arr = get_array_data(model, param["property"])
            if arr is not None:
                np.save(base_arrays_dir / f"{base_pname}.npy", arr[layer])

            # Add each pilot point as a parameter
            pp_names_list = pp_df["name"].tolist()
            for pp_name in pp_names_list:
                param_names.append(pp_name)
                param_info[pp_name] = {
                    "property": param["property"],
                    "layer": layer,
                    "package_type": "pilotpoint",
                    "base_param": base_pname,
                }

            pilot_point_info[base_pname] = {
                "property": param["property"],
                "layer": layer,
                "pp_names": pp_names_list,
                "factors_file": str(factors_file.name),
                "nrow": nrow,
                "ncol": ncol,
            }

        else:
            # Standard multiplier approach
            pname = f"{param['property']}_l{param['layer']}"
            arr = get_array_data(model, param["property"])
            if arr is None:
                continue

            layer_arr = arr[param["layer"]]
            np.save(base_arrays_dir / f"{pname}.npy", layer_arr)

            param_info[pname] = {
                "property": param["property"],
                "layer": param["layer"],
                "package_type": "array",
            }
            param_names.append(pname)

    if not param_names:
        raise ValueError("No valid parameters could be extracted from the model")

    # Build observation info
    obs_info: list[dict] = []
    obs_names: list[str] = []
    for obs in observations:
        obs_names.append(obs["name"])
        obs_info.append(
            {
                "name": obs["name"],
                "obsval": obs["value"],
                "time": obs["time"],
                "layer": obs["layer"],
                "row": obs.get("row"),
                "col": obs.get("col"),
                "node": obs.get("node"),
            }
        )

    if not obs_info:
        raise ValueError("No observations provided for calibration")

    # List of model files to restore each forward run
    model_files = [f.name for f in model_dir.iterdir() if f.is_file()]

    # Write pest_config.json (read by forward_run.py)
    config = {
        "model_type": model_type,
        "nam_file": nam_file,
        "executable": executable,
        "param_info": param_info,
        "param_names": param_names,
        "obs_info": obs_info,
        "model_files": model_files,
        "pilot_point_info": pilot_point_info,
        "list_param_info": list_param_info,
    }
    (workspace_dir / "pest_config.json").write_text(json.dumps(config, indent=2))

    # Write forward_run.py
    (workspace_dir / "forward_run.py").write_text(_generate_forward_run_script())

    # Write template file
    _write_template_file(workspace_dir / "mult_pars.dat.tpl", param_names)

    # Write initial parameter file
    # Build initial values list matching param_names order
    initial_values = []
    for pname in param_names:
        pinfo = param_info.get(pname, {})
        pkg_type = pinfo.get("package_type", "array")
        if pkg_type == "pilotpoint":
            # Pilot points start at 1.0 (multiplier)
            initial_values.append(1.0)
        else:
            # Find matching parameter in original list
            found = False
            for p in parameters:
                if pkg_type == "list":
                    if p["property"] == pname:
                        initial_values.append(p.get("initial_value", 1.0))
                        found = True
                        break
                else:
                    check_name = f"{p['property']}_l{p.get('layer', 0)}"
                    if check_name == pname:
                        initial_values.append(p.get("initial_value", 1.0))
                        found = True
                        break
            if not found:
                initial_values.append(1.0)

    _write_param_file(workspace_dir / "mult_pars.dat", param_names, initial_values)

    # Write instruction file
    _write_instruction_file(workspace_dir / "sim_obs.dat.ins", obs_names)

    # Write placeholder sim_obs.dat
    with open(workspace_dir / "sim_obs.dat", "w") as f:
        for obs in obs_info:
            f.write(f'{obs["obsval"]:.10e}\n')

    # Write PST control file
    pst_path = workspace_dir / "pest_run.pst"
    _write_pst_file(
        pst_path,
        parameters,
        param_names,
        observations,
        obs_names,
        pest_settings,
        param_info,
    )

    return pst_path


def _generate_forward_run_script() -> str:
    """Generate the forward_run.py script that PEST++ executes each iteration."""
    return r'''#!/usr/bin/env python3
"""PEST++ forward model run script. Auto-generated.

Supports three parameter types:
- Array-based multipliers (standard layer parameters)
- List-based multipliers (HFB, SFR package parameters)
- Pilot points (spatially variable via kriging)
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


def main():
    with open("pest_config.json") as f:
        config = json.load(f)

    model_type = config["model_type"]
    executable = config["executable"]
    nam_file = config.get("nam_file")
    param_info = config["param_info"]
    obs_info = config["obs_info"]
    model_files = config["model_files"]
    pilot_point_info = config.get("pilot_point_info", {})
    list_param_info = config.get("list_param_info", {})

    # Read parameter values written by PEST++
    param_values = {}
    with open("mult_pars.dat") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                param_values[parts[0]] = float(parts[1])

    # Restore original model files before modification
    for fn in model_files:
        src = os.path.join("base_model", fn)
        if os.path.exists(src):
            shutil.copy2(src, fn)

    # Load model with FloPy
    import flopy

    if model_type == "mf6":
        sim = flopy.mf6.MFSimulation.load(sim_ws=".", verbosity_level=0)
        model = sim.get_model()
    elif model_type == "mfusg":
        import flopy.mfusg as mfusg
        model = mfusg.MfUsg.load(nam_file, model_ws=".", check=False, verbose=False)
    else:
        model = flopy.modflow.Modflow.load(
            nam_file, model_ws=".", check=False, verbose=False
        )

    # Track which base arrays have been processed for pilot points
    processed_pp = set()

    # Apply pilot point parameters (krige to full grid, then apply as multiplier)
    for base_pname, pp_info in pilot_point_info.items():
        # Collect pilot point values
        pp_names = pp_info["pp_names"]
        pp_values = {name: param_values.get(name, 1.0) for name in pp_names}

        # Apply kriging to get full grid
        factors_file = os.path.join("base_arrays", pp_info["factors_file"])
        nrow = pp_info["nrow"]
        ncol = pp_info["ncol"]
        multiplier_grid = _apply_kriged_values(pp_values, factors_file, nrow, ncol)

        # Load base array and apply multiplier grid
        base = np.load(os.path.join("base_arrays", f"{base_pname}.npy"))
        new_arr = base * multiplier_grid

        # Set in model
        _set_array(model, model_type, pp_info["property"], pp_info["layer"], new_arr)
        processed_pp.add(base_pname)

    # Apply standard array multipliers (skip those handled by pilot points)
    for pname, pinfo in param_info.items():
        pkg_type = pinfo.get("package_type", "array")

        if pkg_type == "pilotpoint":
            # Already handled above via pilot_point_info
            continue

        if pkg_type == "list":
            # List-based parameters handled separately
            continue

        if pname in processed_pp:
            continue

        if pname not in param_values:
            continue

        mult = param_values[pname]
        base_file = os.path.join("base_arrays", f"{pname}.npy")
        if not os.path.exists(base_file):
            continue

        base = np.load(base_file)
        new_arr = base * mult
        _set_array(model, model_type, pinfo["property"], pinfo["layer"], new_arr)

    # Apply list-based multipliers (HFB, SFR, GHB, RIV, DRN, RCH, EVT)
    for pname, linfo in list_param_info.items():
        if pname not in param_values:
            continue
        mult = param_values[pname]
        mult_mode = linfo.get("multiplier_mode", "base_values")

        if mult_mode == "stress_period":
            # Stress-period multipliers: apply directly to model package
            _set_list_package_multiplier(model, model_type, linfo["property"], mult)
        else:
            # Base-values mode: load saved .npy and multiply
            base_file = os.path.join("base_arrays", f"{pname}_base.npy")
            if not os.path.exists(base_file):
                continue
            base = np.load(base_file)
            new_vals = base * mult
            _set_list_package(model, model_type, linfo["property"], new_vals)

    # Write modified model
    if model_type == "mf6":
        sim.write_simulation()
    else:
        model.write_input()

    # Run MODFLOW
    if model_type == "mf6":
        cmd = [executable]
    else:
        cmd = [executable, nam_file]

    subprocess.run(cmd, capture_output=True, text=True)

    # Extract simulated observations
    _extract_observations(model_type, obs_info)


def _apply_kriged_values(pp_values, factors_file, nrow, ncol):
    """Interpolate pilot point values to full grid using kriging factors."""
    factors_df = pd.read_csv(factors_file)

    # Initialize output grid with 1.0 (neutral multiplier)
    result = np.ones((nrow, ncol), dtype=np.float64)

    # Group factors by target cell
    for (row, col), group in factors_df.groupby(["target_row", "target_col"]):
        value = 0.0
        for _, factor in group.iterrows():
            pp_name = factor["pp_name"]
            weight = factor["weight"]
            if pp_name in pp_values:
                value += weight * pp_values[pp_name]
        result[int(row), int(col)] = value

    return result


def _set_array(model, model_type, prop, layer, arr):
    """Set a parameter array in the model."""
    if model_type == "mf6":
        if prop in ("hk", "k"):
            model.npf.k.set_data(arr, layer=layer)
        elif prop in ("vka", "k33"):
            model.npf.k33.set_data(arr, layer=layer)
        elif prop == "ss":
            model.get_package("STO").ss.set_data(arr, layer=layer)
        elif prop == "sy":
            model.get_package("STO").sy.set_data(arr, layer=layer)
    else:
        pkg = getattr(model, "lpf", None) or getattr(model, "upw", None)
        if not pkg:
            return
        target = getattr(pkg, prop, None)
        if target is not None:
            target[layer] = arr


def _set_list_package(model, model_type, prop, new_vals):
    """Set list-based package values (HFB, SFR)."""
    if prop == "hfb":
        _set_hfb_values(model, model_type, new_vals)
    elif prop == "sfr_cond":
        _set_sfr_cond_values(model, model_type, new_vals)


def _set_hfb_values(model, model_type, new_vals):
    """Apply multiplied HFB values back to model."""
    if model_type == "mf6":
        try:
            hfb = model.get_package("HFB")
            if hfb is None:
                return

            spd = hfb.stress_period_data
            if spd is None:
                return

            # Get data for stress period 0
            if hasattr(spd, 'get_data'):
                data = spd.get_data(0)
            else:
                data = spd.data.get(0, None)

            if data is None or len(data) != len(new_vals):
                return

            # Update hydchr values
            data['hydchr'] = new_vals.astype(data['hydchr'].dtype)
            spd.set_data(data, 0)
        except Exception as e:
            print(f"Warning: Could not set HFB values: {e}")
    else:
        try:
            hfb = getattr(model, 'hfb6', None)
            if hfb is None:
                return

            hfb_data = hfb.hfb_data
            if hfb_data is None:
                return

            # Update hydchr column (last column)
            if isinstance(hfb_data, list):
                for i, period_data in enumerate(hfb_data):
                    if period_data is not None and len(period_data) > 0:
                        arr = np.array(period_data)
                        if arr.ndim == 2 and len(new_vals) == len(arr):
                            arr[:, -1] = new_vals
                            hfb_data[i] = arr
                        break
            elif isinstance(hfb_data, np.ndarray):
                if hfb_data.ndim == 2 and len(new_vals) == len(hfb_data):
                    hfb_data[:, -1] = new_vals
        except Exception as e:
            print(f"Warning: Could not set HFB values: {e}")


def _set_sfr_cond_values(model, model_type, new_vals):
    """Apply multiplied SFR conductance values back to model."""
    if model_type == "mf6":
        try:
            sfr = model.get_package("SFR")
            if sfr is None:
                return

            pkgdata = sfr.packagedata
            if pkgdata is None:
                return

            if hasattr(pkgdata, 'get_data'):
                data = pkgdata.get_data()
            else:
                data = pkgdata.data

            if data is None:
                return

            # Find and update the conductivity field
            for field in ['rhk', 'hk', 'rbed_k', 'rbdk']:
                if field in data.dtype.names:
                    if len(data[field]) == len(new_vals):
                        data[field] = new_vals.astype(data[field].dtype)
                    break

            pkgdata.set_data(data)
        except Exception as e:
            print(f"Warning: Could not set SFR values: {e}")
    else:
        try:
            sfr = getattr(model, 'sfr', None)
            if sfr is None:
                return

            reach_data = getattr(sfr, 'reach_data', None)
            if reach_data is None:
                return

            if 'strhc1' in reach_data.dtype.names:
                if len(reach_data['strhc1']) == len(new_vals):
                    reach_data['strhc1'] = new_vals.astype(reach_data['strhc1'].dtype)
        except Exception as e:
            print(f"Warning: Could not set SFR values: {e}")


def _set_list_package_multiplier(model, model_type, prop, mult):
    """Apply a multiplier directly to a stress-period-varying package."""
    if prop == "ghb_cond":
        _set_ghb_cond_multiplier(model, model_type, mult)
    elif prop == "riv_cond":
        _set_riv_cond_multiplier(model, model_type, mult)
    elif prop == "drn_cond":
        _set_drn_cond_multiplier(model, model_type, mult)
    elif prop == "rch":
        _set_rch_multiplier(model, model_type, mult)
    elif prop == "evt":
        _set_evt_multiplier(model, model_type, mult)


def _set_ghb_cond_multiplier(model, model_type, mult):
    """Multiply GHB conductance across all stress periods."""
    try:
        if model_type == "mf6":
            ghb = model.get_package("GHB")
            if ghb is None:
                return
            spd = ghb.stress_period_data
            if spd is None:
                return
            nper = model.simulation.tdis.nper.data if hasattr(model, 'simulation') else 1
            for kper in range(nper):
                try:
                    data = spd.get_data(kper)
                    if data is not None and 'cond' in data.dtype.names:
                        data['cond'] = data['cond'] * mult
                        spd.set_data(data, kper)
                except Exception:
                    pass
        else:
            ghb = getattr(model, 'ghb', None)
            if ghb is None:
                return
            for kper in list(ghb.stress_period_data.data.keys()):
                data = ghb.stress_period_data[kper]
                if data is not None and 'cond' in data.dtype.names:
                    data['cond'] = data['cond'] * mult
    except Exception as e:
        print(f"Warning: Could not set GHB multiplier: {e}")


def _set_riv_cond_multiplier(model, model_type, mult):
    """Multiply RIV conductance across all stress periods."""
    try:
        if model_type == "mf6":
            riv = model.get_package("RIV")
            if riv is None:
                return
            spd = riv.stress_period_data
            if spd is None:
                return
            nper = model.simulation.tdis.nper.data if hasattr(model, 'simulation') else 1
            for kper in range(nper):
                try:
                    data = spd.get_data(kper)
                    if data is not None and 'cond' in data.dtype.names:
                        data['cond'] = data['cond'] * mult
                        spd.set_data(data, kper)
                except Exception:
                    pass
        else:
            riv = getattr(model, 'riv', None)
            if riv is None:
                return
            for kper in list(riv.stress_period_data.data.keys()):
                data = riv.stress_period_data[kper]
                if data is not None and 'cond' in data.dtype.names:
                    data['cond'] = data['cond'] * mult
    except Exception as e:
        print(f"Warning: Could not set RIV multiplier: {e}")


def _set_drn_cond_multiplier(model, model_type, mult):
    """Multiply DRN conductance across all stress periods."""
    try:
        if model_type == "mf6":
            drn = model.get_package("DRN")
            if drn is None:
                return
            spd = drn.stress_period_data
            if spd is None:
                return
            nper = model.simulation.tdis.nper.data if hasattr(model, 'simulation') else 1
            for kper in range(nper):
                try:
                    data = spd.get_data(kper)
                    if data is not None and 'cond' in data.dtype.names:
                        data['cond'] = data['cond'] * mult
                        spd.set_data(data, kper)
                except Exception:
                    pass
        else:
            drn = getattr(model, 'drn', None)
            if drn is None:
                return
            for kper in list(drn.stress_period_data.data.keys()):
                data = drn.stress_period_data[kper]
                if data is not None and 'cond' in data.dtype.names:
                    data['cond'] = data['cond'] * mult
    except Exception as e:
        print(f"Warning: Could not set DRN multiplier: {e}")


def _set_rch_multiplier(model, model_type, mult):
    """Multiply Recharge rate array."""
    try:
        if model_type == "mf6":
            rch = model.get_package("RCHA")
            if rch is None:
                rch = model.get_package("RCH")
            if rch is None:
                return
            if hasattr(rch, 'recharge'):
                data = rch.recharge.get_data()
                if data is not None:
                    rch.recharge.set_data(np.array(data) * mult)
        else:
            rch = getattr(model, 'rch', None)
            if rch is None:
                return
            if hasattr(rch, 'rech') and rch.rech is not None:
                for kper in range(len(rch.rech)):
                    arr = rch.rech[kper].array
                    if arr is not None:
                        rch.rech[kper] = arr * mult
    except Exception as e:
        print(f"Warning: Could not set RCH multiplier: {e}")


def _set_evt_multiplier(model, model_type, mult):
    """Multiply Evapotranspiration rate array."""
    try:
        if model_type == "mf6":
            evt = model.get_package("EVTA")
            if evt is None:
                evt = model.get_package("EVT")
            if evt is None:
                return
            if hasattr(evt, 'rate'):
                data = evt.rate.get_data()
                if data is not None:
                    evt.rate.set_data(np.array(data) * mult)
        else:
            evt = getattr(model, 'evt', None)
            if evt is None:
                return
            if hasattr(evt, 'evtr') and evt.evtr is not None:
                for kper in range(len(evt.evtr)):
                    arr = evt.evtr[kper].array
                    if arr is not None:
                        evt.evtr[kper] = arr * mult
    except Exception as e:
        print(f"Warning: Could not set EVT multiplier: {e}")


def _extract_observations(model_type, obs_info):
    """Read simulated heads at observation points and write output file."""
    import flopy.utils

    # Find head file (.hds or .hed extension)
    hds_file = None
    for f in os.listdir("."):
        lower_f = f.lower()
        if lower_f.endswith(".hds") or lower_f.endswith(".hed"):
            hds_file = f
            break

    if not hds_file:
        _write_penalty_obs(obs_info)
        return

    try:
        if model_type == "mfusg":
            hds = flopy.utils.HeadUFile(hds_file)
        else:
            hds = flopy.utils.HeadFile(hds_file)

        times = hds.get_times()
        kstpkper = hds.get_kstpkper()

        sim_vals = []
        for obs in obs_info:
            try:
                idx = min(range(len(times)), key=lambda i: abs(times[i] - obs["time"]))
                ks, kp = kstpkper[idx]
                data = hds.get_data(kstpkper=(ks, kp))

                if model_type == "mfusg":
                    layers = data if isinstance(data, list) else [data]
                    val = float(layers[obs["layer"]].flatten()[obs["node"]])
                else:
                    val = float(data[obs["layer"], obs["row"], obs["col"]])

                # Check for dry/inactive cells
                if abs(val) > 1e20 or val < -880:
                    val = obs["obsval"] + 999.0
            except Exception:
                val = obs["obsval"] + 999.0
            sim_vals.append(val)

        hds.close()
    except Exception:
        sim_vals = [obs["obsval"] + 999.0 for obs in obs_info]

    with open("sim_obs.dat", "w") as f:
        for val in sim_vals:
            f.write(f"{val:.10e}\n")


def _write_penalty_obs(obs_info):
    """Write penalty observation values when model fails."""
    with open("sim_obs.dat", "w") as f:
        for obs in obs_info:
            f.write(f'{obs["obsval"] + 999.0:.10e}\n')


if __name__ == "__main__":
    main()
'''


def _write_template_file(path: Path, param_names: list[str]) -> None:
    """Write PEST++ template file for parameter multipliers."""
    lines = ["ptf ~"]
    for pname in param_names:
        # Format: name  ~  padded_param_name  ~
        # PEST++ replaces the ~...~ field with the parameter value
        padded = f"  {pname}  "
        lines.append(f"{pname}  ~{padded:>22s}~")
    path.write_text("\n".join(lines) + "\n")


def _write_param_file(
    path: Path, param_names: list[str], values: list[float]
) -> None:
    """Write parameter value file (initial values)."""
    lines = []
    for pname, val in zip(param_names, values):
        lines.append(f"{pname:22s}  {val:.10e}")
    path.write_text("\n".join(lines) + "\n")


def _write_instruction_file(path: Path, obs_names: list[str]) -> None:
    """Write PEST++ instruction file for reading simulated observations."""
    lines = ["pif @"]
    for oname in obs_names:
        lines.append(f"l1 !{oname}!")
    path.write_text("\n".join(lines) + "\n")


def _write_pst_file(
    path: Path,
    parameters: list[dict],
    param_names: list[str],
    observations: list[dict],
    obs_names: list[str],
    pest_settings: dict,
    param_info: Optional[dict] = None,
) -> None:
    """Write PEST++ control file (.pst).

    Handles three parameter types:
    - Array-based multipliers (one parameter per layer)
    - List-based multipliers (one parameter for package)
    - Pilot points (many parameters for spatial heterogeneity)
    """
    param_info = param_info or {}
    npar = len(param_names)
    nobs = len(obs_names)

    # Build parameter lookup by property for getting settings
    param_lookup = {}
    for p in parameters:
        pkg_type = p.get("package_type", "array")
        if pkg_type == "list":
            key = p["property"]
        else:
            key = f"{p['property']}_l{p.get('layer', 0)}"
        param_lookup[key] = p

    # Collect unique parameter and observation groups
    par_groups = list(dict.fromkeys(p.get("group", "pargp") for p in parameters))
    # Add pilotpoint groups for each property using pilot points
    for p in parameters:
        if p.get("approach") == "pilotpoints":
            pp_group = f"pp_{p['property']}"
            if pp_group not in par_groups:
                par_groups.append(pp_group)

    obs_groups = list(dict.fromkeys(o.get("group", "head") for o in observations))

    noptmax = pest_settings.get("noptmax", 20)
    phiredstp = pest_settings.get("phiredstp", 0.005)
    nphinored = pest_settings.get("nphinored", 4)

    lines = []
    lines.append("pcf")
    lines.append("* control data")
    lines.append("restart estimation")
    lines.append(f"{npar} {nobs} {len(par_groups)} 0 {len(obs_groups)}")
    lines.append("1 1 single point")
    lines.append("5.0 2.0 0.3 0.03 10")
    lines.append("5.0")
    lines.append(f"{noptmax} {phiredstp} {nphinored} 0.01 3")
    lines.append("0 0 0")

    # SVD section
    lines.append("* singular value decomposition")
    lines.append("1")
    maxsing = pest_settings.get("maxsing", min(npar, nobs, 50))
    eigthresh = pest_settings.get("eigthresh", 1e-6)
    lines.append(f"{maxsing} {eigthresh}")
    lines.append("0")

    # Parameter groups
    lines.append("* parameter groups")
    for grp in par_groups:
        lines.append(f"{grp} relative 0.01 0.0 switch 2.0 parabolic")

    # Parameter data
    # Note: param_names may be longer than parameters when pilot points are used
    lines.append("* parameter data")
    for pname in param_names:
        pinfo = param_info.get(pname, {})
        pkg_type = pinfo.get("package_type", "array")

        if pkg_type == "pilotpoint":
            # Pilot point parameter - get settings from base parameter
            base_param = pinfo.get("base_param", "")
            base_settings = param_lookup.get(base_param, {})
            trans = base_settings.get("transform", "log")
            val = 1.0  # Pilot points start at 1.0
            lb = base_settings.get("lower_bound", 0.01)
            ub = base_settings.get("upper_bound", 100.0)
            grp = f"pp_{pinfo.get('property', 'pargp')}"
        elif pkg_type == "list":
            # List-based parameter
            settings = param_lookup.get(pname, {})
            trans = settings.get("transform", "log")
            val = settings.get("initial_value", 1.0)
            lb = settings.get("lower_bound", 0.01)
            ub = settings.get("upper_bound", 100.0)
            grp = settings.get("group", "pargp")
        else:
            # Standard array multiplier
            settings = param_lookup.get(pname, {})
            trans = settings.get("transform", "log")
            val = settings.get("initial_value", 1.0)
            lb = settings.get("lower_bound", 0.01)
            ub = settings.get("upper_bound", 100.0)
            grp = settings.get("group", "pargp")

        lines.append(
            f"{pname} {trans} relative {val} {lb} {ub} {grp} 1.0 0.0 1"
        )

    # Observation groups
    lines.append("* observation groups")
    for grp in obs_groups:
        lines.append(grp)

    # Observation data
    lines.append("* observation data")
    for obs, oname in zip(observations, obs_names):
        val = obs["value"]
        weight = obs.get("weight", 1.0)
        grp = obs.get("group", "head")
        lines.append(f"{oname} {val} {weight} {grp}")

    # Model command line
    lines.append("* model command line")
    lines.append("python forward_run.py")

    # Model input/output
    lines.append("* model input/output")
    lines.append("mult_pars.dat.tpl mult_pars.dat")
    lines.append("sim_obs.dat.ins sim_obs.dat")

    # PEST++ options
    method = pest_settings.get("method", "glm")
    if method == "ies":
        ies_num_reals = pest_settings.get("ies_num_reals", 50)
        ies_iter = pest_settings.get("ies_iter", 3)
        ies_lambda = pest_settings.get("ies_initial_lambda", 100.0)
        ies_bad_phi = pest_settings.get("ies_bad_phi_sigma", 2.0)
        ies_subset = pest_settings.get(
            "ies_subset_size", max(4, ies_num_reals // 10)
        )
        lines.append(f"++ ies_num_reals = {ies_num_reals}")
        lines.append(f"++ ies_lambda_mults = 0.1,1.0,10.0")
        lines.append(f"++ ies_initial_lambda = {ies_lambda}")
        lines.append(f"++ ies_subset_size = {ies_subset}")
        lines.append(f"++ ies_bad_phi_sigma = {ies_bad_phi}")
        lines.append(f"++ ies_verbose_level = 1")
    else:
        lines.append("++ n_iter_base = 1")
        lines.append("++ n_iter_super = 4")
        lines.append(f"++ super_eigthresh = {eigthresh}")

    path.write_text("\n".join(lines) + "\n")


def parse_pest_results(workspace_dir: Path) -> dict:
    """
    Parse PEST++ output files for calibration results.

    Reads .rec file for phi history, .par for final parameters,
    and .rei for residuals (observed vs simulated).

    Args:
        workspace_dir: PEST workspace directory.

    Returns:
        Dict with phi_history, parameters, residuals, converged.
    """
    results: dict = {
        "phi_history": [],
        "parameters": {},
        "residuals": [],
        "converged": False,
    }

    # Parse .rec file for phi history
    rec_files = list(workspace_dir.glob("*.rec"))
    if rec_files:
        _parse_rec_file(rec_files[0], results)

    # Parse final .par file for parameter values
    par_files = sorted(workspace_dir.glob("pest_run.par*"))
    if par_files:
        _parse_par_file(par_files[-1], results)

    # Parse .rei file for residuals
    rei_files = list(workspace_dir.glob("*.rei"))
    if rei_files:
        _parse_rei_file(rei_files[0], results)

    return results


def _parse_rec_file(path: Path, results: dict) -> None:
    """Parse PEST++ .rec file for phi (objective function) history."""
    content = path.read_text(errors="replace")

    # Match lines like: "  1   5.432e+02   ..."
    # pestpp-glm prints iteration summary tables
    phi_pattern = re.compile(
        r"^\s*(\d+)\s+([\d.eE+\-]+)\s+", re.MULTILINE
    )

    # Also look for explicit phi reporting
    phi_alt = re.compile(
        r"(?:iteration|Iteration)\s+(\d+).*?"
        r"(?:phi|Phi|PHI)\s*[:=]\s*([\d.eE+\-]+)",
        re.DOTALL,
    )

    found = set()
    for match in phi_alt.finditer(content):
        try:
            iteration = int(match.group(1))
            phi = float(match.group(2))
            if iteration not in found:
                results["phi_history"].append(
                    {"iteration": iteration, "phi": phi}
                )
                found.add(iteration)
        except ValueError:
            continue

    # If no matches, try the table format
    if not results["phi_history"]:
        # Look for the optimization iteration summary section
        in_summary = False
        for line in content.split("\n"):
            stripped = line.strip()
            if "iteration" in stripped.lower() and "phi" in stripped.lower():
                in_summary = True
                continue
            if in_summary and stripped:
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        iteration = int(parts[0])
                        phi = float(parts[1])
                        results["phi_history"].append(
                            {"iteration": iteration, "phi": phi}
                        )
                    except ValueError:
                        if results["phi_history"]:
                            break

    if "OPTIMIZATION COMPLETE" in content.upper():
        results["converged"] = True
    if "converged" in content.lower():
        results["converged"] = True


def _parse_par_file(path: Path, results: dict) -> None:
    """Parse PEST++ .par file for final parameter values."""
    try:
        content = path.read_text()
        for line in content.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    results["parameters"][parts[0]] = float(parts[1])
                except ValueError:
                    continue
    except Exception:
        pass


def _parse_rei_file(path: Path, results: dict) -> None:
    """Parse PEST++ .rei file for residuals (observed vs simulated)."""
    try:
        content = path.read_text()
        lines = content.strip().split("\n")
        data_started = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    obs_name = parts[0]
                    obs_val = float(parts[1])
                    sim_val = float(parts[2])
                    residual = obs_val - sim_val
                    if len(parts) >= 5:
                        try:
                            residual = float(parts[3])
                        except ValueError:
                            pass
                    results["residuals"].append(
                        {
                            "name": obs_name,
                            "observed": obs_val,
                            "simulated": sim_val,
                            "residual": residual,
                        }
                    )
                    data_started = True
                except ValueError:
                    if data_started:
                        break
                    continue
    except Exception:
        pass


def parse_ies_results(workspace_dir: Path) -> dict:
    """
    Parse PEST++ IES output files for ensemble calibration results.

    Reads phi.actual.csv for per-realization phi history,
    iteration par.csv files for parameter ensembles,
    and iteration obs.csv files for observation ensembles.

    Args:
        workspace_dir: PEST workspace directory.

    Returns:
        Dict with ensemble-specific results.
    """
    import csv

    results: dict = {
        "phi_history": [],
        "parameters": {},
        "residuals": [],
        "converged": False,
        "ensemble": {
            "phi_per_real": {},
            "par_summary": {},
            "obs_summary": {},
            "n_reals": 0,
            "n_failed": 0,
        },
    }

    # Parse .rec file for basic info
    rec_files = list(workspace_dir.glob("*.rec"))
    if rec_files:
        _parse_rec_file(rec_files[0], results)

    # Parse phi.actual.csv — rows are iterations, columns are realizations
    phi_csv_files = list(workspace_dir.glob("*.phi.actual.csv"))
    if phi_csv_files:
        _parse_ies_phi_csv(phi_csv_files[0], results)

    # Parse final parameter ensemble csv
    par_csv_files = sorted(workspace_dir.glob("pest_run.*.par.csv"))
    if par_csv_files:
        _parse_ies_par_csv(par_csv_files[-1], results)
        # Also get prior (iteration 0)
        prior_files = [f for f in par_csv_files if ".0.par.csv" in f.name]
        if prior_files:
            _parse_ies_par_csv(prior_files[0], results, key="prior_summary")

    # Parse final observation ensemble csv for obs vs sim
    obs_csv_files = sorted(workspace_dir.glob("pest_run.*.obs.csv"))
    if obs_csv_files:
        _parse_ies_obs_csv(obs_csv_files[-1], results)

    # Parse .rei for mean residuals (same as GLM)
    rei_files = list(workspace_dir.glob("*.rei"))
    if rei_files:
        _parse_rei_file(rei_files[0], results)

    return results


def _parse_ies_phi_csv(path: Path, results: dict) -> None:
    """Parse IES phi.actual.csv — each row is an iteration, columns are realizations."""
    try:
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            # header: iteration_number, real_0, real_1, ...
            real_names = [h.strip() for h in header[1:]]
            results["ensemble"]["n_reals"] = len(real_names)

            phi_per_real: dict[str, list[dict]] = {
                rn: [] for rn in real_names
            }
            all_iters: list[dict] = []

            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    iteration = int(row[0].strip())
                    phi_vals = []
                    for i, rn in enumerate(real_names):
                        val = float(row[i + 1].strip())
                        phi_per_real[rn].append(
                            {"iteration": iteration, "phi": val}
                        )
                        phi_vals.append(val)

                    valid = [v for v in phi_vals if v < 1e10]
                    if valid:
                        all_iters.append(
                            {
                                "iteration": iteration,
                                "phi": float(np.mean(valid)),
                                "phi_min": float(np.min(valid)),
                                "phi_max": float(np.max(valid)),
                                "phi_std": float(np.std(valid)),
                                "n_failed": len(phi_vals) - len(valid),
                            }
                        )
                except (ValueError, IndexError):
                    continue

            results["phi_history"] = all_iters
            results["ensemble"]["phi_per_real"] = phi_per_real
            if all_iters:
                results["ensemble"]["n_failed"] = all_iters[-1].get(
                    "n_failed", 0
                )
    except Exception:
        pass


def _parse_ies_par_csv(
    path: Path, results: dict, key: str = "par_summary"
) -> None:
    """Parse IES parameter ensemble CSV — rows are realizations, columns are parameters."""
    try:
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            param_names = [h.strip() for h in header[1:]]

            param_vals: dict[str, list[float]] = {
                pn: [] for pn in param_names
            }
            for row in reader:
                if len(row) < 2:
                    continue
                for i, pn in enumerate(param_names):
                    try:
                        param_vals[pn].append(float(row[i + 1].strip()))
                    except (ValueError, IndexError):
                        pass

            summary = {}
            for pn, vals in param_vals.items():
                if vals:
                    arr = np.array(vals)
                    summary[pn] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "p5": float(np.percentile(arr, 5)),
                        "p25": float(np.percentile(arr, 25)),
                        "p50": float(np.percentile(arr, 50)),
                        "p75": float(np.percentile(arr, 75)),
                        "p95": float(np.percentile(arr, 95)),
                        "values": [float(v) for v in arr.tolist()],
                    }

            results["ensemble"][key] = summary
            # Also set the mean as the "best" parameter values
            if key == "par_summary":
                results["parameters"] = {
                    pn: s["mean"] for pn, s in summary.items()
                }
    except Exception:
        pass


def _parse_ies_obs_csv(path: Path, results: dict) -> None:
    """Parse IES observation ensemble CSV — rows are realizations, columns are observations."""
    try:
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            obs_names = [h.strip() for h in header[1:]]

            obs_vals: dict[str, list[float]] = {
                on: [] for on in obs_names
            }
            for row in reader:
                if len(row) < 2:
                    continue
                for i, on in enumerate(obs_names):
                    try:
                        obs_vals[on].append(float(row[i + 1].strip()))
                    except (ValueError, IndexError):
                        pass

            summary = {}
            for on, vals in obs_vals.items():
                if vals:
                    arr = np.array(vals)
                    summary[on] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "p5": float(np.percentile(arr, 5)),
                        "p95": float(np.percentile(arr, 95)),
                    }

            results["ensemble"]["obs_summary"] = summary
    except Exception:
        pass


def build_observations_for_pest(obs_data: dict) -> list[dict]:
    """
    Convert observation data from the observations API format
    to the PEST observation list format.

    Args:
        obs_data: Dict with 'wells' key from observations API.

    Returns:
        List of observation dicts with name, value, time, layer, row/col/node.
    """
    observations = []
    wells = obs_data.get("wells", {})

    for well_name, well_data in wells.items():
        # Sanitize well name for PEST
        sanitized = re.sub(r"[^a-zA-Z0-9]", "_", well_name).lower()
        if len(sanitized) > 12:
            sanitized = sanitized[:12]

        layer = well_data.get("layer", 0)
        row = well_data.get("row")
        col = well_data.get("col")
        node = well_data.get("node")
        times = well_data.get("times", [])
        heads = well_data.get("heads", [])

        for i, (t, h) in enumerate(zip(times, heads)):
            obs_name = f"{sanitized}_t{i}"
            obs = {
                "name": obs_name,
                "well_name": well_name,
                "value": h,
                "time": t,
                "layer": layer,
                "weight": 1.0,
                "group": "head",
            }
            if row is not None:
                obs["row"] = row
            if col is not None:
                obs["col"] = col
            if node is not None:
                obs["node"] = node
            observations.append(obs)

    return observations


def build_observations_from_multiple_sets(
    observation_sets: list[dict],
    set_weights: Optional[dict[str, float]] = None,
    per_well_weights: Optional[dict[str, dict[str, float]]] = None,
) -> list[dict]:
    """
    Merge multiple observation sets into PEST observation list.

    Following pyEMU's add_observations pattern:
    - Each set can have a weight multiplier
    - Per-well weights can override set weights
    - Observation names are prefixed with set identifier to avoid collisions

    Args:
        observation_sets: List of observation set dicts, each containing:
            - id: Set identifier
            - name: Display name
            - data: Dict with 'wells' key containing observation data
        set_weights: Optional dict mapping set_id to weight multiplier
        per_well_weights: Optional dict mapping set_id to per-well weight overrides

    Returns:
        List of observation dicts for PEST, with unique names and adjusted weights.
    """
    observations = []
    set_weights = set_weights or {}
    per_well_weights = per_well_weights or {}

    for obs_set in observation_sets:
        set_id = obs_set.get("id", "")
        set_name = obs_set.get("name", "")
        wells = obs_set.get("data", {}).get("wells", {})

        # Get weight multiplier for this set
        weight_mult = set_weights.get(set_id, 1.0)

        # Create short set prefix (max 4 chars to leave room for well+time)
        set_prefix = re.sub(r"[^a-zA-Z0-9]", "", set_name or set_id)[:4].lower()
        if not set_prefix:
            set_prefix = set_id[:4].lower()

        # Per-well weight overrides for this set
        well_weights = per_well_weights.get(set_id, {})

        for well_name, well_data in wells.items():
            # Sanitize well name for PEST (max 6 chars after prefix)
            sanitized = re.sub(r"[^a-zA-Z0-9]", "_", well_name).lower()
            if len(sanitized) > 6:
                sanitized = sanitized[:6]

            layer = well_data.get("layer", 0)
            row = well_data.get("row")
            col = well_data.get("col")
            node = well_data.get("node")
            times = well_data.get("times", [])
            heads = well_data.get("heads", [])

            # Get weight for this well (override or set default * multiplier)
            base_weight = well_weights.get(well_name, 1.0) * weight_mult

            # Determine observation group based on set
            obs_group = f"head_{set_prefix}" if set_prefix else "head"

            for i, (t, h) in enumerate(zip(times, heads)):
                # Create unique observation name: prefix_well_tN
                obs_name = f"{set_prefix}{sanitized}_t{i}"
                # PEST observation names max 20 chars
                if len(obs_name) > 20:
                    obs_name = obs_name[:20]

                obs = {
                    "name": obs_name,
                    "well_name": well_name,
                    "set_id": set_id,
                    "set_name": set_name,
                    "value": h,
                    "time": t,
                    "layer": layer,
                    "weight": base_weight,
                    "group": obs_group,
                }
                if row is not None:
                    obs["row"] = row
                if col is not None:
                    obs["col"] = col
                if node is not None:
                    obs["node"] = node
                observations.append(obs)

    return observations


def get_observation_sets_summary(observations: list[dict]) -> dict:
    """
    Generate summary statistics for observation sets.

    Args:
        observations: List of observation dicts from build_observations_from_multiple_sets

    Returns:
        Summary dict with counts per set and total
    """
    summary = {
        "total_observations": len(observations),
        "sets": {},
    }

    for obs in observations:
        set_id = obs.get("set_id", "unknown")
        set_name = obs.get("set_name", set_id)

        if set_id not in summary["sets"]:
            summary["sets"][set_id] = {
                "name": set_name,
                "n_observations": 0,
                "wells": set(),
                "groups": set(),
            }

        summary["sets"][set_id]["n_observations"] += 1
        summary["sets"][set_id]["wells"].add(obs.get("well_name", ""))
        summary["sets"][set_id]["groups"].add(obs.get("group", "head"))

    # Convert sets to lists for JSON serialization
    for set_id in summary["sets"]:
        summary["sets"][set_id]["wells"] = list(summary["sets"][set_id]["wells"])
        summary["sets"][set_id]["groups"] = list(summary["sets"][set_id]["groups"])
        summary["sets"][set_id]["n_wells"] = len(summary["sets"][set_id]["wells"])

    return summary
