"""Boundary condition extraction service for MODFLOW models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BoundaryCell:
    """Single boundary condition cell."""

    layer: int
    row: int
    col: int
    values: Dict[str, float]  # e.g., {"head": 45.0} or {"stage": 10.0, "cond": 5.0}


@dataclass
class BoundaryPackage:
    """Boundary condition package data."""

    package_type: str  # CHD, WEL, RIV, DRN, GHB, RCH, EVT
    name: str
    description: str
    cells: List[BoundaryCell]
    stress_period: int
    value_names: List[str]  # Names of values in each cell


def get_boundary_conditions(
    model, stress_period: int = 0
) -> Dict[str, BoundaryPackage]:
    """
    Extract all boundary conditions from a model for a given stress period.

    Returns dict mapping package type to BoundaryPackage data.
    """
    # Check if MF6 by checking model class
    is_mf6 = model.__class__.__module__.startswith('flopy.mf6')

    if is_mf6:
        return _get_mf6_boundaries(model, stress_period)
    else:
        return _get_mf2005_boundaries(model, stress_period)


def _get_mf6_boundaries(model, stress_period: int) -> Dict[str, BoundaryPackage]:
    """Extract boundary conditions from MF6 model."""
    boundaries = {}

    # CHD - Constant Head
    chd = model.get_package("CHD")
    if chd is not None:
        cells = _extract_mf6_list_package(chd, stress_period, ["head"])
        if cells:
            boundaries["CHD"] = BoundaryPackage(
                package_type="CHD",
                name="chd",
                description="Constant Head",
                cells=cells,
                stress_period=stress_period,
                value_names=["head"],
            )

    # WEL - Well
    wel = model.get_package("WEL")
    if wel is not None:
        cells = _extract_mf6_list_package(wel, stress_period, ["q"])
        if cells:
            boundaries["WEL"] = BoundaryPackage(
                package_type="WEL",
                name="wel",
                description="Well",
                cells=cells,
                stress_period=stress_period,
                value_names=["q"],
            )

    # RIV - River
    riv = model.get_package("RIV")
    if riv is not None:
        cells = _extract_mf6_list_package(riv, stress_period, ["stage", "cond", "rbot"])
        if cells:
            boundaries["RIV"] = BoundaryPackage(
                package_type="RIV",
                name="riv",
                description="River",
                cells=cells,
                stress_period=stress_period,
                value_names=["stage", "cond", "rbot"],
            )

    # DRN - Drain
    drn = model.get_package("DRN")
    if drn is not None:
        cells = _extract_mf6_list_package(drn, stress_period, ["elev", "cond"])
        if cells:
            boundaries["DRN"] = BoundaryPackage(
                package_type="DRN",
                name="drn",
                description="Drain",
                cells=cells,
                stress_period=stress_period,
                value_names=["elev", "cond"],
            )

    # GHB - General Head Boundary
    ghb = model.get_package("GHB")
    if ghb is not None:
        cells = _extract_mf6_list_package(ghb, stress_period, ["bhead", "cond"])
        if cells:
            boundaries["GHB"] = BoundaryPackage(
                package_type="GHB",
                name="ghb",
                description="General Head Boundary",
                cells=cells,
                stress_period=stress_period,
                value_names=["bhead", "cond"],
            )

    # RCH - Recharge (array-based)
    rcha = model.get_package("RCHA")
    if rcha is not None:
        cells = _extract_mf6_array_package(model, rcha, stress_period, "recharge")
        if cells:
            boundaries["RCH"] = BoundaryPackage(
                package_type="RCH",
                name="rcha",
                description="Recharge",
                cells=cells,
                stress_period=stress_period,
                value_names=["recharge"],
            )

    # EVT - Evapotranspiration (array-based)
    evta = model.get_package("EVTA")
    if evta is not None:
        cells = _extract_mf6_array_package(model, evta, stress_period, "rate")
        if cells:
            boundaries["EVT"] = BoundaryPackage(
                package_type="EVT",
                name="evta",
                description="Evapotranspiration",
                cells=cells,
                stress_period=stress_period,
                value_names=["rate"],
            )

    return boundaries


def _extract_mf6_list_package(
    package, stress_period: int, value_names: List[str]
) -> List[BoundaryCell]:
    """Extract cells from MF6 list-based package (CHD, WEL, RIV, DRN, GHB)."""
    cells = []

    try:
        # Get stress period data
        spd = package.stress_period_data
        if spd is None:
            return cells

        # Get data for the requested stress period
        data = spd.get_data(key=stress_period)
        if data is None:
            # Try stress period 0 as fallback
            data = spd.get_data(key=0)
        if data is None:
            return cells

        # Extract cell data
        for rec in data:
            # First element is cellid (k, i, j) for structured grids
            cellid = rec[0] if isinstance(rec[0], tuple) else (rec[0], rec[1], rec[2])

            # Remaining elements are values
            values = {}
            for idx, name in enumerate(value_names):
                if isinstance(rec[0], tuple):
                    val_idx = idx + 1
                else:
                    val_idx = idx + 3  # Skip k, i, j

                if val_idx < len(rec):
                    values[name] = float(rec[val_idx])

            cells.append(
                BoundaryCell(
                    layer=int(cellid[0]),
                    row=int(cellid[1]),
                    col=int(cellid[2]),
                    values=values,
                )
            )

    except Exception as e:
        print(f"Error extracting list package: {e}")

    return cells


def _extract_mf6_array_package(
    model, package, stress_period: int, value_name: str
) -> List[BoundaryCell]:
    """Extract cells from MF6 array-based package (RCH, EVT)."""
    cells = []

    try:
        # Get DIS for grid dimensions (get_package("DIS") returns DISV/DISU too)
        dis = model.get_package("DIS")
        pkg_type = getattr(dis, 'package_type', '').upper() if dis else ''
        if pkg_type == "DIS":
            nrow = dis.nrow.data
            ncol = dis.ncol.data
        elif pkg_type == "DISV":
            nrow = 1
            ncol = dis.ncpl.data
        elif pkg_type == "DISU":
            nrow = 1
            ncol = dis.nodes.data
        else:
            return cells

        # Get the array data
        arr_obj = getattr(package, value_name, None)
        if arr_obj is None:
            return cells

        # Get data - might be time-varying or a dict
        data = arr_obj.get_data()
        if data is None:
            return cells

        # Handle dict (stress period indexed data)
        if isinstance(data, dict):
            data = data.get(stress_period, data.get(0))
            if data is None:
                return cells

        # Convert to array - handle scalar constant values
        if isinstance(data, (int, float)):
            # Constant value for entire grid
            arr = np.full((nrow, ncol), data, dtype=np.float64)
        else:
            arr = np.array(data, dtype=np.float64)

        # Handle 2D array (single layer recharge)
        if arr.ndim == 2 and arr.shape == (nrow, ncol):
            for i in range(nrow):
                for j in range(ncol):
                    val = arr[i, j]
                    if val != 0:  # Only include non-zero cells
                        cells.append(
                            BoundaryCell(
                                layer=0,
                                row=i,
                                col=j,
                                values={value_name: float(val)},
                            )
                        )

    except Exception as e:
        print(f"Error extracting array package: {e}")

    return cells


def _get_mf2005_boundaries(model, stress_period: int) -> Dict[str, BoundaryPackage]:
    """Extract boundary conditions from MF2005/NWT model."""
    boundaries = {}

    # CHD
    if model.chd is not None:
        cells = _extract_mf2005_list_package(
            model.chd, stress_period, ["shead", "ehead"]
        )
        if cells:
            boundaries["CHD"] = BoundaryPackage(
                package_type="CHD",
                name="chd",
                description="Constant Head",
                cells=cells,
                stress_period=stress_period,
                value_names=["shead", "ehead"],
            )

    # WEL
    if model.wel is not None:
        cells = _extract_mf2005_list_package(model.wel, stress_period, ["flux"])
        if cells:
            boundaries["WEL"] = BoundaryPackage(
                package_type="WEL",
                name="wel",
                description="Well",
                cells=cells,
                stress_period=stress_period,
                value_names=["flux"],
            )

    # RIV
    if model.riv is not None:
        cells = _extract_mf2005_list_package(
            model.riv, stress_period, ["stage", "cond", "rbot"]
        )
        if cells:
            boundaries["RIV"] = BoundaryPackage(
                package_type="RIV",
                name="riv",
                description="River",
                cells=cells,
                stress_period=stress_period,
                value_names=["stage", "cond", "rbot"],
            )

    # DRN
    if model.drn is not None:
        cells = _extract_mf2005_list_package(model.drn, stress_period, ["elev", "cond"])
        if cells:
            boundaries["DRN"] = BoundaryPackage(
                package_type="DRN",
                name="drn",
                description="Drain",
                cells=cells,
                stress_period=stress_period,
                value_names=["elev", "cond"],
            )

    # GHB
    if model.ghb is not None:
        cells = _extract_mf2005_list_package(
            model.ghb, stress_period, ["bhead", "cond"]
        )
        if cells:
            boundaries["GHB"] = BoundaryPackage(
                package_type="GHB",
                name="ghb",
                description="General Head Boundary",
                cells=cells,
                stress_period=stress_period,
                value_names=["bhead", "cond"],
            )

    # RCH
    if model.rch is not None:
        cells = _extract_mf2005_rch(model.rch, stress_period)
        if cells:
            boundaries["RCH"] = BoundaryPackage(
                package_type="RCH",
                name="rch",
                description="Recharge",
                cells=cells,
                stress_period=stress_period,
                value_names=["recharge"],
            )

    return boundaries


def _extract_mf2005_list_package(
    package, stress_period: int, value_names: List[str]
) -> List[BoundaryCell]:
    """Extract cells from MF2005 list-based package."""
    cells = []

    try:
        spd = package.stress_period_data
        if spd is None:
            return cells

        # MfList can be accessed as a dict or through .data property
        if hasattr(spd, 'data'):
            # spd.data is a dict keyed by stress period
            data_dict = spd.data
            if isinstance(data_dict, dict):
                data = data_dict.get(stress_period, data_dict.get(0))
            else:
                # Might be a recarray directly
                data = data_dict
        elif isinstance(spd, dict):
            data = spd.get(stress_period, spd.get(0))
        else:
            # Try to index it directly
            try:
                data = spd[stress_period]
            except (KeyError, TypeError):
                try:
                    data = spd[0]
                except (KeyError, TypeError):
                    return cells

        if data is None:
            return cells

        for rec in data:
            # rec format: (k, i, j, values...)
            values = {}
            for idx, name in enumerate(value_names):
                val_idx = idx + 3  # Skip k, i, j
                if val_idx < len(rec):
                    values[name] = float(rec[val_idx])

            cells.append(
                BoundaryCell(
                    layer=int(rec[0]),
                    row=int(rec[1]),
                    col=int(rec[2]),
                    values=values,
                )
            )

    except Exception as e:
        print(f"Error extracting MF2005 list package: {e}")

    return cells


def _extract_mf2005_rch(rch_package, stress_period: int) -> List[BoundaryCell]:
    """Extract recharge cells from MF2005 RCH package."""
    cells = []

    try:
        rech = rch_package.rech
        if rech is None:
            return cells

        # Get array for stress period
        arr = rech.array
        if arr is None:
            return cells

        # Handle time-varying
        if arr.ndim == 3:
            sp_idx = min(stress_period, arr.shape[0] - 1)
            arr = arr[sp_idx]

        nrow, ncol = arr.shape
        for i in range(nrow):
            for j in range(ncol):
                val = arr[i, j]
                if val != 0:
                    cells.append(
                        BoundaryCell(
                            layer=0,
                            row=i,
                            col=j,
                            values={"recharge": float(val)},
                        )
                    )

    except Exception as e:
        print(f"Error extracting MF2005 RCH: {e}")

    return cells


def boundary_package_to_dict(pkg: BoundaryPackage) -> dict:
    """Convert BoundaryPackage to JSON-serializable dict."""
    return {
        "package_type": pkg.package_type,
        "name": pkg.name,
        "description": pkg.description,
        "stress_period": pkg.stress_period,
        "value_names": pkg.value_names,
        "cell_count": len(pkg.cells),
        "cells": [
            {
                "layer": c.layer,
                "row": c.row,
                "col": c.col,
                "values": c.values,
            }
            for c in pkg.cells
        ],
    }
