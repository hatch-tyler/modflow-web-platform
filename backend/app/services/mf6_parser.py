"""Direct parser for MF6 DIS file to extract grid arrays."""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def parse_mf6_dis_file(dis_path: Path) -> dict:
    """
    Parse MF6 DIS file to extract grid information.

    Returns dict with: nlay, nrow, ncol, delr, delc, top, botm
    """
    result = {
        "nlay": None,
        "nrow": None,
        "ncol": None,
        "delr": None,
        "delc": None,
        "top": None,
        "botm": None,
    }

    content = dis_path.read_text()
    lines = content.strip().split("\n")

    # Parse DIMENSIONS block
    in_dimensions = False
    for line in lines:
        line_stripped = line.strip().upper()

        if "BEGIN DIMENSIONS" in line_stripped:
            in_dimensions = True
            continue
        if "END DIMENSIONS" in line_stripped:
            in_dimensions = False
            continue

        if in_dimensions:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].upper()
                value = int(parts[1])
                if key == "NLAY":
                    result["nlay"] = value
                elif key == "NROW":
                    result["nrow"] = value
                elif key == "NCOL":
                    result["ncol"] = value

    nlay = result["nlay"]
    nrow = result["nrow"]
    ncol = result["ncol"]

    if not all([nlay, nrow, ncol]):
        return result

    # Parse GRIDDATA block
    in_griddata = False
    current_var = None
    layered_data = []
    layer_count = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip().upper()

        if "BEGIN GRIDDATA" in line_stripped:
            in_griddata = True
            i += 1
            continue
        if "END GRIDDATA" in line_stripped:
            in_griddata = False
            i += 1
            continue

        if in_griddata:
            parts = line.split()
            if not parts:
                i += 1
                continue

            keyword = parts[0].upper()

            if keyword == "DELR":
                # DELR CONSTANT value or DELR followed by array
                if len(parts) >= 3 and parts[1].upper() == "CONSTANT":
                    value = float(parts[2])
                    result["delr"] = np.full(ncol, value, dtype=np.float64)
                i += 1
                continue

            elif keyword == "DELC":
                if len(parts) >= 3 and parts[1].upper() == "CONSTANT":
                    value = float(parts[2])
                    result["delc"] = np.full(nrow, value, dtype=np.float64)
                i += 1
                continue

            elif keyword == "TOP":
                if len(parts) >= 3 and parts[1].upper() == "CONSTANT":
                    value = float(parts[2])
                    result["top"] = np.full((nrow, ncol), value, dtype=np.float64)
                i += 1
                continue

            elif keyword == "BOTM":
                # BOTM LAYERED followed by layer data
                if len(parts) >= 2 and parts[1].upper() == "LAYERED":
                    layered_data = []
                    i += 1
                    # Read layer data
                    for k in range(nlay):
                        if i < len(lines):
                            layer_line = lines[i].strip()
                            layer_parts = layer_line.split()
                            if len(layer_parts) >= 2 and layer_parts[0].upper() == "CONSTANT":
                                value = float(layer_parts[1])
                                layered_data.append(np.full((nrow, ncol), value, dtype=np.float64))
                            i += 1
                    if len(layered_data) == nlay:
                        result["botm"] = np.array(layered_data)
                    continue
                elif len(parts) >= 3 and parts[1].upper() == "CONSTANT":
                    # Single constant for all layers
                    value = float(parts[2])
                    result["botm"] = np.full((nlay, nrow, ncol), value, dtype=np.float64)
                i += 1
                continue

            else:
                # Skip CONSTANT lines that are part of LAYERED data
                if keyword == "CONSTANT":
                    i += 1
                    continue

        i += 1

    return result


def parse_mf6_npf_k(npf_path: Path, nlay: int, nrow: int, ncol: int) -> Optional[np.ndarray]:
    """Parse K array from NPF file."""
    content = npf_path.read_text()
    lines = content.strip().split("\n")

    in_griddata = False
    for i, line in enumerate(lines):
        line_stripped = line.strip().upper()

        if "BEGIN GRIDDATA" in line_stripped:
            in_griddata = True
            continue
        if "END GRIDDATA" in line_stripped:
            in_griddata = False
            continue

        if in_griddata and line_stripped.startswith("K ") or line_stripped == "K":
            parts = line.split()
            if len(parts) >= 3 and parts[1].upper() == "CONSTANT":
                value = float(parts[2])
                return np.full((nlay, nrow, ncol), value, dtype=np.float64)

    return None


def parse_mf6_ic_strt(ic_path: Path, nlay: int, nrow: int, ncol: int) -> Optional[np.ndarray]:
    """Parse STRT array from IC file."""
    content = ic_path.read_text()
    lines = content.strip().split("\n")

    in_griddata = False
    for i, line in enumerate(lines):
        line_stripped = line.strip().upper()

        if "BEGIN GRIDDATA" in line_stripped:
            in_griddata = True
            continue
        if "END GRIDDATA" in line_stripped:
            in_griddata = False
            continue

        if in_griddata and line_stripped.startswith("STRT"):
            parts = line.split()
            if len(parts) >= 3 and parts[1].upper() == "CONSTANT":
                value = float(parts[2])
                return np.full((nlay, nrow, ncol), value, dtype=np.float64)

    return None
