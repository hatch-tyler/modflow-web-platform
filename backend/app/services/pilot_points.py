"""Pilot points parameterization for PEST++ calibration.

This module provides functions for creating pilot point grids,
setting up kriging interpolation, and applying kriged values
to the full model grid.

Pilot points are sparsely-distributed parameter locations where values
are estimated during calibration, then interpolated (kriged) to the
full grid to provide spatial heterogeneity within layers.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def create_pilot_point_grid(
    ibound: np.ndarray,
    pp_space: int,
    param_name: str,
    layer: int,
    output_dir: Path,
    xoff: float = 0.0,
    yoff: float = 0.0,
    delr: Optional[np.ndarray] = None,
    delc: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Create a regular grid of pilot points on active cells.

    Places pilot points at regular intervals (pp_space) across the grid,
    only on cells that are active (ibound > 0).

    Args:
        ibound: Active cell indicator array (nrow, ncol) or (nlay, nrow, ncol).
        pp_space: Spacing between pilot points (every Nth row/column).
        param_name: Parameter name for naming pilot points.
        layer: Layer index for this set of pilot points.
        output_dir: Directory to write pilot point file.
        xoff: X-coordinate offset.
        yoff: Y-coordinate offset.
        delr: Column widths (for calculating x-coordinates).
        delc: Row heights (for calculating y-coordinates).

    Returns:
        DataFrame with columns: name, x, y, zone, value (initial=1.0)
    """
    # Handle 2D or 3D ibound
    if ibound.ndim == 3:
        ib_layer = ibound[layer]
    else:
        ib_layer = ibound

    nrow, ncol = ib_layer.shape

    # Calculate cell center coordinates
    if delr is None:
        delr = np.ones(ncol)
    if delc is None:
        delc = np.ones(nrow)

    # X coordinates (column centers)
    x_edges = np.concatenate([[0], np.cumsum(delr)])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2 + xoff

    # Y coordinates (row centers) - note: rows go from top to bottom
    y_edges = np.concatenate([[0], np.cumsum(delc)])
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    y_centers = yoff + np.sum(delc) - y_centers  # Flip for top-down rows

    pp_data = []
    pp_idx = 0

    # Create pilot points at regular intervals on active cells
    for i in range(0, nrow, pp_space):
        for j in range(0, ncol, pp_space):
            if ib_layer[i, j] > 0:
                pp_name = f"pp_{param_name}_l{layer}_{pp_idx:04d}"
                pp_data.append({
                    "name": pp_name,
                    "x": float(x_centers[j]),
                    "y": float(y_centers[i]),
                    "zone": 1,
                    "value": 1.0,  # Initial multiplier
                    "row": i,
                    "col": j,
                })
                pp_idx += 1

    df = pd.DataFrame(pp_data)

    # Write pilot point file
    pp_file = output_dir / f"{param_name}_l{layer}.pp"
    df[["name", "x", "y", "zone", "value"]].to_csv(
        pp_file, sep=" ", index=False, header=False
    )

    return df


def setup_kriging(
    pp_file: Path,
    variogram_config: dict,
    grid_coords: dict,
    output_dir: Path,
) -> Path:
    """
    Set up kriging interpolation and generate factors file.

    Uses ordinary kriging with specified variogram to create interpolation
    factors that map pilot point values to grid cells.

    Args:
        pp_file: Path to pilot point file.
        variogram_config: Dict with variogram parameters:
            - type: 'exponential', 'spherical', or 'gaussian'
            - correlation_length: Range/correlation length
            - anisotropy: Anisotropy ratio (1.0 = isotropic)
            - bearing: Direction of anisotropy (degrees)
        grid_coords: Dict with grid coordinate information:
            - x_centers: 1D array of cell x-coordinates
            - y_centers: 1D array of cell y-coordinates
            - nrow: Number of rows
            - ncol: Number of columns
        output_dir: Directory to write factors file.

    Returns:
        Path to the kriging factors file.
    """
    # Parse pilot points
    pp_df = pd.read_csv(
        pp_file, sep=r"\s+", header=None,
        names=["name", "x", "y", "zone", "value"]
    )

    # Extract variogram parameters
    vario_type = variogram_config.get("type", "exponential")
    corr_len = variogram_config.get("correlation_length", 1000.0)
    anisotropy = variogram_config.get("anisotropy", 1.0)
    bearing = variogram_config.get("bearing", 0.0)

    # Create grid cell coordinates
    nrow = grid_coords["nrow"]
    ncol = grid_coords["ncol"]
    x_centers = np.array(grid_coords["x_centers"])
    y_centers = np.array(grid_coords["y_centers"])

    # Build full grid of target points
    target_x = []
    target_y = []
    for i in range(nrow):
        for j in range(ncol):
            target_x.append(x_centers[j])
            target_y.append(y_centers[i])

    target_x = np.array(target_x)
    target_y = np.array(target_y)

    # Pilot point coordinates
    pp_x = pp_df["x"].values
    pp_y = pp_df["y"].values
    pp_names = pp_df["name"].tolist()
    n_pp = len(pp_names)
    n_targets = len(target_x)

    # Compute kriging weights using ordinary kriging
    # For each target point, compute weights for all pilot points

    # Build distance matrix between pilot points
    pp_dist = np.zeros((n_pp, n_pp))
    for i in range(n_pp):
        for j in range(n_pp):
            dx = pp_x[i] - pp_x[j]
            dy = pp_y[i] - pp_y[j]
            # Apply anisotropy rotation
            if anisotropy != 1.0 and bearing != 0.0:
                rad = np.radians(bearing)
                dx_rot = dx * np.cos(rad) + dy * np.sin(rad)
                dy_rot = -dx * np.sin(rad) + dy * np.cos(rad)
                dx = dx_rot
                dy = dy_rot / anisotropy
            pp_dist[i, j] = np.sqrt(dx**2 + dy**2)

    # Compute variogram values (semivariance)
    def variogram_value(h, vtype, a):
        """Calculate semivariance for distance h."""
        if h == 0:
            return 0.0
        if vtype == "exponential":
            return 1.0 - np.exp(-3.0 * h / a)
        elif vtype == "spherical":
            if h < a:
                return 1.5 * (h / a) - 0.5 * (h / a) ** 3
            return 1.0
        elif vtype == "gaussian":
            return 1.0 - np.exp(-3.0 * (h / a) ** 2)
        return 1.0 - np.exp(-3.0 * h / a)

    # Build kriging matrix (ordinary kriging: add Lagrange multiplier row/col)
    K = np.zeros((n_pp + 1, n_pp + 1))
    for i in range(n_pp):
        for j in range(n_pp):
            K[i, j] = variogram_value(pp_dist[i, j], vario_type, corr_len)
    K[n_pp, :n_pp] = 1.0
    K[:n_pp, n_pp] = 1.0
    K[n_pp, n_pp] = 0.0

    # Invert kriging matrix
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        # Add small regularization if singular
        K_inv = np.linalg.inv(K + 1e-10 * np.eye(n_pp + 1))

    # Compute weights for each target point
    factors = []
    for t in range(n_targets):
        tx, ty = target_x[t], target_y[t]

        # Distance from target to each pilot point
        k = np.zeros(n_pp + 1)
        for p in range(n_pp):
            dx = tx - pp_x[p]
            dy = ty - pp_y[p]
            if anisotropy != 1.0 and bearing != 0.0:
                rad = np.radians(bearing)
                dx_rot = dx * np.cos(rad) + dy * np.sin(rad)
                dy_rot = -dx * np.sin(rad) + dy * np.cos(rad)
                dx = dx_rot
                dy = dy_rot / anisotropy
            dist = np.sqrt(dx**2 + dy**2)
            k[p] = variogram_value(dist, vario_type, corr_len)
        k[n_pp] = 1.0  # Lagrange constraint

        # Compute weights
        weights = K_inv @ k

        # Store non-zero weights for this target
        row = t // ncol
        col = t % ncol
        for p in range(n_pp):
            if abs(weights[p]) > 1e-10:
                factors.append({
                    "target_row": row,
                    "target_col": col,
                    "pp_name": pp_names[p],
                    "weight": float(weights[p]),
                })

    # Write factors file
    factors_df = pd.DataFrame(factors)
    factors_file = output_dir / f"{pp_file.stem}.fac"
    factors_df.to_csv(factors_file, index=False)

    return factors_file


def apply_kriged_values(
    pp_values: dict,
    factors_file: Path,
    nrow: int,
    ncol: int,
) -> np.ndarray:
    """
    Interpolate pilot point values to full grid using kriging factors.

    Args:
        pp_values: Dict mapping pilot point names to their values.
        factors_file: Path to kriging factors file.
        nrow: Number of rows in output grid.
        ncol: Number of columns in output grid.

    Returns:
        2D array (nrow, ncol) of interpolated values.
    """
    # Read factors
    factors_df = pd.read_csv(factors_file)

    # Initialize output grid
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


def estimate_pilot_point_count(
    nrow: int,
    ncol: int,
    pp_space: int,
    active_fraction: float = 0.8,
) -> int:
    """
    Estimate the number of pilot points that would be created.

    Args:
        nrow: Number of grid rows.
        ncol: Number of grid columns.
        pp_space: Spacing between pilot points.
        active_fraction: Estimated fraction of active cells.

    Returns:
        Estimated pilot point count.
    """
    pp_rows = (nrow + pp_space - 1) // pp_space
    pp_cols = (ncol + pp_space - 1) // pp_space
    total = pp_rows * pp_cols
    return int(total * active_fraction)


def get_default_variogram(
    property_type: str,
    grid_extent: float,
) -> dict:
    """
    Get default variogram configuration for a property type.

    Args:
        property_type: Type of property ('hk', 'vka', 'ss', 'sy').
        grid_extent: Approximate extent of the model grid.

    Returns:
        Dict with variogram configuration.
    """
    # Default correlation length as fraction of grid extent
    if property_type in ("hk", "k"):
        corr_frac = 0.3  # HK often has shorter correlation
    elif property_type in ("vka", "k33"):
        corr_frac = 0.4
    elif property_type == "ss":
        corr_frac = 0.5  # Storage properties often more uniform
    else:
        corr_frac = 0.4

    return {
        "type": "exponential",
        "correlation_length": grid_extent * corr_frac,
        "anisotropy": 1.0,
        "bearing": 0.0,
    }


def write_pilot_points_info(
    output_dir: Path,
    pp_info: dict,
) -> Path:
    """
    Write pilot points configuration info for forward_run.py.

    Args:
        output_dir: Directory to write info file.
        pp_info: Dict with pilot point configuration:
            - param_name: Base parameter name
            - layer: Layer index
            - pp_names: List of pilot point names
            - factors_file: Path to factors file
            - nrow: Grid rows
            - ncol: Grid columns

    Returns:
        Path to the info file.
    """
    info_file = output_dir / "pilot_points_info.json"

    # Convert Path objects to strings for JSON
    serializable = {}
    for key, val in pp_info.items():
        if isinstance(val, Path):
            serializable[key] = str(val)
        elif isinstance(val, list):
            serializable[key] = [str(v) if isinstance(v, Path) else v for v in val]
        else:
            serializable[key] = val

    with open(info_file, "w") as f:
        json.dump(serializable, f, indent=2)

    return info_file
