"""
Detailed convergence data parser for MODFLOW listing files.

Extracts per-timestep iteration counts, head changes, residuals,
backtracking events, and solver settings from MF6 and classic listing files.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


def parse_mf6_listing(
    mfsim_lst_path: Optional[Path],
    flow_lst_path: Optional[Path],
) -> dict:
    """
    Parse MF6 listing files for detailed convergence data.

    Args:
        mfsim_lst_path: Path to mfsim.lst (simulation-level listing)
        flow_lst_path: Path to model .lst (flow model listing)

    Returns:
        Dict with timesteps, stress_period_summary, problem_cells, solver_settings
    """
    timesteps = []
    problem_cell_counts: dict[str, dict] = defaultdict(
        lambda: {"occurrences": 0, "type": "head_change", "affected_sps": set()}
    )
    solver_settings: dict = {}
    failed_timesteps_count = 0

    # --- Parse mfsim.lst for iteration counts and convergence status ---
    if mfsim_lst_path and mfsim_lst_path.exists():
        _parse_mfsim_lst(mfsim_lst_path, timesteps, problem_cell_counts, solver_settings)

    # Count failures
    failed_timesteps_count = sum(1 for ts in timesteps if not ts.get("converged", True))

    # --- Parse flow.lst for percent discrepancy ---
    sp_discrepancies: dict[int, float] = {}
    if flow_lst_path and flow_lst_path.exists():
        sp_discrepancies = _parse_flow_lst_discrepancies(flow_lst_path)

    # --- Build stress period summary ---
    sp_data: dict[int, list] = defaultdict(list)
    for ts in timesteps:
        sp_data[ts["kper"]].append(ts)

    stress_period_summary = []
    for kper in sorted(sp_data.keys()):
        ts_list = sp_data[kper]
        outer_iters = [t["outer_iterations"] for t in ts_list]
        failed = sum(1 for t in ts_list if not t.get("converged", True))
        total_iters = sum(outer_iters)
        max_iters = max(outer_iters) if outer_iters else 0
        avg_iters = total_iters / len(outer_iters) if outer_iters else 0

        # Classify difficulty
        if failed > 0:
            difficulty = "failed"
        elif max_iters > 50:
            difficulty = "high"
        elif max_iters > 20:
            difficulty = "moderate"
        else:
            difficulty = "low"

        stress_period_summary.append({
            "kper": kper,
            "total_outer_iters": total_iters,
            "max_iterations": max_iters,
            "failed_timesteps": failed,
            "avg_iterations": round(avg_iters, 1),
            "percent_discrepancy": sp_discrepancies.get(kper),
            "difficulty": difficulty,
        })

    # --- Build problem cells list ---
    problem_cells = []
    for cell_id, info in sorted(
        problem_cell_counts.items(),
        key=lambda x: x[1]["occurrences"],
        reverse=True,
    )[:50]:  # Top 50 problem cells
        problem_cells.append({
            "cell_id": cell_id,
            "occurrences": info["occurrences"],
            "type": info["type"],
            "affected_sps": sorted(info["affected_sps"]),
        })

    return {
        "model_type": "mf6",
        "total_timesteps": len(timesteps),
        "failed_timesteps": failed_timesteps_count,
        "timesteps": timesteps,
        "stress_period_summary": stress_period_summary,
        "problem_cells": problem_cells,
        "solver_settings": solver_settings,
    }


def _parse_mfsim_lst(
    path: Path,
    timesteps: list,
    problem_cell_counts: dict,
    solver_settings: dict,
) -> None:
    """Parse mfsim.lst for iteration data and solver settings."""
    # Patterns
    calls_pat = re.compile(
        r"(\d+)\s+CALLS?\s+TO\s+NUMERICAL\s+SOLUTION\s+IN\s+TIME\s+STEP\s+(\d+)\s+"
        r"STRESS\s+PERIOD\s+(\d+)",
        re.IGNORECASE,
    )
    # Inner iteration table line: outer_iter, dvmax, cell, rclose, cell
    inner_pat = re.compile(
        r"^\s*(\d+)\s+"                           # outer iteration number
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"   # max head change (dvmax)
        r"(\S+)\s+"                                # cell ID for dvmax
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"   # max residual (rclose)
        r"(\S+)",                                  # cell ID for rclose
    )
    failed_pat = re.compile(
        r"FAILED\s+TO\s+CONVERGE", re.IGNORECASE
    )
    backtrack_pat = re.compile(
        r"BACKTRACKING", re.IGNORECASE
    )
    # Solver settings patterns
    complexity_pat = re.compile(r"COMPLEXITY\s+OPTION\s*:\s*(\w+)", re.IGNORECASE)
    outer_dvclose_pat = re.compile(r"OUTER_DVCLOSE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)
    outer_max_pat = re.compile(r"OUTER_MAXIMUM\s*=\s*(\d+)", re.IGNORECASE)
    inner_rclose_pat = re.compile(r"INNER_RCLOSE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)
    inner_dvclose_pat = re.compile(r"INNER_DVCLOSE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)
    under_relax_pat = re.compile(r"UNDER.RELAXATION\s+(?:OPTION|SCHEME)\s*:\s*(\w+)", re.IGNORECASE)
    backtrack_num_pat = re.compile(r"BACKTRACKING_NUMBER\s*=\s*(\d+)", re.IGNORECASE)
    linear_accel_pat = re.compile(r"LINEAR_ACCELERATION\s*=\s*(\w+)", re.IGNORECASE)

    current_sp = 0
    current_ts = 0
    current_outer_iters = 0
    current_max_dvmax = 0.0
    current_max_dvmax_cell = ""
    current_max_rclose = 0.0
    current_max_rclose_cell = ""
    current_backtracking = 0
    current_converged = True
    in_iteration_table = False
    solver_settings["solver_type"] = "IMS"

    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                # Solver settings (parsed from early in the file)
                if not solver_settings.get("complexity"):
                    m = complexity_pat.search(line)
                    if m:
                        solver_settings["complexity"] = m.group(1).upper()
                if not solver_settings.get("outer_dvclose"):
                    m = outer_dvclose_pat.search(line)
                    if m:
                        solver_settings["outer_dvclose"] = float(m.group(1))
                if not solver_settings.get("outer_maximum"):
                    m = outer_max_pat.search(line)
                    if m:
                        solver_settings["outer_maximum"] = int(m.group(1))
                if not solver_settings.get("inner_rclose"):
                    m = inner_rclose_pat.search(line)
                    if m:
                        solver_settings["inner_rclose"] = float(m.group(1))
                if not solver_settings.get("inner_dvclose"):
                    m = inner_dvclose_pat.search(line)
                    if m:
                        solver_settings["inner_dvclose"] = float(m.group(1))
                if not solver_settings.get("under_relaxation"):
                    m = under_relax_pat.search(line)
                    if m:
                        solver_settings["under_relaxation"] = m.group(1).upper()
                if not solver_settings.get("backtracking_number"):
                    m = backtrack_num_pat.search(line)
                    if m:
                        solver_settings["backtracking_number"] = int(m.group(1))
                if not solver_settings.get("linear_acceleration"):
                    m = linear_accel_pat.search(line)
                    if m:
                        solver_settings["linear_acceleration"] = m.group(1).upper()

                # Check for N CALLS TO NUMERICAL SOLUTION
                m = calls_pat.search(line)
                if m:
                    # Save previous timestep if any
                    if current_sp > 0 or current_ts > 0:
                        timesteps.append({
                            "kper": current_sp - 1,  # 0-indexed
                            "kstp": current_ts - 1,
                            "outer_iterations": current_outer_iters,
                            "converged": current_converged,
                            "max_dvmax": current_max_dvmax,
                            "max_dvmax_cell": current_max_dvmax_cell,
                            "max_rclose": current_max_rclose,
                            "max_rclose_cell": current_max_rclose_cell,
                            "backtracking_events": current_backtracking,
                        })

                    current_outer_iters = int(m.group(1))
                    current_ts = int(m.group(2))
                    current_sp = int(m.group(3))
                    current_max_dvmax = 0.0
                    current_max_dvmax_cell = ""
                    current_max_rclose = 0.0
                    current_max_rclose_cell = ""
                    current_backtracking = 0
                    current_converged = True
                    in_iteration_table = False
                    continue

                # Inner iteration table lines
                m = inner_pat.match(line)
                if m:
                    in_iteration_table = True
                    dvmax = abs(float(m.group(2)))
                    dvmax_cell = m.group(3)
                    rclose = abs(float(m.group(4)))
                    rclose_cell = m.group(5)

                    if dvmax > current_max_dvmax:
                        current_max_dvmax = dvmax
                        current_max_dvmax_cell = dvmax_cell

                    if rclose > current_max_rclose:
                        current_max_rclose = rclose
                        current_max_rclose_cell = rclose_cell

                    # Track problem cells (cells appearing in max columns)
                    for cell_id in (dvmax_cell, rclose_cell):
                        if cell_id and cell_id not in ("", "0"):
                            entry = problem_cell_counts[cell_id]
                            entry["occurrences"] += 1
                            entry["affected_sps"].add(current_sp - 1)
                    continue

                # Failed to converge
                if failed_pat.search(line):
                    current_converged = False

                # Backtracking events
                if backtrack_pat.search(line):
                    current_backtracking += 1

        # Don't forget the last timestep
        if current_sp > 0 or current_ts > 0:
            timesteps.append({
                "kper": current_sp - 1,
                "kstp": current_ts - 1,
                "outer_iterations": current_outer_iters,
                "converged": current_converged,
                "max_dvmax": current_max_dvmax,
                "max_dvmax_cell": current_max_dvmax_cell,
                "max_rclose": current_max_rclose,
                "max_rclose_cell": current_max_rclose_cell,
                "backtracking_events": current_backtracking,
            })

    except Exception as e:
        print(f"Warning: Error parsing mfsim.lst: {e}")


def _parse_flow_lst_discrepancies(path: Path) -> dict[int, float]:
    """Parse flow model listing for percent discrepancy per stress period."""
    discrepancy_pat = re.compile(
        r"PERCENT\s+DISCREPANCY\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    sp_pat = re.compile(
        r"STRESS\s+PERIOD\s+(\d+)", re.IGNORECASE
    )

    sp_discrepancies: dict[int, float] = {}
    current_sp = 0

    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = sp_pat.search(line)
                if m:
                    current_sp = int(m.group(1)) - 1  # 0-indexed

                m = discrepancy_pat.search(line)
                if m:
                    val = abs(float(m.group(1)))
                    # Keep max discrepancy per SP
                    if current_sp not in sp_discrepancies or val > sp_discrepancies[current_sp]:
                        sp_discrepancies[current_sp] = val
    except Exception as e:
        print(f"Warning: Error parsing flow listing: {e}")

    return sp_discrepancies


def parse_classic_listing(lst_path: Path) -> dict:
    """
    Parse MF2005/NWT/USG listing file for convergence data.

    Args:
        lst_path: Path to the .lst file

    Returns:
        Dict with timesteps, stress_period_summary, problem_cells, solver_settings
    """
    timesteps = []
    solver_settings: dict = {}
    problem_cell_counts: dict[str, dict] = defaultdict(
        lambda: {"occurrences": 0, "type": "head_change", "affected_sps": set()}
    )

    # Patterns
    sp_ts_pat = re.compile(
        r"STRESS\s+PERIOD\s+NO\.\s*=?\s*(\d+).*?TIME\s+STEP\s+NO\.\s*=?\s*(\d+)",
        re.IGNORECASE,
    )
    head_change_pat = re.compile(
        r"MAXIMUM\s+HEAD\s+CHANGE\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"(?:\s+AT.*?(?:NODE|CELL)\s*\(?(\d+\s*,\s*\d+(?:\s*,\s*\d+)?)\)?)?",
        re.IGNORECASE,
    )
    iterations_pat = re.compile(
        r"(\d+)\s+(?:ITERATION|CALL)S?\s+(?:FOR|TO|IN)\s+(?:THIS\s+)?TIME\s+STEP",
        re.IGNORECASE,
    )
    discrepancy_pat = re.compile(
        r"PERCENT\s+DISCREPANCY\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    failed_pat = re.compile(
        r"FAILED\s+TO\s+(?:MEET\s+SOLVER|CONVERGE)", re.IGNORECASE
    )
    # Solver type detection
    pcg_pat = re.compile(r"PCG\d?\s+--\s+", re.IGNORECASE)
    nwt_pat = re.compile(r"NWT\s+--\s+", re.IGNORECASE)
    sms_pat = re.compile(r"SMS\s+--\s+", re.IGNORECASE)
    gmg_pat = re.compile(r"GMG\s+--\s+", re.IGNORECASE)
    # Solver params
    mxiter_pat = re.compile(r"MXITER\s*=\s*(\d+)", re.IGNORECASE)
    iter1_pat = re.compile(r"ITER1\s*=\s*(\d+)", re.IGNORECASE)
    hclose_pat = re.compile(r"HCLOSE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)
    rclose_pat = re.compile(r"RCLOSE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)

    current_sp = 0
    current_ts = 0
    current_max_hc = 0.0
    current_max_hc_cell = ""
    current_iters = 0
    current_converged = True
    sp_discrepancies: dict[int, float] = {}

    try:
        with open(lst_path, "r", errors="replace") as fh:
            for line in fh:
                # Solver type
                if not solver_settings.get("solver_type"):
                    if pcg_pat.search(line):
                        solver_settings["solver_type"] = "PCG"
                    elif nwt_pat.search(line):
                        solver_settings["solver_type"] = "NWT"
                    elif sms_pat.search(line):
                        solver_settings["solver_type"] = "SMS"
                    elif gmg_pat.search(line):
                        solver_settings["solver_type"] = "GMG"

                if not solver_settings.get("outer_maximum"):
                    m = mxiter_pat.search(line)
                    if m:
                        solver_settings["outer_maximum"] = int(m.group(1))
                if not solver_settings.get("inner_maximum"):
                    m = iter1_pat.search(line)
                    if m:
                        solver_settings["inner_maximum"] = int(m.group(1))
                if not solver_settings.get("hclose"):
                    m = hclose_pat.search(line)
                    if m:
                        solver_settings["hclose"] = float(m.group(1))
                if not solver_settings.get("rclose"):
                    m = rclose_pat.search(line)
                    if m:
                        solver_settings["rclose"] = float(m.group(1))

                # Stress period / time step
                m = sp_ts_pat.search(line)
                if m:
                    # Save previous timestep
                    if current_sp > 0 or current_ts > 0:
                        timesteps.append({
                            "kper": current_sp - 1,
                            "kstp": current_ts - 1,
                            "outer_iterations": current_iters,
                            "converged": current_converged,
                            "max_dvmax": current_max_hc,
                            "max_dvmax_cell": current_max_hc_cell,
                            "max_rclose": 0.0,
                            "max_rclose_cell": "",
                            "backtracking_events": 0,
                        })
                    current_sp = int(m.group(1))
                    current_ts = int(m.group(2))
                    current_max_hc = 0.0
                    current_max_hc_cell = ""
                    current_iters = 0
                    current_converged = True
                    continue

                # Max head change
                m = head_change_pat.search(line)
                if m:
                    hc = abs(float(m.group(1)))
                    cell_id = m.group(2).replace(" ", "") if m.group(2) else ""
                    if hc > current_max_hc:
                        current_max_hc = hc
                        current_max_hc_cell = cell_id
                    if cell_id:
                        entry = problem_cell_counts[cell_id]
                        entry["occurrences"] += 1
                        entry["affected_sps"].add(current_sp - 1)
                    continue

                # Iteration count
                m = iterations_pat.search(line)
                if m:
                    current_iters = int(m.group(1))
                    continue

                # Percent discrepancy
                m = discrepancy_pat.search(line)
                if m:
                    val = abs(float(m.group(1)))
                    sp_key = current_sp - 1
                    if sp_key not in sp_discrepancies or val > sp_discrepancies[sp_key]:
                        sp_discrepancies[sp_key] = val

                # Failed
                if failed_pat.search(line):
                    current_converged = False

        # Last timestep
        if current_sp > 0 or current_ts > 0:
            timesteps.append({
                "kper": current_sp - 1,
                "kstp": current_ts - 1,
                "outer_iterations": current_iters,
                "converged": current_converged,
                "max_dvmax": current_max_hc,
                "max_dvmax_cell": current_max_hc_cell,
                "max_rclose": 0.0,
                "max_rclose_cell": "",
                "backtracking_events": 0,
            })

    except Exception as e:
        print(f"Warning: Error parsing classic listing: {e}")

    # Build stress period summary
    sp_data: dict[int, list] = defaultdict(list)
    for ts in timesteps:
        sp_data[ts["kper"]].append(ts)

    stress_period_summary = []
    for kper in sorted(sp_data.keys()):
        ts_list = sp_data[kper]
        outer_iters = [t["outer_iterations"] for t in ts_list]
        failed = sum(1 for t in ts_list if not t.get("converged", True))
        total_iters = sum(outer_iters)
        max_iters = max(outer_iters) if outer_iters else 0
        avg_iters = total_iters / len(outer_iters) if outer_iters else 0

        if failed > 0:
            difficulty = "failed"
        elif max_iters > 50:
            difficulty = "high"
        elif max_iters > 20:
            difficulty = "moderate"
        else:
            difficulty = "low"

        stress_period_summary.append({
            "kper": kper,
            "total_outer_iters": total_iters,
            "max_iterations": max_iters,
            "failed_timesteps": failed,
            "avg_iterations": round(avg_iters, 1),
            "percent_discrepancy": sp_discrepancies.get(kper),
            "difficulty": difficulty,
        })

    failed_count = sum(1 for ts in timesteps if not ts.get("converged", True))

    # Problem cells
    problem_cells = []
    for cell_id, info in sorted(
        problem_cell_counts.items(),
        key=lambda x: x[1]["occurrences"],
        reverse=True,
    )[:50]:
        problem_cells.append({
            "cell_id": cell_id,
            "occurrences": info["occurrences"],
            "type": info["type"],
            "affected_sps": sorted(info["affected_sps"]),
        })

    return {
        "model_type": "classic",
        "total_timesteps": len(timesteps),
        "failed_timesteps": failed_count,
        "timesteps": timesteps,
        "stress_period_summary": stress_period_summary,
        "problem_cells": problem_cells,
        "solver_settings": solver_settings,
    }
