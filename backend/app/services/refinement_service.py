"""
Refinement recommendation engine for MODFLOW convergence issues.

Analyzes convergence detail and stress summary to generate actionable
recommendations, and applies modifications to model files.
"""

import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.services.storage import get_storage_service

settings = get_settings()


def generate_recommendations(
    convergence_detail: dict,
    stress_summary: dict,
) -> list[dict]:
    """
    Generate refinement recommendations based on convergence and stress data.

    Args:
        convergence_detail: Parsed convergence data from convergence_parser
        stress_summary: Parsed stress data from stress_extractor

    Returns:
        List of recommendation dicts with id, category, priority, etc.
    """
    recommendations = []
    solver = convergence_detail.get("solver_settings", {})
    sp_summary = convergence_detail.get("stress_period_summary", [])
    problem_cells = convergence_detail.get("problem_cells", [])
    model_type = convergence_detail.get("model_type", "mf6")

    # --- Solver recommendations ---

    # 1. Complexity upgrade
    complexity = solver.get("complexity", "").upper()
    if complexity in ("SIMPLE", "MODERATE"):
        high_iter_sps = [s for s in sp_summary if s.get("difficulty") in ("high", "failed")]
        if len(high_iter_sps) > len(sp_summary) * 0.1:
            next_complexity = "MODERATE" if complexity == "SIMPLE" else "COMPLEX"
            recommendations.append({
                "id": "ims_complexity",
                "category": "solver",
                "priority": "high",
                "title": f"Upgrade solver complexity to {next_complexity}",
                "description": (
                    f"{len(high_iter_sps)} stress periods have high iteration counts or failures. "
                    f"Upgrading from {complexity} to {next_complexity} enables more robust "
                    f"solver defaults for difficult problems."
                ),
                "current_value": complexity,
                "suggested_value": next_complexity,
                "file_modification": {
                    "package": "IMS" if model_type == "mf6" else solver.get("solver_type", "PCG"),
                    "block": "OPTIONS",
                    "variable": "COMPLEXITY",
                    "old_value": complexity,
                    "new_value": next_complexity,
                },
            })

    # 2. Outer maximum increase
    outer_max = solver.get("outer_maximum", 0)
    failed_count = convergence_detail.get("failed_timesteps", 0)
    if failed_count > 0 and outer_max > 0:
        # Check if failures are due to hitting iteration limit
        max_sp_iters = max((s.get("max_iterations", 0) for s in sp_summary), default=0)
        if max_sp_iters >= outer_max * 0.9:
            new_max = min(outer_max * 2, 500)
            recommendations.append({
                "id": "ims_outer_maximum",
                "category": "solver",
                "priority": "high",
                "title": "Increase OUTER_MAXIMUM iterations",
                "description": (
                    f"Solver hit the {outer_max}-iteration limit in {failed_count} timestep(s). "
                    f"Increasing to {new_max} gives the solver more room to converge."
                ),
                "current_value": str(outer_max),
                "suggested_value": str(int(new_max)),
                "file_modification": {
                    "package": "IMS" if model_type == "mf6" else solver.get("solver_type", "PCG"),
                    "block": "OPTIONS" if model_type == "mf6" else None,
                    "variable": "OUTER_MAXIMUM" if model_type == "mf6" else "MXITER",
                    "old_value": str(outer_max),
                    "new_value": str(int(new_max)),
                },
            })

    # 3. Inner RCLOSE adjustment
    inner_rclose = solver.get("inner_rclose", 0)
    if inner_rclose > 0:
        high_residual_sps = [
            s for s in sp_summary
            if s.get("difficulty") in ("high", "failed")
        ]
        if len(high_residual_sps) > 3:
            new_rclose = inner_rclose * 0.1
            recommendations.append({
                "id": "ims_inner_rclose",
                "category": "solver",
                "priority": "medium",
                "title": "Tighten INNER_RCLOSE residual criterion",
                "description": (
                    f"Multiple stress periods have high iteration counts. "
                    f"Tightening INNER_RCLOSE from {inner_rclose:.2e} to {new_rclose:.2e} "
                    f"may improve convergence quality at the cost of more inner iterations."
                ),
                "current_value": f"{inner_rclose:.2e}",
                "suggested_value": f"{new_rclose:.2e}",
                "file_modification": {
                    "package": "IMS",
                    "block": "LINEAR",
                    "variable": "INNER_RCLOSE",
                    "old_value": f"{inner_rclose}",
                    "new_value": f"{new_rclose}",
                },
            })

    # --- Temporal refinement recommendations ---

    # 4. NSTP increase for high-iteration stress periods
    if stress_summary and stress_summary.get("periods"):
        periods = stress_summary["periods"]
        for sp_info in sp_summary:
            kper = sp_info["kper"]
            if sp_info.get("difficulty") not in ("high", "failed"):
                continue

            # Find matching stress period data
            matching_period = None
            for p in periods:
                if p.get("kper") == kper:
                    matching_period = p
                    break

            if matching_period:
                current_nstp = matching_period.get("nstp", 1)
                if current_nstp < 10:
                    new_nstp = min(current_nstp * 2, 20)
                    recommendations.append({
                        "id": f"tdis_nstp_sp{kper}",
                        "category": "temporal",
                        "priority": "medium",
                        "title": f"Increase timesteps in SP {kper + 1}",
                        "description": (
                            f"Stress period {kper + 1} has {sp_info.get('difficulty')} difficulty "
                            f"({sp_info.get('max_iterations', 0)} max iterations). "
                            f"Increasing NSTP from {current_nstp} to {new_nstp} reduces the "
                            f"stress change per timestep."
                        ),
                        "current_value": str(current_nstp),
                        "suggested_value": str(new_nstp),
                        "file_modification": {
                            "package": "TDIS" if model_type == "mf6" else "DIS",
                            "block": "PERIODDATA" if model_type == "mf6" else None,
                            "variable": "NSTP",
                            "stress_period": kper,
                            "old_value": str(current_nstp),
                            "new_value": str(new_nstp),
                        },
                    })
                    break  # Only suggest for the worst SP to avoid overwhelming

    # 5. Large stress transitions
    if stress_summary and stress_summary.get("periods"):
        periods = stress_summary["periods"]
        packages = stress_summary.get("packages", [])
        for pkg in packages:
            prev_rate = None
            for p in periods:
                pkg_data = p.get(pkg, {})
                rate = pkg_data.get("total_rate", 0)
                if prev_rate is not None and prev_rate != 0:
                    change_ratio = abs(rate - prev_rate) / abs(prev_rate)
                    if change_ratio > 5.0:  # > 500% change
                        recommendations.append({
                            "id": f"stress_transition_{pkg}_sp{p['kper']}",
                            "category": "package",
                            "priority": "low",
                            "title": f"Large {pkg} rate change at SP {p['kper'] + 1}",
                            "description": (
                                f"{pkg} rate changes by {change_ratio:.0f}x between SP {p['kper']} "
                                f"and SP {p['kper'] + 1} ({prev_rate:.2e} to {rate:.2e}). "
                                f"Consider adding transitional stress periods or reducing "
                                f"the rate change magnitude."
                            ),
                            "current_value": f"{prev_rate:.2e}",
                            "suggested_value": "Add transitional periods",
                            "file_modification": None,
                        })
                        break  # One per package
                prev_rate = rate

    # 6. Problem cells near specific packages
    if problem_cells and len(problem_cells) > 5:
        top_cell = problem_cells[0]
        recommendations.append({
            "id": "problem_cells_review",
            "category": "package",
            "priority": "medium",
            "title": f"Review problem cell {top_cell['cell_id']}",
            "description": (
                f"Cell {top_cell['cell_id']} appears in {top_cell['occurrences']} "
                f"convergence iterations across {len(top_cell.get('affected_sps', []))} "
                f"stress periods. Check if this cell is near a boundary condition "
                f"with oscillating head/flux behavior."
            ),
            "current_value": f"{top_cell['occurrences']} occurrences",
            "suggested_value": "Review cell location and nearby BCs",
            "file_modification": None,
        })

    # Deduplicate by id
    seen_ids = set()
    unique_recs = []
    for r in recommendations:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            unique_recs.append(r)

    return unique_recs


def apply_refinements(
    project_id: str,
    storage_path: str,
    refinement_ids: list[str],
    recommendations: list[dict],
    model_type: str = "mf6",
) -> dict:
    """
    Apply selected refinements to model files in MinIO.

    1. Backup current files
    2. Download, modify, and re-upload affected files

    Args:
        project_id: Project UUID
        storage_path: MinIO prefix for model files
        refinement_ids: List of recommendation IDs to apply
        recommendations: Full list of recommendations
        model_type: Model type string

    Returns:
        Dict with backup_timestamp and modified_files list
    """
    storage = get_storage_service()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Filter recommendations to apply
    to_apply = [r for r in recommendations if r["id"] in refinement_ids and r.get("file_modification")]
    if not to_apply:
        return {"backup_timestamp": timestamp, "modified_files": [], "message": "No applicable refinements"}

    # Group by package to minimize file operations
    by_package: dict[str, list] = {}
    for rec in to_apply:
        mod = rec["file_modification"]
        pkg = mod["package"]
        by_package.setdefault(pkg, []).append(rec)

    modified_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download all model files
        try:
            objects = list(storage.list_objects(
                settings.minio_bucket_models,
                prefix=storage_path,
                recursive=True,
            ))
        except Exception as e:
            return {"error": f"Failed to list model files: {e}"}

        # Find files for each package
        for pkg_name, recs in by_package.items():
            # Find the file for this package
            pkg_ext = _package_to_extension(pkg_name, model_type)
            target_obj = None

            for obj_name in objects:
                filename = obj_name.rsplit("/", 1)[-1].lower()
                if pkg_ext and filename.endswith(pkg_ext):
                    target_obj = obj_name
                    break
                # Also check by package name in filename
                if pkg_name.lower() in filename:
                    target_obj = obj_name
                    break

            if not target_obj:
                continue

            filename = target_obj.rsplit("/", 1)[-1]
            local_path = temp_path / filename

            # Download file
            storage.download_to_file(
                settings.minio_bucket_models, target_obj, local_path
            )

            # Backup original
            backup_path = f"projects/{project_id}/model_backup/{timestamp}/{filename}"
            storage.upload_file(
                settings.minio_bucket_models,
                backup_path,
                local_path,
            )

            # Apply modifications
            content = local_path.read_text(errors="replace")
            for rec in recs:
                mod = rec["file_modification"]
                if model_type == "mf6":
                    content = _apply_mf6_modification(content, mod)
                else:
                    content = _apply_classic_modification(content, mod)

            # Write modified content
            local_path.write_text(content)

            # Upload back
            storage.upload_file(
                settings.minio_bucket_models, target_obj, local_path
            )

            modified_files.append({
                "file": filename,
                "refinements": [r["id"] for r in recs],
                "backup": backup_path,
            })

    return {
        "backup_timestamp": timestamp,
        "modified_files": modified_files,
    }


def revert_refinements(
    project_id: str,
    storage_path: str,
    backup_timestamp: str,
) -> dict:
    """
    Revert model files from a backup.

    Args:
        project_id: Project UUID
        storage_path: MinIO prefix for model files
        backup_timestamp: Timestamp of the backup to restore

    Returns:
        Dict with restored_files list
    """
    storage = get_storage_service()
    backup_prefix = f"projects/{project_id}/model_backup/{backup_timestamp}/"

    restored = []

    try:
        backup_objects = list(storage.list_objects(
            settings.minio_bucket_models,
            prefix=backup_prefix,
            recursive=True,
        ))
    except Exception as e:
        return {"error": f"Failed to list backup files: {e}"}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for backup_obj in backup_objects:
            filename = backup_obj.rsplit("/", 1)[-1]
            local_path = temp_path / filename

            # Download backup
            storage.download_to_file(
                settings.minio_bucket_models, backup_obj, local_path
            )

            # Find original path in storage
            target_obj = f"{storage_path}/{filename}"

            # Upload restored file
            storage.upload_file(
                settings.minio_bucket_models, target_obj, local_path
            )

            restored.append(filename)

    return {"restored_files": restored, "backup_timestamp": backup_timestamp}


def _package_to_extension(pkg_name: str, model_type: str) -> Optional[str]:
    """Map package name to file extension."""
    mf6_map = {
        "IMS": ".ims", "TDIS": ".tdis", "DIS": ".dis", "DISV": ".disv",
        "NPF": ".npf", "STO": ".sto", "IC": ".ic", "OC": ".oc",
        "WEL": ".wel", "RCH": ".rch", "EVT": ".evt", "CHD": ".chd",
        "GHB": ".ghb", "DRN": ".drn", "RIV": ".riv", "SFR": ".sfr",
        "MAW": ".maw", "LAK": ".lak", "UZF": ".uzf",
    }
    classic_map = {
        "PCG": ".pcg", "NWT": ".nwt", "SMS": ".sms", "GMG": ".gmg",
        "DIS": ".dis", "BAS6": ".bas", "LPF": ".lpf", "BCF": ".bcf",
        "UPW": ".upw", "WEL": ".wel", "RCH": ".rch", "EVT": ".evt",
        "CHD": ".chd", "GHB": ".ghb", "DRN": ".drn", "RIV": ".riv",
        "SFR": ".sfr", "OC": ".oc",
    }

    ext_map = mf6_map if model_type == "mf6" else classic_map
    return ext_map.get(pkg_name.upper())


def _apply_mf6_modification(content: str, mod: dict) -> str:
    """Apply a modification to MF6 block-structured file content."""
    variable = mod.get("variable", "")
    old_value = mod.get("old_value", "")
    new_value = mod.get("new_value", "")
    block = mod.get("block", "")

    if not variable or not new_value:
        return content

    # For block-based files, find the variable within the correct block
    lines = content.split("\n")
    in_target_block = False if block else True
    modified = False

    for i, line in enumerate(lines):
        stripped = line.strip().upper()

        # Track block entry/exit
        if block:
            if stripped.startswith("BEGIN") and block.upper() in stripped:
                in_target_block = True
                continue
            if stripped.startswith("END") and block.upper() in stripped:
                in_target_block = False
                continue

        if not in_target_block:
            continue

        # Look for the variable
        if variable.upper() in stripped:
            # Replace the value
            pattern = re.compile(
                rf"({re.escape(variable)})\s+{re.escape(old_value)}",
                re.IGNORECASE,
            )
            new_line = pattern.sub(rf"\1  {new_value}", line)
            if new_line != line:
                lines[i] = new_line
                modified = True
                break

            # Try simpler replacement for keyword-style options
            pattern2 = re.compile(
                rf"\b{re.escape(old_value)}\b",
                re.IGNORECASE,
            )
            if variable.upper() in stripped and pattern2.search(line):
                lines[i] = pattern2.sub(new_value, line)
                modified = True
                break

    if not modified:
        # If variable not found, try direct old_value -> new_value replacement
        content = content.replace(old_value, new_value, 1)
        return content

    return "\n".join(lines)


def _apply_classic_modification(content: str, mod: dict) -> str:
    """Apply a modification to classic MODFLOW free-format file content."""
    old_value = mod.get("old_value", "")
    new_value = mod.get("new_value", "")

    if old_value and new_value:
        # Simple value replacement
        content = content.replace(old_value, new_value, 1)

    return content
