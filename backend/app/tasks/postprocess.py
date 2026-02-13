"""Post-processing task definitions for simulation results."""

import json
import re
import tempfile
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
from celery import current_task
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.base import SessionLocal
from app.models.project import Project, Run, RunStatus
from app.services.storage import get_storage_service
from celery_app import celery_app

settings = get_settings()

# Module-level context for full-mode streaming uploads from _process_heads
_full_mode_output_prefix: Optional[str] = None


@celery_app.task(bind=True, name="app.tasks.postprocess.extract_results")
def extract_results(self, run_id: str, project_id: str, quick_mode: bool = True) -> dict:
    """
    Extract and process simulation results.

    Downloads output files from MinIO, processes heads/budget/listing,
    uploads processed JSON slices back to MinIO, and updates Run.convergence_info.

    Args:
        run_id: UUID of the run record
        project_id: UUID of the project
        quick_mode: If True (default), only process last timestep for fast results.
                   Full processing happens in background.

    Returns:
        Dict with processing status
    """
    storage = get_storage_service()

    with SessionLocal() as db:
        project = db.execute(
            select(Project).where(Project.id == UUID(project_id))
        ).scalar_one_or_none()

        run = db.execute(
            select(Run).where(Run.id == UUID(run_id))
        ).scalar_one_or_none()

        if not project or not run:
            return {"error": "Project or run not found"}

        if not run.results_path:
            return {"error": "No results path set on run"}

        model_type = project.model_type.value if project.model_type else "mf6"

        # Update convergence_info to indicate post-processing started
        run.convergence_info = {
            **(run.convergence_info or {}),
            "postprocess_status": "running",
        }
        db.commit()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download output files from MinIO (stream to disk, not into memory)
            try:
                output_objects = storage.list_objects(
                    settings.minio_bucket_models,
                    prefix=run.results_path,
                    recursive=True,
                )

                local_files = {}
                for obj_name in output_objects:
                    filename = obj_name.rsplit("/", 1)[-1]
                    local_path = temp_path / filename
                    storage.download_to_file(
                        settings.minio_bucket_models, obj_name, local_path
                    )
                    ext = local_path.suffix.lower()
                    # For .lst files, prefer the model listing over mfsim.lst
                    # (mfsim.lst is a simulation log; the model .lst has the
                    # volumetric budget with PERCENT DISCREPANCY).
                    if ext == ".lst" and filename.lower() == "mfsim.lst" and ext in local_files:
                        pass  # don't overwrite model listing
                    else:
                        local_files[ext] = local_path
                    # Also store by full filename for listing files
                    local_files[filename.lower()] = local_path

            except Exception as e:
                run.convergence_info = {
                    **(run.convergence_info or {}),
                    "postprocess_status": "failed",
                    "postprocess_error": f"Failed to download output files: {e}",
                }
                db.commit()
                return {"error": str(e)}

            # Track completed stages for frontend display
            completed_stages = []

            def update_progress(current, total, message, stage=None):
                """Update progress in the database for frontend polling."""
                nonlocal completed_stages
                if stage and stage not in completed_stages:
                    completed_stages.append(stage)
                run.convergence_info = {
                    **(run.convergence_info or {}),
                    "postprocess_status": "running",
                    "postprocess_progress": int((current / total) * 100) if total > 0 else 0,
                    "postprocess_message": message,
                    "postprocess_stage": stage,
                    "postprocess_completed": list(completed_stages),
                }
                db.commit()

            # Process heads then budget sequentially to keep memory footprint low
            update_progress(5, 100, "Processing head data...", "heads")

            heads_result = {}
            budget_result = {}
            budget_warning = None

            try:
                global _full_mode_output_prefix
                if not quick_mode:
                    _full_mode_output_prefix = run.results_path
                heads_result = _process_heads(
                    local_files, model_type, project, quick_mode, None
                )
                _full_mode_output_prefix = None
                update_progress(35, 100, "Head data processed", "heads")
            except Exception as e:
                _full_mode_output_prefix = None
                print(f"Warning: heads processing failed: {e}")
                heads_result = {"summary": {}, "head_arrays": {}}

            update_progress(40, 100, "Processing budget data...", "budget")

            try:
                budget_result = _process_budget(
                    local_files, model_type, quick_mode, None
                )
                update_progress(70, 100, "Budget data processed", "budget")
            except Exception as e:
                print(f"Warning: budget processing failed: {e}")
                budget_result = {}
                budget_warning = f"Budget processing failed: {e}"

            # Flag if no budget data was produced (no CBC file found)
            if not budget_result and not budget_warning:
                budget_warning = (
                    "No cell budget file (CBC) found in simulation output. "
                    "Enable 'Save Water Budget' when starting a simulation to generate budget data."
                )

            # Parse listing file
            update_progress(72, 100, "Parsing listing file for convergence...", "listing")
            convergence_result = _parse_listing_file(local_files)

            # --- Detailed convergence analysis ---
            update_progress(73, 100, "Running detailed convergence analysis...", "listing")
            convergence_detail = None
            try:
                from app.services.convergence_parser import parse_mf6_listing, parse_classic_listing

                if model_type == "mf6":
                    mfsim_lst = local_files.get("mfsim.lst")
                    # Find model listing (non-mfsim .lst file)
                    flow_lst = None
                    for key, path in local_files.items():
                        if key.endswith(".lst") and key != "mfsim.lst" and isinstance(path, Path):
                            flow_lst = path
                            break
                    if not flow_lst:
                        flow_lst = local_files.get(".lst")
                    convergence_detail = parse_mf6_listing(mfsim_lst, flow_lst)
                else:
                    lst_file = local_files.get(".lst") or local_files.get(".list")
                    if lst_file:
                        convergence_detail = parse_classic_listing(lst_file)

                if convergence_detail:
                    detail_json = json.dumps(convergence_detail, default=_json_serialize)
                    storage.upload_bytes(
                        settings.minio_bucket_models,
                        f"{output_prefix}/processed/convergence_detail.json",
                        detail_json.encode("utf-8"),
                        content_type="application/json",
                    )
                    completed_stages.append("convergence_detail")
            except Exception as e:
                print(f"Warning: Detailed convergence parsing failed: {e}")

            # --- Stress data extraction ---
            update_progress(74, 100, "Extracting stress period data...", "listing")
            try:
                from app.services.stress_extractor import extract_stress_summary

                if project.storage_path:
                    stress_summary = extract_stress_summary(
                        project_id=str(project.id),
                        storage_path=project.storage_path,
                        model_type=model_type,
                        nper=project.nper or 1,
                        stress_period_data=project.stress_period_data,
                    )
                    if stress_summary and stress_summary.get("packages"):
                        stress_json = json.dumps(stress_summary, default=_json_serialize)
                        storage.upload_bytes(
                            settings.minio_bucket_models,
                            f"{output_prefix}/processed/stress_summary.json",
                            stress_json.encode("utf-8"),
                            content_type="application/json",
                        )
                        completed_stages.append("stress_summary")
            except Exception as e:
                print(f"Warning: Stress extraction failed: {e}")

            # Fallback: if listing file had no percent discrepancy lines but we
            # have budget data, compute mass balance error from CBC periods.
            if convergence_result.get("mass_balance_error_pct") is None and budget_result:
                periods = budget_result.get("periods", {})
                if periods:
                    budget_discrepancies = []
                    for p in periods.values():
                        pd_val = p.get("percent_discrepancy")
                        if pd_val is not None:
                            budget_discrepancies.append(abs(pd_val))
                    if budget_discrepancies:
                        convergence_result["mass_balance_error_pct"] = max(budget_discrepancies)
                        convergence_result["percent_discrepancies"] = budget_discrepancies
                        convergence_result["mass_balance_source"] = "budget"

            # Determine grid type
            grid_type = _detect_grid_type(model_type, project)

            # Extract grid geometry for unstructured grids
            if grid_type in ("vertex", "unstructured"):
                update_progress(75, 100, "Extracting grid geometry...", "geometry")
                _extract_grid_geometry(
                    storage, project, local_files, run.results_path, model_type,
                )

            # Pre-build HDS index for fast on-demand access
            # This caches byte offsets in Redis so first Dashboard load is fast
            update_progress(78, 100, "Building HDS index for fast access...", "geometry")
            try:
                from app.services.hds_streaming import build_hds_index
                build_hds_index(project_id, run_id, run.results_path)
            except Exception as e:
                print(f"Warning: Could not pre-build HDS index: {e}")

            update_progress(80, 100, "Uploading processed results...", "uploading")

            # Build results summary
            results_summary = {
                "heads_summary": heads_result.get("summary", {}),
                "budget": budget_result,
                "convergence": convergence_result,
                "metadata": {
                    "model_type": model_type,
                    "grid_type": grid_type,
                    "nlay": project.nlay or 1,
                    "nrow": project.nrow or 1,
                    "ncol": project.ncol or 1,
                    "nper": project.nper or 1,
                    "xoff": project.xoff,
                    "yoff": project.yoff,
                    "angrot": project.angrot,
                    "epsg": project.epsg,
                    "stress_period_data": project.stress_period_data,
                    "start_date": str(project.start_date) if project.start_date else None,
                    "time_unit": project.time_unit,
                    "length_unit": project.length_unit,
                    "delr": project.delr,
                    "delc": project.delc,
                },
            }

            # Upload results_summary.json
            output_prefix = run.results_path
            summary_json = json.dumps(results_summary, default=_json_serialize)
            storage.upload_bytes(
                settings.minio_bucket_models,
                f"{output_prefix}/results_summary.json",
                summary_json.encode("utf-8"),
                content_type="application/json",
            )

            # Upload per-slice head files (only in quick mode — full mode
            # streams uploads during processing to avoid accumulating in memory)
            head_arrays = heads_result.get("head_arrays", {})
            for key, slice_data in head_arrays.items():
                slice_json = json.dumps(slice_data, default=_json_serialize)
                layer = slice_data["layer"]
                kper = slice_data["kper"]
                kstp = slice_data["kstp"]
                obj_name = (
                    f"{output_prefix}/processed/"
                    f"heads_L{layer}_SP{kper}_TS{kstp}.json"
                )
                storage.upload_bytes(
                    settings.minio_bucket_models,
                    obj_name,
                    slice_json.encode("utf-8"),
                    content_type="application/json",
                )

            # Upload budget.json
            if budget_result:
                budget_json = json.dumps(budget_result, default=_json_serialize)
                storage.upload_bytes(
                    settings.minio_bucket_models,
                    f"{output_prefix}/processed/budget.json",
                    budget_json.encode("utf-8"),
                    content_type="application/json",
                )

            update_progress(95, 100, "Finalizing...", "finalizing")

            # Update Run.convergence_info
            heads_summary = heads_result.get("summary", {})
            summary_stats = {
                "postprocess_status": "completed",
                "converged": convergence_result.get("converged", True),
                "mass_balance_error_pct": convergence_result.get(
                    "mass_balance_error_pct"
                ),
                "max_head": heads_summary.get("max_head"),
                "min_head": heads_summary.get("min_head"),
                "nstp_total": heads_summary.get("nstp_total", 0),
                "nlay": project.nlay or 1,
                "nrow": project.nrow or 1,
                "ncol": project.ncol or 1,
                "quick_mode": quick_mode,
                "timesteps_processed": heads_summary.get("timesteps_processed", 0),
            }

            if convergence_result.get("max_head_changes"):
                summary_stats["max_head_changes"] = convergence_result[
                    "max_head_changes"
                ]

            if convergence_result.get("warnings"):
                summary_stats["warnings"] = convergence_result["warnings"]

            if budget_warning:
                summary_stats["budget_warning"] = budget_warning

            run.convergence_info = {
                **(run.convergence_info or {}),
                **summary_stats,
            }
            db.commit()

            return {
                "run_id": run_id,
                "status": "completed",
                "message": "Post-processing completed",
                "slices_uploaded": len(head_arrays),
            }


def _process_heads(
    local_files: dict, model_type: str, project: Project,
    quick_mode: bool = False, progress_callback=None
) -> dict:
    """
    Process binary head file using FloPy.

    Uses HeadUFile for unstructured (USG) models, HeadFile for structured.

    In quick_mode, we now do minimal processing - just extract metadata and
    statistics. The actual head slice data is fetched on-demand when requested.

    Args:
        local_files: Dict mapping extension/filename to local Path
        model_type: MODFLOW model type string
        project: Project database object
        quick_mode: If True, only extract metadata and min/max statistics
        progress_callback: Optional callback(current, total, message) for progress updates

    Returns:
        Dict with summary and head_arrays
    """
    import flopy.utils

    # Look for head file with either .hds or .hed extension
    hds_path = local_files.get(".hds") or local_files.get(".hed")
    if not hds_path or not hds_path.exists():
        return {"summary": {}, "head_arrays": {}}

    is_unstructured = model_type == "mfusg"

    try:
        if is_unstructured:
            hds = flopy.utils.HeadUFile(str(hds_path))
        else:
            hds = flopy.utils.HeadFile(str(hds_path))
    except Exception:
        return {"summary": {}, "head_arrays": {}}

    kstpkper_list = hds.get_kstpkper()
    times = hds.get_times()

    # In quick mode, just scan for min/max without extracting full slices
    # The actual slice data is fetched on-demand when requested
    if quick_mode:
        # Sample 3 timesteps for min/max statistics
        if len(kstpkper_list) > 3:
            indices_to_sample = [0, len(kstpkper_list) // 2, len(kstpkper_list) - 1]
        else:
            indices_to_sample = list(range(len(kstpkper_list)))

        timesteps_to_sample = [kstpkper_list[i] for i in indices_to_sample]

        global_max = None
        global_min = None

        for kstp, kper in timesteps_to_sample:
            try:
                data = hds.get_data(kstpkper=(kstp, kper))
            except Exception:
                continue

            if is_unstructured:
                layers_data = data if isinstance(data, list) else [data]
            else:
                layers_data = [data[i] for i in range(data.shape[0])]

            for arr in layers_data:
                arr = np.asarray(arr)
                masked = np.where(
                    (np.abs(arr) > 1e20) | (arr < -880) | (np.isclose(arr, 999.0)),
                    np.nan,
                    arr,
                )
                valid = masked[~np.isnan(masked)]
                if valid.size > 0:
                    layer_max = float(np.max(valid))
                    layer_min = float(np.min(valid))
                    if global_max is None or layer_max > global_max:
                        global_max = layer_max
                    if global_min is None or layer_min < global_min:
                        global_min = layer_min

        hds.close()

        # In quick mode, don't pre-extract any slices - they'll be fetched on-demand
        summary = {
            "nstp_total": len(kstpkper_list),
            "kstpkper_list": [[int(ks), int(kp)] for ks, kp in kstpkper_list],
            "times": [float(t) for t in times] if times else [],
            "max_head": global_max,
            "min_head": global_min,
            "quick_mode": True,
            "timesteps_processed": 0,  # No slices pre-processed
            "on_demand": True,  # Indicate slices are fetched on-demand
        }

        return {"summary": summary, "head_arrays": {}}

    # Full processing mode - extract one timestep at a time, upload immediately,
    # then discard to keep memory low. Only accumulate summary statistics.
    timesteps_to_process = kstpkper_list

    global_max = None
    global_min = None
    slices_uploaded = 0
    total_steps = len(timesteps_to_process)

    # We need storage and output_prefix for streaming uploads.
    # These are passed via a module-level context set by the caller.
    from app.services.storage import get_storage_service
    storage = get_storage_service()

    for step_idx, (kstp, kper) in enumerate(timesteps_to_process):
        if progress_callback:
            progress_callback(
                step_idx, total_steps,
                f"Processing heads: timestep {step_idx + 1}/{total_steps}"
            )

        try:
            data = hds.get_data(kstpkper=(kstp, kper))
        except Exception:
            continue

        if is_unstructured:
            layers_data = data if isinstance(data, list) else [data]
            nlay = len(layers_data)
        else:
            nlay = data.shape[0]
            layers_data = [data[i] for i in range(nlay)]

        for layer_idx in range(nlay):
            arr = np.asarray(layers_data[layer_idx])
            masked = np.where(
                (np.abs(arr) > 1e20) | (arr < -880) | (np.isclose(arr, 999.0)),
                np.nan,
                arr,
            )

            valid = masked[~np.isnan(masked)]
            if valid.size > 0:
                layer_max = float(np.max(valid))
                layer_min = float(np.min(valid))
                if global_max is None or layer_max > global_max:
                    global_max = layer_max
                if global_min is None or layer_min < global_min:
                    global_min = layer_min

            # Build slice data for this single timestep/layer
            if arr.ndim == 1:
                flat = [None if (v != v) else v for v in masked.tolist()]
                data_list = [flat]
                shape = [1, len(flat)]
            else:
                data_list = []
                for row in masked.tolist():
                    data_list.append(
                        [None if (v != v) else v for v in row]
                    )
                shape = list(arr.shape)

            slice_data = {
                "layer": layer_idx,
                "kper": int(kper),
                "kstp": int(kstp),
                "shape": shape,
                "data": data_list,
            }

            # Upload immediately and discard — don't accumulate in memory
            if _full_mode_output_prefix:
                slice_json = json.dumps(slice_data, default=_json_serialize)
                obj_name = (
                    f"{_full_mode_output_prefix}/processed/"
                    f"heads_L{layer_idx}_SP{kper}_TS{kstp}.json"
                )
                storage.upload_bytes(
                    settings.minio_bucket_models,
                    obj_name,
                    slice_json.encode("utf-8"),
                    content_type="application/json",
                )
                slices_uploaded += 1

        # Free the timestep data
        del data, layers_data

    hds.close()

    summary = {
        "nstp_total": len(kstpkper_list),
        "kstpkper_list": [[int(ks), int(kp)] for ks, kp in kstpkper_list],
        "times": [float(t) for t in times] if times else [],
        "max_head": global_max,
        "min_head": global_min,
        "quick_mode": quick_mode,
        "timesteps_processed": len(timesteps_to_process),
        "slices_uploaded": slices_uploaded,
    }

    # Return empty head_arrays — they've already been uploaded individually
    return {"summary": summary, "head_arrays": {}}


# Internal cell-to-cell flow records that should NOT be included in mass balance totals.
# These appear in CBC files but are not external sources/sinks.
_INTERNAL_FLOW_RECORDS = {
    'FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
    'FLOW_RIGHT_FACE', 'FLOW_FRONT_FACE', 'FLOW_LOWER_FACE',
    'FLOW JA FACE', 'FLOW_JA_FACE', 'FLOW-JA-FACE',
    'DATA-SPDIS', 'DATA-SAT', 'DATA-STOSS', 'DATA-STOSY',
}


def _is_internal_flow(name: str) -> bool:
    """Check if a budget record name is an internal cell-to-cell flow."""
    return name.upper() in _INTERNAL_FLOW_RECORDS


def _process_budget(
    local_files: dict, model_type: str,
    quick_mode: bool = False, progress_callback=None
) -> dict:
    """
    Process cell budget file using FloPy.

    In quick_mode, only process the last timestep for a quick summary.
    Full budget data can be computed on-demand if needed.

    Args:
        local_files: Dict mapping extension/filename to local Path
        model_type: MODFLOW model type string
        quick_mode: If True, only process the last timestep for quick summary
        progress_callback: Optional callback(current, total, message) for progress updates

    Returns:
        Dict with record_names and per-period budget breakdown
    """
    import flopy.utils

    # Look for budget file with .cbc, .bud, or .cbb extension
    cbc_path = local_files.get(".cbc") or local_files.get(".bud") or local_files.get(".cbb")
    if not cbc_path or not cbc_path.exists():
        return {}

    try:
        # USG writes CBC in double precision
        if model_type == "mfusg":
            cbc = flopy.utils.CellBudgetFile(str(cbc_path), precision="double")
        else:
            cbc = flopy.utils.CellBudgetFile(str(cbc_path))
    except Exception:
        return {}

    # Get unique record names (byte strings)
    try:
        raw_names = cbc.get_unique_record_names()
    except Exception:
        cbc.close()
        return {}

    record_names = []
    for name in raw_names:
        if isinstance(name, bytes):
            record_names.append(name.decode("ascii", errors="ignore").strip())
        else:
            record_names.append(str(name).strip())

    kstpkper_list = cbc.get_kstpkper()

    # In quick mode, only process the last timestep for a quick summary
    if quick_mode and len(kstpkper_list) > 1:
        timesteps_to_process = [kstpkper_list[-1]]  # Just the last timestep
    else:
        timesteps_to_process = kstpkper_list

    periods = {}
    total_steps = len(timesteps_to_process)

    for step_idx, (kstp, kper) in enumerate(timesteps_to_process):
        if progress_callback:
            progress_callback(
                step_idx, total_steps,
                f"Processing budget: timestep {step_idx + 1}/{total_steps}"
            )

        period_key = f"SP{kper}_TS{kstp}"
        period_in = {}
        period_out = {}
        total_in = 0.0
        total_out = 0.0

        for raw_name, clean_name in zip(raw_names, record_names):
            try:
                records = cbc.get_data(kstpkper=(kstp, kper), text=raw_name)
            except Exception:
                continue

            comp_in = 0.0
            comp_out = 0.0

            for rec in records:
                if isinstance(rec, np.recarray):
                    # List-based budget (e.g., wells)
                    if "q" in rec.dtype.names:
                        vals = rec["q"]
                    elif "flux" in rec.dtype.names:
                        vals = rec["flux"]
                    else:
                        continue
                else:
                    vals = np.asarray(rec).flatten()

                positives = vals[vals > 0]
                negatives = vals[vals < 0]
                comp_in += float(np.sum(positives)) if positives.size > 0 else 0.0
                comp_out += float(np.sum(np.abs(negatives))) if negatives.size > 0 else 0.0

            period_in[clean_name] = comp_in
            period_out[clean_name] = comp_out
            # Only count external stresses in mass balance totals
            if not _is_internal_flow(clean_name):
                total_in += comp_in
                total_out += comp_out

        discrepancy = total_in - total_out
        percent_discrepancy = (
            (discrepancy / ((total_in + total_out) / 2.0)) * 100.0
            if (total_in + total_out) > 0
            else 0.0
        )

        periods[period_key] = {
            "kstp": int(kstp),
            "kper": int(kper),
            "in": period_in,
            "out": period_out,
            "total_in": total_in,
            "total_out": total_out,
            "discrepancy": discrepancy,
            "percent_discrepancy": percent_discrepancy,
        }

    cbc.close()

    return {
        "record_names": record_names,
        "periods": periods,
        "quick_mode": quick_mode,
        "total_timesteps": len(kstpkper_list),
        "timesteps_processed": len(timesteps_to_process),
    }


def _parse_listing_file(local_files: dict) -> dict:
    """
    Parse MODFLOW listing file for convergence info.

    Args:
        local_files: Dict mapping extension/filename to local Path

    Returns:
        Dict with converged flag, mass balance error, max head changes, warnings
    """
    lst_path = local_files.get(".lst") or local_files.get(".list")
    if not lst_path or not lst_path.exists():
        return {
            "converged": True,
            "mass_balance_error_pct": None,
            "max_head_changes": [],
            "warnings": [],
        }

    # Parse line-by-line to avoid loading entire listing file into memory
    solver_failed = False
    warnings = []
    cumulative_discrepancies = []
    max_head_changes = []
    package_warning_counts = {
        "SFR": 0, "LAK": 0, "UZF": 0, "SWR": 0,
    }

    solver_failure_patterns = [
        re.compile(r"(?<!SFR\s)(?<!LAK\s)(?<!UZF\s)(?<!SWR\s)FAILED\s+TO\s+MEET\s+SOLVER", re.IGNORECASE),
        re.compile(r"SOLVER\s+FAILED\s+TO\s+CONVERGE", re.IGNORECASE),
        re.compile(r"(?:PCG|GMG|NWT|IMS|SMS).*FAILED", re.IGNORECASE),
        re.compile(r"BUDGET\s+DID\s+NOT\s+CONVERGE", re.IGNORECASE),
        re.compile(r"SIMULATION\s+FAILED\s+TO\s+CONVERGE", re.IGNORECASE),
    ]
    discrepancy_pattern = re.compile(
        r"PERCENT\s+DISCREPANCY\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    head_change_pattern = re.compile(
        r"MAXIMUM\s+HEAD\s+CHANGE\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )

    try:
        with open(lst_path, "r", errors="replace") as fh:
            for line in fh:
                line_upper = line.upper()

                # Solver failure check
                if not solver_failed and "FAILED" in line_upper:
                    for pat in solver_failure_patterns:
                        if pat.search(line):
                            solver_failed = True
                            break

                # Package warnings
                for pkg in package_warning_counts:
                    if f"{pkg}" in line_upper and "FAILED" in line_upper and "CONVERGE" in line_upper:
                        package_warning_counts[pkg] += 1

                # Percent discrepancy
                if "PERCENT DISCREPANCY" in line_upper:
                    matches = discrepancy_pattern.findall(line)
                    if matches:
                        cumulative_discrepancies.append(abs(float(matches[0])))

                # Max head change
                if "MAXIMUM" in line_upper and "HEAD" in line_upper and "CHANGE" in line_upper:
                    m = head_change_pattern.search(line)
                    if m:
                        max_head_changes.append(abs(float(m.group(1))))

    except Exception:
        return {
            "converged": True,
            "mass_balance_error_pct": None,
            "max_head_changes": [],
            "warnings": [],
        }

    for pkg, count in package_warning_counts.items():
        if count > 0:
            pkg_names = {"SFR": "stream", "LAK": "lake", "UZF": "unsaturated zone", "SWR": "surface water"}
            warnings.append(f"{pkg} ({pkg_names[pkg]}) convergence warning ({count} occurrence(s))")

    max_cumulative_error = max(cumulative_discrepancies) if cumulative_discrepancies else None

    # Determine convergence status
    # Converged if: no solver failure AND cumulative mass balance < 1%
    if solver_failed:
        converged = False
    elif max_cumulative_error is not None and max_cumulative_error > 1.0:
        converged = False
        warnings.append(f"Cumulative mass balance error {max_cumulative_error:.2f}% exceeds 1%")
    else:
        converged = True

    return {
        "converged": converged,
        "mass_balance_error_pct": max_cumulative_error,
        "max_head_changes": max_head_changes,
        "percent_discrepancies": cumulative_discrepancies,
        "warnings": warnings,
    }


def _json_serialize(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _detect_grid_type(model_type: str, project: Project) -> str:
    """Determine grid type from model type and packages."""
    if model_type == "mfusg":
        return "unstructured"
    if model_type == "mf6":
        pkgs = project.packages or {}
        all_names = [
            str(v).upper() for v in list(pkgs.keys()) + list(pkgs.values())
        ]
        if any("DISV" in n for n in all_names):
            return "vertex"
        if any("DISU" in n for n in all_names):
            return "unstructured"
    return "structured"


def _extract_grid_geometry(
    storage, project: Project, local_files: dict,
    output_prefix: str, model_type: str,
) -> None:
    """
    Extract 2D cell polygon geometry from the model for contour rendering.

    For MF6 with a .grb file: uses MfGrdFile for fast loading.
    For USG: loads the full model from storage, uses gridspec data or
    modelgrid vertices to build cell polygons.

    Saves grid_geometry.json to MinIO with cell vertices per layer.
    """
    import flopy

    grid = None
    model = None

    # Try .grb file first (MF6 binary grid file, written after simulation)
    grb_path = local_files.get(".grb")
    if grb_path and grb_path.exists():
        try:
            from flopy.mf6.utils import MfGrdFile
            grb = MfGrdFile(str(grb_path))
            grid = grb.modelgrid
        except Exception as e:
            print(f"Warning: Could not load .grb file: {e}")

    # Fall back to loading full model from storage
    if grid is None and project.storage_path:
        try:
            from app.services.mesh import load_model_from_storage
            model = load_model_from_storage(
                str(project.id), project.storage_path
            )
            if model is not None:
                grid = model.modelgrid
        except Exception as e:
            print(f"Warning: Could not load model for grid geometry: {e}")

    if grid is None:
        return

    # Only extract polygons for non-structured grids
    if grid.grid_type == "structured":
        return

    try:
        # Determine cells per layer
        ncpl = grid.ncpl
        if isinstance(ncpl, np.ndarray):
            ncpl_val = int(ncpl[0])
        else:
            ncpl_val = int(ncpl)

        # Try to get polygons via FloPy's get_cell_vertices
        cell_polys = _get_cell_polys_from_grid(grid, ncpl_val)

        # If FloPy grid doesn't have vertices, try gridspec data
        if cell_polys is None and model is not None:
            cell_polys = _get_cell_polys_from_gridspec(model, ncpl_val)

        if cell_polys is None:
            print("Warning: No cell polygon data available for grid geometry")
            return

        # Compute extent from polygon data
        all_x, all_y = [], []
        for poly in cell_polys:
            for pt in poly:
                if len(pt) >= 2:
                    all_x.append(pt[0])
                    all_y.append(pt[1])
        if all_x and all_y:
            extent = [min(all_x), max(all_x), min(all_y), max(all_y)]
        else:
            extent = [0.0, 1.0, 0.0, 1.0]

        geometry = {
            "grid_type": grid.grid_type,
            "extent": extent,
            "nlay": int(grid.nlay),
            "ncpl": ncpl_val,
        }

        # For most models, grid x/y geometry is the same for all layers
        geometry["layers"] = {"0": {"polygons": cell_polys}}

        geom_json = json.dumps(geometry, default=_json_serialize)
        storage.upload_bytes(
            settings.minio_bucket_models,
            f"{output_prefix}/processed/grid_geometry.json",
            geom_json.encode("utf-8"),
            content_type="application/json",
        )
    except Exception as e:
        print(f"Warning: Grid geometry extraction failed: {e}")
        import traceback
        traceback.print_exc()


def _get_cell_polys_from_grid(grid, ncpl_val: int):
    """Try to extract cell polygons from FloPy modelgrid."""
    try:
        # Test if grid has vertex data
        _ = grid.xvertices
    except (TypeError, AttributeError):
        return None

    cell_polys = []
    for cellid in range(ncpl_val):
        try:
            verts = grid.get_cell_vertices(cellid)
            cell_polys.append(
                [[float(v[0]), float(v[1])] for v in verts]
            )
        except Exception:
            cell_polys.append([])
    return cell_polys


def _get_cell_polys_from_gridspec(model, ncpl_val: int):
    """Extract cell polygons from gridspec data attached by mesh.py."""
    gsf = getattr(model, "_gridspec_data", None)
    if not gsf or "cells" not in gsf or "vertices_2d" not in gsf:
        return None

    gsf_verts = gsf["vertices_2d"]  # list of (x, y), 0-indexed
    gsf_cells = gsf["cells"]  # list of dicts with vertex_indices (1-based)

    # Use only layer-1 cells for plan-view geometry, sorted by cell ID
    # to ensure polygon order matches head data node ordering
    layer1_cells = sorted(
        [c for c in gsf_cells if c["layer"] == 1],
        key=lambda c: c["id"],
    )
    if len(layer1_cells) != ncpl_val:
        # Fall back to first ncpl_val cells (sorted by id)
        layer1_cells = sorted(gsf_cells[:ncpl_val], key=lambda c: c["id"])

    cell_polys = []
    for cell in layer1_cells:
        vis = cell["vertex_indices"]  # 1-based indices into vertices_2d
        poly = []
        for vi in vis:
            if 0 < vi <= len(gsf_verts):
                vx, vy = gsf_verts[vi - 1]
                poly.append([float(vx), float(vy)])
        # Close the polygon
        if poly and poly[0] != poly[-1]:
            poly.append(poly[0])
        cell_polys.append(poly)

    return cell_polys
