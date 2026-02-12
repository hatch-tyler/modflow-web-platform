"""Live result processing during simulation execution.

This module provides background processing of simulation results while the
simulation is still running. MODFLOW writes to HDS/CBC files incrementally,
so we can read and process completed timesteps before the simulation finishes.
"""

import logging
import time
from pathlib import Path
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

from app.config import get_settings
from app.models.base import SessionLocal
from app.models.project import Project, Run, RunStatus
from app.services.redis_manager import get_sync_client
from app.services.slice_cache import cache_slice, cache_timestep_index, cache_live_budget
from celery_app import celery_app
from sqlalchemy import select

settings = get_settings()


@celery_app.task(bind=True, name="app.tasks.live_results.process_live_results")
def process_live_results(
    self,
    run_id: str,
    project_id: str,
    model_dir: str,
    model_type: str,
) -> dict:
    """
    Monitor HDS file during simulation and process timesteps as they complete.

    This task runs in parallel with the simulation, periodically checking for
    new timesteps in the HDS file. When new timesteps are found, they are
    extracted and cached in Redis for immediate access.

    Args:
        run_id: UUID of the run record
        project_id: UUID of the project
        model_dir: Path to the model directory (temp dir during simulation)
        model_type: MODFLOW model type (mf6, mf2005, etc.)

    Returns:
        Dict with processing statistics
    """
    import flopy.utils
    import numpy as np

    redis_client = get_sync_client()
    channel = f"simulation:{run_id}:output"

    def publish(message: str):
        """Publish message to Redis channel."""
        redis_client.publish(channel, message)

    model_path = Path(model_dir)
    is_unstructured = model_type == "mfusg"

    # If the run is already completed/failed (e.g. concurrency=1 and simulation
    # finished before this task started), bail out immediately.
    with SessionLocal() as db:
        run = db.execute(
            select(Run).where(Run.id == UUID(run_id))
        ).scalar_one_or_none()
        if run and run.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
            publish(f"[Live Results] Run already {run.status.value}, skipping live processing")
            return {"run_id": run_id, "timesteps_processed": 0, "skipped": True}

    # Find HDS file (supports .hds, .hed, .bhd extensions)
    hds_path = None
    hds_extensions = [".hds", ".HDS", ".hed", ".HED", ".bhd", ".BHD"]
    for ext in hds_extensions:
        candidates = list(model_path.glob(f"*{ext}"))
        if candidates:
            hds_path = candidates[0]
            break

    if not hds_path:
        # HDS file may not exist yet, wait and retry
        for _ in range(30):  # Wait up to 5 minutes
            time.sleep(10)
            for ext in hds_extensions:
                candidates = list(model_path.glob(f"*{ext}"))
                if candidates:
                    hds_path = candidates[0]
                    break
            if hds_path:
                break

    if not hds_path:
        return {"error": "HDS file not found", "timesteps_processed": 0}

    # Find CBC file (supports .cbc, .bud, .cbb extensions)
    cbc_path = None
    cbc_extensions = [".cbc", ".CBC", ".bud", ".BUD", ".cbb", ".CBB"]
    for ext in cbc_extensions:
        candidates = list(model_path.glob(f"*{ext}"))
        if candidates:
            cbc_path = candidates[0]
            break

    processed_timesteps = set()
    processed_budget_timesteps = set()
    total_processed = 0
    last_check_time = 0
    check_interval = 5  # seconds between checks
    global_min = None
    global_max = None
    live_budget_data = {"record_names": [], "periods": {}, "live": True}

    # Get project info for grid dimensions
    with SessionLocal() as db:
        project = db.execute(
            select(Project).where(Project.id == UUID(project_id))
        ).scalar_one_or_none()

        if not project:
            return {"error": "Project not found"}

    publish("[Live Results] Starting background result processing...")

    while True:
        # Check if simulation is still running
        with SessionLocal() as db:
            run = db.execute(
                select(Run).where(Run.id == UUID(run_id))
            ).scalar_one_or_none()

            if not run:
                break

            # Exit if simulation completed, failed, or cancelled
            if run.status in [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED]:
                publish(f"[Live Results] Simulation {run.status.value}, stopping live processing")
                break

        # Wait between checks
        current_time = time.time()
        if current_time - last_check_time < check_interval:
            time.sleep(1)
            continue
        last_check_time = current_time

        # Check if HDS file exists and has data
        if not hds_path.exists():
            continue

        hds = None
        try:
            # Try to read the HDS file (FloPy handles partial files)
            if is_unstructured:
                hds = flopy.utils.HeadUFile(str(hds_path))
            else:
                hds = flopy.utils.HeadFile(str(hds_path))

            kstpkper_list = hds.get_kstpkper()
            times = hds.get_times()

            # Find new timesteps to process
            new_timesteps = [ts for ts in kstpkper_list if ts not in processed_timesteps]

            if new_timesteps:
                publish(f"[Live Results] Found {len(new_timesteps)} new timestep(s) to process")

                for kstp, kper in new_timesteps:
                    try:
                        # Extract head data for all layers
                        data = hds.get_data(kstpkper=(kstp, kper))

                        if is_unstructured:
                            layers_data = data if isinstance(data, list) else [data]
                            nlay = len(layers_data)
                        else:
                            nlay = data.shape[0]
                            layers_data = [data[i] for i in range(nlay)]

                        # Process each layer
                        for layer_idx in range(nlay):
                            arr = np.asarray(layers_data[layer_idx])

                            # Mask dry/inactive cells
                            masked = np.where(
                                (np.abs(arr) > 1e20) | (arr < -880) | (np.isclose(arr, 999.0)),
                                np.nan,
                                arr,
                            )

                            # Update global min/max
                            valid = masked[~np.isnan(masked)]
                            if valid.size > 0:
                                layer_max = float(np.max(valid))
                                layer_min = float(np.min(valid))
                                if global_max is None or layer_max > global_max:
                                    global_max = layer_max
                                if global_min is None or layer_min < global_min:
                                    global_min = layer_min

                            # Convert to JSON-serializable format
                            if arr.ndim == 1:
                                flat = [None if (v != v) else float(v) for v in masked.tolist()]
                                data_list = [flat]
                                shape = [1, len(flat)]
                            else:
                                data_list = []
                                for row in masked.tolist():
                                    data_list.append([None if (v != v) else float(v) for v in row])
                                shape = list(arr.shape)

                            slice_data = {
                                "layer": layer_idx,
                                "kper": kper,
                                "kstp": kstp,
                                "shape": shape,
                                "data": data_list,
                            }

                            # Cache the slice
                            cache_slice(
                                project_id, run_id,
                                layer_idx, kper, kstp,
                                slice_data,
                                ttl=7200,  # 2 hours for live results
                            )

                        processed_timesteps.add((kstp, kper))
                        total_processed += 1

                    except Exception as e:
                        # Timestep might not be fully written yet
                        publish(f"[Live Results] Skipping timestep SP{kper+1}/TS{kstp+1}: {e}")
                        continue

                # Update timestep index in cache
                index_data = {
                    "kstpkper_list": [[int(ks), int(kp)] for ks, kp in sorted(processed_timesteps)],
                    "times": [float(t) for t in times[:len(processed_timesteps)]] if times else [],
                    "nlay": project.nlay or 1,
                    "grid_shape": [project.nlay or 1, project.nrow or 1, project.ncol or 1],
                    "is_unstructured": is_unstructured,
                    "min_head": global_min,
                    "max_head": global_max,
                    "live_processing": True,
                    "timesteps_available": len(processed_timesteps),
                }
                cache_timestep_index(project_id, run_id, index_data, ttl=7200)

                publish(f"[Live Results] Processed {total_processed} timestep(s), min={global_min:.2f}, max={global_max:.2f}" if global_min else f"[Live Results] Processed {total_processed} timestep(s)")

        except Exception as e:
            # File might be locked or incomplete, just continue
            logger.debug(f"[Live Results] HDS read error (file may be locked/incomplete): {e}")
        finally:
            if hds is not None:
                try:
                    hds.close()
                except Exception as e:
                    logger.debug(f"[Live Results] Error closing HDS file: {e}")

        # Poll CBC file for live budget data
        if cbc_path is None:
            # Re-check for CBC file (may appear after simulation starts writing)
            for ext in cbc_extensions:
                candidates = list(model_path.glob(f"*{ext}"))
                if candidates:
                    cbc_path = candidates[0]
                    break

        if cbc_path and cbc_path.exists():
            cbc = None
            try:
                if is_unstructured:
                    cbc = flopy.utils.CellBudgetFile(str(cbc_path), precision="double")
                else:
                    cbc = flopy.utils.CellBudgetFile(str(cbc_path))

                cbc_kstpkper = cbc.get_kstpkper()
                new_budget_ts = [ts for ts in cbc_kstpkper if ts not in processed_budget_timesteps]

                if new_budget_ts:
                    raw_names = cbc.get_unique_record_names()
                    record_names = []
                    for name in raw_names:
                        if isinstance(name, bytes):
                            record_names.append(name.decode("ascii", errors="ignore").strip())
                        else:
                            record_names.append(str(name).strip())

                    live_budget_data["record_names"] = record_names

                    for kstp, kper in new_budget_ts:
                        try:
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
                                total_in += comp_in
                                total_out += comp_out

                            discrepancy = total_in - total_out
                            percent_discrepancy = (
                                (discrepancy / ((total_in + total_out) / 2.0)) * 100.0
                                if (total_in + total_out) > 0
                                else 0.0
                            )

                            live_budget_data["periods"][period_key] = {
                                "kstp": int(kstp),
                                "kper": int(kper),
                                "in": period_in,
                                "out": period_out,
                                "total_in": total_in,
                                "total_out": total_out,
                                "discrepancy": discrepancy,
                                "percent_discrepancy": percent_discrepancy,
                            }

                            processed_budget_timesteps.add((kstp, kper))
                        except Exception:
                            # Timestep might not be fully written yet
                            continue

                    # Cap periods to prevent unbounded memory growth
                    # (all data is already cached incrementally in Redis)
                    if len(live_budget_data["periods"]) > 200:
                        sorted_keys = sorted(live_budget_data["periods"].keys())
                        for old_key in sorted_keys[:-200]:
                            del live_budget_data["periods"][old_key]

                    # Cache accumulated live budget
                    cache_live_budget(project_id, run_id, live_budget_data, ttl=7200)

            except Exception as e:
                # CBC file might be locked or partially written
                logger.debug(f"[Live Results] CBC read error (file may be locked/incomplete): {e}")
            finally:
                if cbc is not None:
                    try:
                        cbc.close()
                    except Exception as e:
                        logger.debug(f"[Live Results] Error closing CBC file: {e}")

        # Check more frequently during active simulation
        check_interval = 5 if total_processed < 10 else 10

    publish(f"[Live Results] Completed. Total timesteps processed: {total_processed}")

    return {
        "run_id": run_id,
        "timesteps_processed": total_processed,
        "min_head": global_min,
        "max_head": global_max,
    }


def start_live_processing(run_id: str, project_id: str, model_dir: str, model_type: str):
    """
    Start live result processing as a background task.

    Called from the simulation task after model files are downloaded but
    before the simulation starts.
    """
    return process_live_results.delay(run_id, project_id, model_dir, model_type)
