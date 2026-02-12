"""Results API endpoints for serving processed simulation output."""

import asyncio
import csv
import io
import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.base import get_db
from app.models.project import Project, Run, RunStatus
from app.services.head_extractor import (
    extract_head_slice_on_demand,
    get_head_statistics,
    get_timestep_index,
)
from app.services.cache_service import get_cache_service
from app.services.hds_streaming import (
    get_head_slice_streaming,
    get_timeseries_streaming,
)
from app.services.slice_cache import (
    get_cached_slice,
    get_cached_timestep_index,
    get_cached_live_budget,
)
from app.services.storage import get_storage_service

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/projects/{project_id}/runs/{run_id}/results",
    tags=["results"],
)
settings = get_settings()

# Internal cell-to-cell flow records excluded from mass balance totals
_INTERNAL_FLOW_RECORDS = {
    'FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
    'FLOW_RIGHT_FACE', 'FLOW_FRONT_FACE', 'FLOW_LOWER_FACE',
    'FLOW JA FACE', 'FLOW_JA_FACE', 'FLOW-JA-FACE',
    'DATA-SPDIS', 'DATA-SAT', 'DATA-STOSS', 'DATA-STOSY',
}


def _is_internal_flow(name: str) -> bool:
    """Check if a budget record name is an internal cell-to-cell flow."""
    return name.upper() in _INTERNAL_FLOW_RECORDS


_DISCREPANCY_RE = re.compile(
    r"PERCENT\s+DISCREPANCY\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def _extract_listing_discrepancies(run, storage) -> list[float]:
    """Extract per-stress-period percent discrepancy values from listing file in MinIO."""
    output_objects = storage.list_objects(
        settings.minio_bucket_models,
        prefix=run.results_path,
        recursive=True,
    )
    lst_obj = None
    for obj_name in output_objects:
        if obj_name.lower().endswith(".lst") or obj_name.lower().endswith(".list"):
            lst_obj = obj_name
            break

    if not lst_obj:
        return []

    lst_data = storage.download_file(settings.minio_bucket_models, lst_obj)
    discrepancies: list[float] = []
    for line in lst_data.decode("utf-8", errors="replace").splitlines():
        if "PERCENT DISCREPANCY" in line.upper():
            m = _DISCREPANCY_RE.findall(line)
            if m:
                discrepancies.append(abs(float(m[0])))

    return discrepancies


async def _get_run_or_404(
    project_id: UUID, run_id: UUID, db: AsyncSession
) -> tuple[Run, Project]:
    """Validate that a completed run exists and return it with its project."""
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found in project {project_id}",
        )

    if run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run is not completed (status: {run.status.value})",
        )

    if not run.results_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results available for this run",
        )

    stmt_proj = select(Project).where(Project.id == project_id)
    result_proj = await db.execute(stmt_proj)
    project = result_proj.scalar_one_or_none()

    return run, project


async def _get_run_for_live_results(
    project_id: UUID, run_id: UUID, db: AsyncSession
) -> tuple[Run, Project]:
    """Get run for live results - allows running or completed status."""
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found in project {project_id}",
        )

    # Allow running or completed status for live results
    if run.status not in [RunStatus.RUNNING, RunStatus.COMPLETED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run is not running or completed (status: {run.status.value})",
        )

    stmt_proj = select(Project).where(Project.id == project_id)
    result_proj = await db.execute(stmt_proj)
    project = result_proj.scalar_one_or_none()

    return run, project


@router.get("/summary")
async def get_results_summary(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get the results summary for a completed run.

    Returns heads summary, budget overview, convergence info, and metadata.
    """
    run, _ = await _get_run_or_404(project_id, run_id, db)
    storage = get_storage_service()

    obj_name = f"{run.results_path}/results_summary.json"
    if not storage.object_exists(settings.minio_bucket_models, obj_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results summary not yet available. Post-processing may still be running.",
        )

    data = storage.download_file(settings.minio_bucket_models, obj_name)
    summary = json.loads(data)

    # Backfill percent_discrepancies from listing file if missing
    convergence = summary.get("convergence", {})
    if not convergence.get("percent_discrepancies"):
        try:
            discrepancies = _extract_listing_discrepancies(run, storage)
            if discrepancies:
                convergence["percent_discrepancies"] = discrepancies
                summary["convergence"] = convergence
                # Persist enriched summary back to MinIO for future requests
                enriched = json.dumps(summary).encode("utf-8")
                storage.upload_bytes(
                    settings.minio_bucket_models, obj_name,
                    enriched, content_type="application/json",
                )
        except Exception as e:
            logger.warning(f"Failed to backfill listing discrepancies: {e}")

    return summary


@router.get("/heads")
async def get_head_slice(
    project_id: UUID,
    run_id: UUID,
    layer: int = Query(0, ge=0),
    kper: int = Query(0, ge=0),
    kstp: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get a 2D head array slice for a specific layer, stress period, and time step.

    Returns the shape and data array for the requested slice.

    This endpoint supports multiple extraction strategies in priority order:
    1. Pre-processed JSON file (fastest)
    2. HTTP Range streaming from HDS file (efficient for large files)
    3. Full HDS file download with FloPy (fallback)
    """
    total_start = time.time()

    t0 = time.time()
    run, project = await _get_run_or_404(project_id, run_id, db)
    db_time = time.time() - t0
    logger.info(f"[PERF] DB lookup took {db_time:.3f}s")

    storage = get_storage_service()

    # Fast path: check for pre-processed file first
    obj_name = (
        f"{run.results_path}/processed/"
        f"heads_L{layer}_SP{kper}_TS{kstp}.json"
    )

    t0 = time.time()
    exists = storage.object_exists(settings.minio_bucket_models, obj_name)
    exists_time = time.time() - t0

    if exists:
        t0 = time.time()
        data = storage.download_file(settings.minio_bucket_models, obj_name)
        download_time = time.time() - t0
        t0 = time.time()
        result = json.loads(data)
        parse_time = time.time() - t0
        total_time = time.time() - total_start
        logger.info(f"[PERF] Pre-processed file: exists_check={exists_time:.3f}s, download={download_time:.3f}s, parse={parse_time:.3f}s, total={total_time:.3f}s")
        return result

    logger.info(f"[PERF] No pre-processed file (check took {exists_time:.3f}s), using streaming")

    model_type = project.model_type.value if project and project.model_type else "mf6"

    # Try streaming extraction first (uses HTTP Range requests, much faster)
    # Only for structured grids - unstructured grids need special handling
    if model_type != "mfusg":
        t0 = time.time()
        streaming_result = await asyncio.to_thread(
            get_head_slice_streaming,
            project_id=str(project_id),
            run_id=str(run_id),
            results_path=run.results_path,
            layer=layer,
            kper=kper,
            kstp=kstp,
        )
        streaming_time = time.time() - t0

        if "error" not in streaming_result:
            total_time = time.time() - total_start
            logger.info(f"[PERF] Streaming extraction: {streaming_time:.3f}s, total={total_time:.3f}s")
            return streaming_result
        logger.warning(f"[PERF] Streaming failed ({streaming_time:.3f}s): {streaming_result.get('error')}, falling back to FloPy")

    # Fallback: extract directly from HDS file (downloads full file)
    t0 = time.time()
    result = await asyncio.to_thread(
        extract_head_slice_on_demand,
        project_id=str(project_id),
        run_id=str(run_id),
        results_path=run.results_path,
        model_type=model_type,
        layer=layer,
        kper=kper,
        kstp=kstp,
    )
    fallback_time = time.time() - t0
    total_time = time.time() - total_start
    logger.info(f"[PERF] FloPy fallback: {fallback_time:.3f}s, total={total_time:.3f}s")

    if "error" in result:
        # Provide helpful error with available timesteps if applicable
        if "available_timesteps" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "message": result["error"],
                    "available_timesteps": result["available_timesteps"],
                },
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["error"],
        )

    return result


@router.get("/heads/available")
async def get_available_timesteps(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get list of available timesteps and grid information.

    Returns the timestep index which includes:
    - kstpkper_list: All available (kstp, kper) pairs
    - times: Simulation times corresponding to each timestep
    - nlay: Number of layers
    - grid_shape: Grid dimensions

    This is cached for fast subsequent access.
    """
    run, project = await _get_run_or_404(project_id, run_id, db)
    model_type = project.model_type.value if project and project.model_type else "mf6"

    result = get_timestep_index(
        project_id=str(project_id),
        run_id=str(run_id),
        results_path=run.results_path,
        model_type=model_type,
    )

    if "error" in result and not result.get("kstpkper_list"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["error"],
        )

    return result


@router.get("/heads/statistics")
async def get_head_stats(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get head statistics (min/max) for color scaling.

    Samples multiple timesteps to determine global min/max head values
    without loading all data into memory.
    """
    run, project = await _get_run_or_404(project_id, run_id, db)
    model_type = project.model_type.value if project and project.model_type else "mf6"

    result = get_head_statistics(
        project_id=str(project_id),
        run_id=str(run_id),
        results_path=run.results_path,
        model_type=model_type,
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["error"],
        )

    return result


# =============================================================================
# Live Results Endpoints - Available during simulation
# =============================================================================

@router.get("/live/available")
async def get_live_timesteps(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get available timesteps from live processing during simulation.

    Returns cached timestep index from live result processing.
    Works while simulation is running or after completion.
    """
    run, project = await _get_run_for_live_results(project_id, run_id, db)

    # Check cache for live results
    cached_index = get_cached_timestep_index(str(project_id), str(run_id))

    if cached_index:
        return {
            **cached_index,
            "simulation_status": run.status.value,
            "live": True,
        }

    # No live results yet
    return {
        "kstpkper_list": [],
        "times": [],
        "nlay": project.nlay or 1,
        "grid_shape": [project.nlay or 1, project.nrow or 1, project.ncol or 1],
        "is_unstructured": (project.model_type and project.model_type.value == "mfusg") or (hasattr(project, 'grid_type') and project.grid_type in ("vertex", "unstructured")),
        "simulation_status": run.status.value,
        "live": True,
        "message": "Live results not yet available. Processing may not have started.",
    }


@router.get("/live/heads")
async def get_live_head_slice(
    project_id: UUID,
    run_id: UUID,
    layer: int = Query(0, ge=0),
    kper: int = Query(0, ge=0),
    kstp: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get a live head slice from cache during simulation.

    Returns cached head data processed by live result processing.
    Works while simulation is running or after completion.
    """
    run, project = await _get_run_for_live_results(project_id, run_id, db)

    # Check cache for this slice
    cached_slice = get_cached_slice(str(project_id), str(run_id), layer, kper, kstp)

    if cached_slice:
        return {
            **cached_slice,
            "simulation_status": run.status.value,
            "live": True,
        }

    # Not cached - check if simulation is completed and we should fall back to on-demand
    if run.status == RunStatus.COMPLETED and run.results_path:
        model_type = project.model_type.value if project and project.model_type else "mf6"
        result = extract_head_slice_on_demand(
            project_id=str(project_id),
            run_id=str(run_id),
            results_path=run.results_path,
            model_type=model_type,
            layer=layer,
            kper=kper,
            kstp=kstp,
        )
        if "error" not in result:
            return {
                **result,
                "simulation_status": run.status.value,
                "live": False,
            }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Head slice not available yet for layer={layer}, kper={kper}, kstp={kstp}. "
               f"Simulation status: {run.status.value}",
    )


@router.get("/live/summary")
async def get_live_summary(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get a live results summary during simulation.

    Returns partial summary with whatever data is available from live processing.
    """
    run, project = await _get_run_for_live_results(project_id, run_id, db)

    # Check cache for live timestep index
    cached_index = get_cached_timestep_index(str(project_id), str(run_id))

    if cached_index:
        # Retrieve live budget data if available
        live_budget = get_cached_live_budget(str(project_id), str(run_id)) or {}

        return {
            "heads_summary": {
                "nstp_total": cached_index.get("timesteps_available", len(cached_index.get("kstpkper_list", []))),
                "kstpkper_list": cached_index.get("kstpkper_list", []),
                "times": cached_index.get("times", []),
                "max_head": cached_index.get("max_head"),
                "min_head": cached_index.get("min_head"),
                "live_processing": True,
            },
            "budget": live_budget,
            "convergence": {"converged": None, "mass_balance_error_pct": None, "max_head_changes": []},
            "metadata": {
                "model_type": project.model_type.value if project.model_type else "mf6",
                "grid_type": project.grid_type or ("unstructured" if cached_index.get("is_unstructured") else "structured"),
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
            "simulation_status": run.status.value,
            "live": True,
        }

    # No live head results yet - still check for live budget
    live_budget = get_cached_live_budget(str(project_id), str(run_id)) or {}

    return {
        "heads_summary": {
            "nstp_total": 0,
            "kstpkper_list": [],
            "times": [],
            "max_head": None,
            "min_head": None,
            "live_processing": True,
        },
        "budget": live_budget,
        "convergence": {"converged": None, "mass_balance_error_pct": None, "max_head_changes": []},
        "metadata": {
            "model_type": project.model_type.value if project.model_type else "mf6",
            "grid_type": project.grid_type or "structured",
            "nlay": project.nlay or 1,
            "nrow": project.nrow or 1,
            "ncol": project.ncol or 1,
            "nper": project.nper or 1,
        },
        "simulation_status": run.status.value,
        "live": True,
        "message": "Waiting for live results...",
    }


def _compute_full_budget_from_cbc(run: Run, project: Project, storage) -> Optional[dict]:
    """
    Compute full water budget by downloading and processing the CBC file.

    Finds CBC file in MinIO (.cbc/.bud/.cbb), downloads to temp dir,
    opens with FloPy CellBudgetFile, and processes all timesteps.

    Returns complete budget dict or None if CBC not found.
    """
    import flopy.utils

    # Find CBC file in results
    output_objects = storage.list_objects(
        settings.minio_bucket_models,
        prefix=run.results_path,
        recursive=True,
    )

    cbc_obj = None
    for obj_name in output_objects:
        lower_name = obj_name.lower()
        if lower_name.endswith(".cbc") or lower_name.endswith(".bud") or lower_name.endswith(".cbb"):
            cbc_obj = obj_name
            break

    if not cbc_obj:
        return None

    model_type = project.model_type.value if project.model_type else "mf6"

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "output.cbc"
        file_data = storage.download_file(settings.minio_bucket_models, cbc_obj)
        local_path.write_bytes(file_data)

        cbc = None
        try:
            if model_type == "mfusg":
                cbc = flopy.utils.CellBudgetFile(str(local_path), precision="double")
            else:
                cbc = flopy.utils.CellBudgetFile(str(local_path))

            try:
                raw_names = cbc.get_unique_record_names()
            except Exception:
                return None

            record_names = []
            for name in raw_names:
                if isinstance(name, bytes):
                    record_names.append(name.decode("ascii", errors="ignore").strip())
                else:
                    record_names.append(str(name).strip())

            kstpkper_list = cbc.get_kstpkper()
            periods = {}

            for kstp, kper in kstpkper_list:
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

            return {
                "record_names": record_names,
                "periods": periods,
                "quick_mode": False,
                "total_timesteps": len(kstpkper_list),
                "timesteps_processed": len(kstpkper_list),
            }
        except Exception:
            return None
        finally:
            if cbc is not None:
                try:
                    cbc.close()
                except Exception:
                    pass


@router.get("/grid-info")
async def get_grid_info(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get structured grid info for building cell polygons on the frontend.

    Returns delr, delc, nrow, ncol, xoff, yoff, angrot, epsg, length_unit.
    Used by the frontend to construct structured grid map views.
    """
    run, project = await _get_run_or_404(project_id, run_id, db)

    return {
        "delr": project.delr,
        "delc": project.delc,
        "nrow": project.nrow or 0,
        "ncol": project.ncol or 0,
        "nlay": project.nlay or 1,
        "xoff": project.xoff or 0.0,
        "yoff": project.yoff or 0.0,
        "angrot": project.angrot or 0.0,
        "epsg": project.epsg,
        "length_unit": project.length_unit,
    }


@router.get("/budget")
async def get_budget(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get the full water budget breakdown.

    Returns record names and per-period budget data with IN/OUT totals.
    If only quick-mode budget is available, computes full budget on-demand from CBC.
    """
    run, project = await _get_run_or_404(project_id, run_id, db)
    storage = get_storage_service()

    obj_name = f"{run.results_path}/processed/budget.json"
    if storage.object_exists(settings.minio_bucket_models, obj_name):
        data = json.loads(storage.download_file(settings.minio_bucket_models, obj_name))
        # If quick_mode, compute full budget on-demand
        if data.get("quick_mode"):
            full_budget = _compute_full_budget_from_cbc(run, project, storage)
            if full_budget:
                return full_budget
        return data

    # No budget.json at all - try computing from CBC directly
    full_budget = _compute_full_budget_from_cbc(run, project, storage)
    if full_budget:
        return full_budget

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Budget data not available",
    )


@router.get("/timeseries")
async def get_timeseries(
    project_id: UUID,
    run_id: UUID,
    row: Optional[int] = Query(None, ge=0),
    col: Optional[int] = Query(None, ge=0),
    layer: int = Query(0, ge=0),
    node: Optional[int] = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get a head time series for a specific cell location.

    For structured grids (MF6, MF2005, MF-NWT): use layer, row, col.
    For unstructured grids (MF-USG): use node index (0-based).

    Results are cached in Redis for fast subsequent requests.
    Uses HTTP Range requests to minimize data transfer for large HDS files.
    """
    import flopy.utils

    run, project = await _get_run_or_404(project_id, run_id, db)

    model_type = project.model_type.value if project and project.model_type else "mf6"
    is_unstructured = model_type == "mfusg"

    if is_unstructured:
        if node is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parameter 'node' is required for unstructured grid models",
            )
    else:
        if row is None or col is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parameters 'row' and 'col' are required for structured grid models",
            )

    cache = get_cache_service()
    project_id_str = str(project_id)
    run_id_str = str(run_id)

    # Check cache first
    cached = cache.get_timeseries(
        project_id_str, run_id_str, layer,
        row=row, col=col, node=node
    )
    if cached:
        return cached

    # For structured grids, try streaming extraction first (much faster for large files)
    if not is_unstructured and run.results_path:
        streaming_result = get_timeseries_streaming(
            project_id_str, run_id_str, run.results_path, layer, row, col
        )
        if "error" not in streaming_result:
            return streaming_result

    # Fall back to full file download for unstructured grids or if streaming fails
    storage = get_storage_service()

    # Find head file in results (supports .hds and .hed extensions)
    output_objects = storage.list_objects(
        settings.minio_bucket_models,
        prefix=run.results_path,
        recursive=True,
    )

    hds_obj = None
    for obj_name in output_objects:
        lower_name = obj_name.lower()
        if lower_name.endswith(".hds") or lower_name.endswith(".hed"):
            hds_obj = obj_name
            break

    if not hds_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Head file (.hds or .hed) not found in results",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "output.hds"
        file_data = storage.download_file(settings.minio_bucket_models, hds_obj)
        local_path.write_bytes(file_data)

        hds = None
        try:
            if is_unstructured:
                hds = flopy.utils.HeadUFile(str(local_path))
                # For unstructured grids, extract head at node index
                # across all time steps manually
                kstpkper_list = hds.get_kstpkper()
                all_times = hds.get_times()
                heads_raw = []
                for kstp_kp in kstpkper_list:
                    data = hds.get_data(kstpkper=kstp_kp)
                    # HeadUFile returns list of arrays per layer
                    layer_data = data if isinstance(data, list) else [data]
                    if layer < len(layer_data):
                        arr = np.asarray(layer_data[layer]).flatten()
                        if node < len(arr):
                            heads_raw.append(float(arr[node]))
                        else:
                            heads_raw.append(float("nan"))
                    else:
                        heads_raw.append(float("nan"))
                times = list(all_times)
            else:
                hds = flopy.utils.HeadFile(str(local_path))
                ts = hds.get_ts((layer, row, col))
                times = ts[:, 0].tolist()
                heads_raw = ts[:, 1].tolist()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error reading time series: {e}",
            )
        finally:
            if hds is not None:
                try:
                    hds.close()
                except Exception:
                    pass

        # Replace dry/inactive values with None
        # MF6 uses 1e30, MF2005/NWT use -888.0 (HDRY), 999.0 (HNOFLO)
        heads = [
            None if (abs(v) > 1e20 or v < -880 or abs(v - 999.0) < 0.01 or v != v)
            else v
            for v in heads_raw
        ]

    response: dict = {
        "layer": layer,
        "times": times,
        "heads": heads,
    }
    if is_unstructured:
        response["node"] = node
    else:
        response["row"] = row
        response["col"] = col

    # Cache for future requests
    cache.set_timeseries(
        project_id_str, run_id_str, layer, response,
        row=row, col=col, node=node
    )

    return response


@router.get("/grid-geometry")
async def get_grid_geometry(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Return pre-extracted grid cell polygons for map rendering.

    Loads grid_geometry.json from MinIO (created during post-processing).
    """
    run, _ = await _get_run_or_404(project_id, run_id, db)
    storage = get_storage_service()

    obj_name = f"{run.results_path}/processed/grid_geometry.json"
    if not storage.object_exists(settings.minio_bucket_models, obj_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Grid geometry not available",
        )

    data = storage.download_file(settings.minio_bucket_models, obj_name)
    return json.loads(data)


@router.get("/heads/render")
async def render_head_contour(
    project_id: UUID,
    run_id: UUID,
    layer: int = Query(0, ge=0),
    kper: int = Query(0, ge=0),
    kstp: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Render a head contour map as a PNG image using matplotlib.

    Used for unstructured and vertex grids where a standard heatmap
    cannot represent the cell geometry. Uses pre-extracted grid
    polygon data with matplotlib PatchCollection.

    Supports on-demand head slice extraction if pre-processed file doesn't exist.
    """
    run, project = await _get_run_or_404(project_id, run_id, db)
    storage = get_storage_service()

    # Load grid geometry
    geom_obj = f"{run.results_path}/processed/grid_geometry.json"
    if not storage.object_exists(settings.minio_bucket_models, geom_obj):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Grid geometry not available. Re-run post-processing.",
        )

    geom_data = json.loads(
        storage.download_file(settings.minio_bucket_models, geom_obj)
    )

    # Load head data - try pre-processed first, then on-demand
    head_obj = (
        f"{run.results_path}/processed/"
        f"heads_L{layer}_SP{kper}_TS{kstp}.json"
    )
    if storage.object_exists(settings.minio_bucket_models, head_obj):
        head_data = json.loads(
            storage.download_file(settings.minio_bucket_models, head_obj)
        )
    else:
        # On-demand extraction
        model_type = project.model_type.value if project and project.model_type else "mf6"
        head_data = extract_head_slice_on_demand(
            project_id=str(project_id),
            run_id=str(run_id),
            results_path=run.results_path,
            model_type=model_type,
            layer=layer,
            kper=kper,
            kstp=kstp,
        )
        if "error" in head_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=head_data["error"],
            )

    # Load summary for global min/max color scale
    summary_obj = f"{run.results_path}/results_summary.json"
    vmin, vmax = None, None
    if storage.object_exists(settings.minio_bucket_models, summary_obj):
        summary = json.loads(
            storage.download_file(settings.minio_bucket_models, summary_obj)
        )
        hs = summary.get("heads_summary", {})
        vmin = hs.get("min_head")
        vmax = hs.get("max_head")

    img_bytes = _render_polygon_contour(geom_data, head_data, layer, vmin, vmax)

    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/export/heads")
async def export_heads_csv(
    project_id: UUID,
    run_id: UUID,
    layer: int = Query(0, ge=0),
    kper: int = Query(0, ge=0),
    kstp: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export head slice as CSV. Supports on-demand extraction."""
    run, project = await _get_run_or_404(project_id, run_id, db)
    storage = get_storage_service()

    # Try pre-processed file first
    obj_name = (
        f"{run.results_path}/processed/"
        f"heads_L{layer}_SP{kper}_TS{kstp}.json"
    )
    if storage.object_exists(settings.minio_bucket_models, obj_name):
        data = json.loads(storage.download_file(settings.minio_bucket_models, obj_name))
    else:
        # On-demand extraction
        model_type = project.model_type.value if project and project.model_type else "mf6"
        data = extract_head_slice_on_demand(
            project_id=str(project_id),
            run_id=str(run_id),
            results_path=run.results_path,
            model_type=model_type,
            layer=layer,
            kper=kper,
            kstp=kstp,
        )
        if "error" in data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=data["error"],
            )
    raw = data.get("data", [])

    model_type = project.model_type.value if project and project.model_type else "mf6"
    is_unstructured = model_type == "mfusg"

    buf = io.StringIO()
    writer = csv.writer(buf)

    if is_unstructured:
        writer.writerow(["cell_id", "head"])
        values = []
        for row in raw:
            if isinstance(row, list):
                values.extend(row)
            else:
                values.append(row)
        for i, v in enumerate(values):
            if v is None or (isinstance(v, (int, float)) and (abs(v) > 1e20 or v < -880)):
                writer.writerow([i + 1, ""])
            else:
                writer.writerow([i + 1, v])
    else:
        ncol = len(raw[0]) if raw else 0
        header = ["row\\col"] + [str(c) for c in range(ncol)]
        writer.writerow(header)
        for r_idx, row in enumerate(raw):
            out = [str(r_idx)]
            for v in row:
                if v is None or (isinstance(v, (int, float)) and (abs(v) > 1e20 or v < -880)):
                    out.append("")
                else:
                    out.append(str(v))
            writer.writerow(out)

    filename = f"heads_L{layer}_SP{kper}_TS{kstp}.csv"
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/drawdown")
async def export_drawdown_csv(
    project_id: UUID,
    run_id: UUID,
    layer: int = Query(0, ge=0),
    kper: int = Query(0, ge=0),
    kstp: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export drawdown (initial - current) as CSV. Supports on-demand extraction."""
    run, project = await _get_run_or_404(project_id, run_id, db)
    storage = get_storage_service()
    model_type = project.model_type.value if project and project.model_type else "mf6"

    # Helper to get slice data (pre-processed or on-demand)
    def get_slice_data(l: int, kp: int, ks: int) -> dict:
        obj = f"{run.results_path}/processed/heads_L{l}_SP{kp}_TS{ks}.json"
        if storage.object_exists(settings.minio_bucket_models, obj):
            return json.loads(storage.download_file(settings.minio_bucket_models, obj))
        # On-demand extraction
        return extract_head_slice_on_demand(
            project_id=str(project_id),
            run_id=str(run_id),
            results_path=run.results_path,
            model_type=model_type,
            layer=l,
            kper=kp,
            kstp=ks,
        )

    # Load initial slice
    init_data = get_slice_data(layer, 0, 0)
    if "error" in init_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Initial head slice not found",
        )

    # Load requested slice
    cur_data = get_slice_data(layer, kper, kstp)
    if "error" in cur_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Head slice not found for layer={layer}, kper={kper}, kstp={kstp}",
        )

    init_raw = init_data.get("data", [])
    cur_raw = cur_data.get("data", [])

    def _is_invalid(v):
        return v is None or (isinstance(v, (int, float)) and (abs(v) > 1e20 or v < -880))

    # Flatten both for uniform handling
    def _flatten(raw):
        vals = []
        for row in raw:
            if isinstance(row, list):
                vals.extend(row)
            else:
                vals.append(row)
        return vals

    model_type = project.model_type.value if project and project.model_type else "mf6"
    is_unstructured = model_type == "mfusg"

    buf = io.StringIO()
    writer = csv.writer(buf)

    if is_unstructured:
        writer.writerow(["cell_id", "drawdown"])
        init_vals = _flatten(init_raw)
        cur_vals = _flatten(cur_raw)
        for i in range(len(init_vals)):
            iv = init_vals[i] if i < len(init_vals) else None
            cv = cur_vals[i] if i < len(cur_vals) else None
            if _is_invalid(iv) or _is_invalid(cv):
                writer.writerow([i + 1, ""])
            else:
                writer.writerow([i + 1, iv - cv])
    else:
        ncol = len(init_raw[0]) if init_raw else 0
        header = ["row\\col"] + [str(c) for c in range(ncol)]
        writer.writerow(header)
        for r_idx in range(len(init_raw)):
            out = [str(r_idx)]
            for c_idx in range(ncol):
                iv = init_raw[r_idx][c_idx] if r_idx < len(init_raw) and c_idx < len(init_raw[r_idx]) else None
                cv = cur_raw[r_idx][c_idx] if r_idx < len(cur_raw) and c_idx < len(cur_raw[r_idx]) else None
                if _is_invalid(iv) or _is_invalid(cv):
                    out.append("")
                else:
                    out.append(str(iv - cv))
            writer.writerow(out)

    filename = f"drawdown_L{layer}_SP{kper}_TS{kstp}.csv"
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/budget")
async def export_budget_csv(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export full water budget as flat CSV. Computes full budget on-demand if needed."""
    run, project = await _get_run_or_404(project_id, run_id, db)
    storage = get_storage_service()

    budget = None
    obj_name = f"{run.results_path}/processed/budget.json"
    if storage.object_exists(settings.minio_bucket_models, obj_name):
        budget = json.loads(storage.download_file(settings.minio_bucket_models, obj_name))
        # If quick_mode, compute full budget for complete CSV export
        if budget.get("quick_mode"):
            full_budget = _compute_full_budget_from_cbc(run, project, storage)
            if full_budget:
                budget = full_budget

    if not budget:
        # No budget.json - try computing from CBC directly
        budget = _compute_full_budget_from_cbc(run, project, storage)

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget data not available",
        )

    periods = budget.get("periods", {})

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["kper", "kstp", "component", "in", "out"])

    for _key, period in periods.items():
        kp = period.get("kper", 0)
        ks = period.get("kstp", 0)
        in_flows = period.get("in", {})
        out_flows = period.get("out", {})
        components = sorted(set(list(in_flows.keys()) + list(out_flows.keys())))
        for comp in components:
            writer.writerow([
                kp, ks, comp,
                in_flows.get(comp, 0),
                out_flows.get(comp, 0),
            ])
        writer.writerow([kp, ks, "TOTAL", period.get("total_in", 0), period.get("total_out", 0)])
        writer.writerow([kp, ks, "DISCREPANCY", period.get("discrepancy", 0), period.get("percent_discrepancy", 0)])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="budget.csv"'},
    )


@router.get("/export/timeseries")
async def export_timeseries_csv(
    project_id: UUID,
    run_id: UUID,
    row: Optional[int] = Query(None, ge=0),
    col: Optional[int] = Query(None, ge=0),
    layer: int = Query(0, ge=0),
    node: Optional[int] = Query(None, ge=0),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export head time series for a specific cell as CSV."""
    import flopy.utils

    run, project = await _get_run_or_404(project_id, run_id, db)
    storage = get_storage_service()

    model_type = project.model_type.value if project and project.model_type else "mf6"
    is_unstructured = model_type == "mfusg"

    if is_unstructured:
        if node is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parameter 'node' is required for unstructured grid models",
            )
    else:
        if row is None or col is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parameters 'row' and 'col' are required for structured grid models",
            )

    output_objects = storage.list_objects(
        settings.minio_bucket_models,
        prefix=run.results_path,
        recursive=True,
    )

    hds_obj = None
    for obj_name in output_objects:
        lower_name = obj_name.lower()
        if lower_name.endswith(".hds") or lower_name.endswith(".hed"):
            hds_obj = obj_name
            break

    if not hds_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Head file (.hds or .hed) not found in results",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "output.hds"
        file_data = storage.download_file(settings.minio_bucket_models, hds_obj)
        local_path.write_bytes(file_data)

        hds = None
        try:
            if is_unstructured:
                hds = flopy.utils.HeadUFile(str(local_path))
                kstpkper_list = hds.get_kstpkper()
                all_times = hds.get_times()
                heads_raw = []
                for kstp_kp in kstpkper_list:
                    data = hds.get_data(kstpkper=kstp_kp)
                    layer_data = data if isinstance(data, list) else [data]
                    if layer < len(layer_data):
                        arr = np.asarray(layer_data[layer]).flatten()
                        if node < len(arr):
                            heads_raw.append(float(arr[node]))
                        else:
                            heads_raw.append(float("nan"))
                    else:
                        heads_raw.append(float("nan"))
                times = list(all_times)
            else:
                hds = flopy.utils.HeadFile(str(local_path))
                ts = hds.get_ts((layer, row, col))
                times = ts[:, 0].tolist()
                heads_raw = ts[:, 1].tolist()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error reading time series: {e}",
            )
        finally:
            if hds is not None:
                try:
                    hds.close()
                except Exception:
                    pass

        heads = [
            None if (abs(v) > 1e20 or v < -880 or abs(v - 999.0) < 0.01 or v != v)
            else v
            for v in heads_raw
        ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["time", "head"])
    for t, h in zip(times, heads):
        writer.writerow([t, h if h is not None else ""])

    if is_unstructured:
        filename = f"timeseries_L{layer}_N{node}.csv"
    else:
        filename = f"timeseries_L{layer}_R{row}_C{col}.csv"

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _render_polygon_contour(
    geom_data: dict,
    head_data: dict,
    layer: int,
    vmin: float | None,
    vmax: float | None,
) -> bytes:
    """
    Render head values on grid polygons using matplotlib.

    Args:
        geom_data: Grid geometry with cell polygon vertices
        head_data: Head slice with data array
        layer: Layer index for polygon lookup
        vmin: Global minimum head for color scale
        vmax: Global maximum head for color scale

    Returns:
        PNG image bytes
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    # Get polygons for the requested layer
    layer_key = str(layer)
    layers = geom_data.get("layers", {})
    # Fall back to layer 0 if this layer's geometry isn't stored separately
    if layer_key not in layers:
        layer_key = "0"
    if layer_key not in layers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No grid geometry for this layer",
        )

    cell_polys = layers[layer_key]["polygons"]

    # Flatten head data to 1D array
    raw_data = head_data.get("data", [])
    values = []
    for row in raw_data:
        if isinstance(row, list):
            values.extend(row)
        else:
            values.append(row)

    # Build patches for cells with valid head values
    patches = []
    valid_values = []
    for i, poly_verts in enumerate(cell_polys):
        if not poly_verts:
            continue
        if i >= len(values):
            continue
        val = values[i]
        if val is None:
            continue
        # Skip dry/inactive sentinel values
        if abs(val) > 1e20 or val < -880:
            continue
        patches.append(Polygon(poly_verts, closed=True))
        valid_values.append(val)

    if not patches:
        # Return a blank image with "no data" text
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.text(0.5, 0.5, "No valid head data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#94a3b8")
        ax.set_facecolor("#e2e8f0")
        ax.set_xticks([])
        ax.set_yticks([])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_facecolor("#e2e8f0")

    pc = PatchCollection(
        patches,
        cmap="viridis",
        edgecolor="face",
        linewidth=0.3,
    )
    pc.set_array(np.array(valid_values))
    if vmin is not None and vmax is not None:
        pc.set_clim(vmin, vmax)

    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    cbar = plt.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Head (m)")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
