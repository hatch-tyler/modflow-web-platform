"""PEST++ parameter estimation API endpoints."""

import csv
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.base import get_db
from app.models.project import Project, Run, RunStatus, RunType
from app.services.storage import get_storage_service

router = APIRouter(prefix="/projects/{project_id}/pest", tags=["pest"])
settings = get_settings()


class PestParameterConfig(BaseModel):
    """Configuration for a single PEST parameter.

    Supports three parameter types:
    - Array-based: standard layer parameters (package_type='array')
    - List-based: single multiplier for package (package_type='list')
    - Pilot points: spatial heterogeneity (approach='pilotpoints')
    """
    property: str
    layer: Optional[int] = None  # None for list-based parameters
    approach: str = "multiplier"  # 'multiplier' or 'pilotpoints'
    pp_space: Optional[int] = None  # Pilot point spacing (cells)
    variogram: Optional[dict] = None  # Kriging variogram config
    initial_value: float = 1.0
    lower_bound: float = 0.01
    upper_bound: float = 100.0
    transform: str = "log"
    group: str = "pargp"
    package_type: str = "array"  # 'array' or 'list'


class PestSettings(BaseModel):
    """PEST++ control settings."""
    noptmax: int = 20
    phiredstp: float = 0.005
    nphinored: int = 4
    maxsing: Optional[int] = None
    eigthresh: float = 1e-6
    # IES-specific settings
    ies_num_reals: Optional[int] = None
    ies_initial_lambda: Optional[float] = None
    ies_bad_phi_sigma: Optional[float] = None
    ies_subset_size: Optional[int] = None
    # Parallel execution settings
    num_workers: int = 4
    # Network mode - for distributed agents across machines
    network_mode: bool = False
    # Local container agents (Phase 2) - run on main server
    local_containers: bool = False
    local_agents: int = 0  # Number of local Docker container agents
    remote_agents: int = 0  # Number of expected remote network agents


class PestConfig(BaseModel):
    """Full PEST configuration."""
    parameters: list[PestParameterConfig]
    observation_weights: dict[str, float] = {}
    settings: PestSettings = PestSettings()


class PestRunRequest(BaseModel):
    """Request to start a PEST++ calibration run."""
    method: str = "glm"  # "glm" or "ies"
    name: Optional[str] = None
    config: Optional[PestConfig] = None


async def _get_project_or_404(
    project_id: UUID, db: AsyncSession
) -> Project:
    """Get project or raise 404."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    return project


@router.get("/parameters")
async def get_available_parameters(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Discover adjustable parameters from the uploaded model.

    Scans model packages (LPF/UPW/NPF/STO) and returns available
    parameter arrays with per-layer statistics and suggested bounds.

    Results are cached in MinIO to avoid re-scanning on every page load.
    """
    project = await _get_project_or_404(project_id, db)

    if not project.is_valid or not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project does not have a valid model",
        )

    storage = get_storage_service()

    # Check for cached parameters
    cache_path = f"projects/{project_id}/pest/parameters_cache.json"
    if storage.object_exists(settings.minio_bucket_models, cache_path):
        try:
            cached_data = storage.download_file(settings.minio_bucket_models, cache_path)
            cached = json.loads(cached_data)
            # Verify cache is still valid (matches current model storage path)
            if cached.get("storage_path") == project.storage_path:
                return {"parameters": cached["parameters"]}
        except Exception:
            pass  # Cache invalid or corrupted, re-scan

    from app.services.pest_setup import discover_parameters

    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir)

        # Download model files
        files = storage.list_objects(
            settings.minio_bucket_models,
            prefix=project.storage_path,
            recursive=True,
        )
        for obj_name in files:
            rel_path = obj_name[len(project.storage_path) :].lstrip("/")
            if not rel_path:
                continue
            local_path = model_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            file_data = storage.download_file(
                settings.minio_bucket_models, obj_name
            )
            local_path.write_bytes(file_data)

        params = discover_parameters(model_dir)

    # Cache the discovered parameters
    try:
        cache_data = {
            "storage_path": project.storage_path,
            "parameters": params,
            "cached_at": datetime.utcnow().isoformat(),
        }
        storage.upload_bytes(
            settings.minio_bucket_models,
            cache_path,
            json.dumps(cache_data).encode("utf-8"),
            content_type="application/json",
        )
    except Exception:
        pass  # Non-critical if caching fails

    return {"parameters": params}


@router.get("/config")
async def get_pest_config(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get saved PEST configuration for a project."""
    project = await _get_project_or_404(project_id, db)

    storage = get_storage_service()
    config_obj = f"projects/{project_id}/pest/pest_config.json"

    if not storage.object_exists(settings.minio_bucket_models, config_obj):
        return {"config": None}

    data = storage.download_file(settings.minio_bucket_models, config_obj)
    return {"config": json.loads(data)}


@router.post("/config")
async def save_pest_config(
    project_id: UUID,
    config: PestConfig,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Save PEST configuration for a project."""
    await _get_project_or_404(project_id, db)

    storage = get_storage_service()
    config_obj = f"projects/{project_id}/pest/pest_config.json"

    storage.upload_bytes(
        settings.minio_bucket_models,
        config_obj,
        config.model_dump_json().encode("utf-8"),
        content_type="application/json",
    )

    return {"status": "saved"}


@router.post("/run")
async def start_pest_run(
    project_id: UUID,
    request: PestRunRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Start a PEST++ calibration run.

    Creates a run record and queues the calibration Celery task.
    Requires observations to be uploaded first.
    """
    project = await _get_project_or_404(project_id, db)

    if not project.is_valid or not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project does not have a valid model",
        )

    # Check for existing running calibration
    stmt = select(Run).where(
        Run.project_id == project_id,
        Run.run_type.in_([RunType.PEST_GLM, RunType.PEST_IES]),
        Run.status.in_(
            [RunStatus.PENDING, RunStatus.RUNNING, RunStatus.QUEUED]
        ),
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A calibration run is already {existing.status.value}",
        )

    # Get config - either from request or from saved config
    config_data = None
    if request.config:
        config_data = request.config.model_dump()
    else:
        storage = get_storage_service()
        config_obj = f"projects/{project_id}/pest/pest_config.json"
        if storage.object_exists(settings.minio_bucket_models, config_obj):
            data = storage.download_file(
                settings.minio_bucket_models, config_obj
            )
            config_data = json.loads(data)

    if not config_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No PEST configuration provided. Configure parameters first.",
        )

    # Determine run type
    run_type = (
        RunType.PEST_IES if request.method == "ies" else RunType.PEST_GLM
    )

    # Create run record
    run = Run(
        project_id=project_id,
        name=request.name
        or f"PEST++ {request.method.upper()} {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        run_type=run_type,
        status=RunStatus.QUEUED,
        config=config_data,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    # Queue the appropriate Celery task
    if request.method == "ies":
        from app.tasks.calibrate import run_pest_ies
        task = run_pest_ies.delay(
            str(run.id), str(project_id), config_data
        )
    else:
        from app.tasks.calibrate import run_pest_glm
        task = run_pest_glm.delay(
            str(run.id), str(project_id), config_data
        )

    # Update with task ID
    run.celery_task_id = task.id
    await db.commit()

    return {
        "run_id": str(run.id),
        "task_id": task.id,
        "status": "queued",
        "message": f"PEST++ {request.method.upper()} calibration queued",
    }


@router.get("/runs")
async def list_pest_runs(
    project_id: UUID,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list:
    """List calibration runs for a project."""
    await _get_project_or_404(project_id, db)

    stmt = (
        select(Run)
        .where(
            Run.project_id == project_id,
            Run.run_type.in_([RunType.PEST_GLM, RunType.PEST_IES]),
        )
        .order_by(Run.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()

    return [
        {
            "id": str(run.id),
            "name": run.name,
            "run_type": run.run_type.value,
            "status": run.status.value,
            "started_at": run.started_at.isoformat()
            if run.started_at
            else None,
            "completed_at": run.completed_at.isoformat()
            if run.completed_at
            else None,
            "created_at": run.created_at.isoformat()
            if run.created_at
            else None,
        }
        for run in runs
    ]


@router.get("/runs/{run_id}/results")
async def get_pest_results(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get calibration results for a completed PEST++ run."""
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
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

    storage = get_storage_service()
    results_obj = f"{run.results_path}/pest_results.json"

    if not storage.object_exists(settings.minio_bucket_models, results_obj):
        return {"results": None}

    data = storage.download_file(settings.minio_bucket_models, results_obj)
    return {"results": json.loads(data)}


async def _get_pest_results_or_404(
    project_id: UUID, run_id: UUID, db: AsyncSession
) -> dict:
    """Load pest_results.json for a completed PEST run, or raise 404."""
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
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
    storage = get_storage_service()
    results_obj = f"{run.results_path}/pest_results.json"
    if not storage.object_exists(settings.minio_bucket_models, results_obj):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PEST results not found",
        )
    data = storage.download_file(settings.minio_bucket_models, results_obj)
    return json.loads(data)


@router.get("/runs/{run_id}/export/phi")
async def export_phi_csv(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export phi history as CSV."""
    results = await _get_pest_results_or_404(project_id, run_id, db)
    phi_history = results.get("phi_history", [])

    buf = io.StringIO()
    writer = csv.writer(buf)

    has_ies = phi_history and phi_history[0].get("phi_min") is not None
    if has_ies:
        writer.writerow(["iteration", "phi", "phi_min", "phi_max", "phi_std"])
        for entry in phi_history:
            writer.writerow([
                entry.get("iteration", ""),
                entry.get("phi", ""),
                entry.get("phi_min", ""),
                entry.get("phi_max", ""),
                entry.get("phi_std", ""),
            ])
    else:
        writer.writerow(["iteration", "phi"])
        for entry in phi_history:
            writer.writerow([
                entry.get("iteration", ""),
                entry.get("phi", ""),
            ])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="phi_history.csv"'},
    )


@router.get("/runs/{run_id}/export/parameters")
async def export_parameters_csv(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export parameter values as CSV."""
    results = await _get_pest_results_or_404(project_id, run_id, db)
    parameters = results.get("parameters", {})
    ensemble = results.get("ensemble")
    par_summary = ensemble.get("par_summary", {}) if ensemble else {}

    buf = io.StringIO()
    writer = csv.writer(buf)

    has_ensemble = bool(par_summary)
    if has_ensemble:
        writer.writerow(["parameter", "value", "mean", "std", "p5", "p95"])
        for name, value in parameters.items():
            s = par_summary.get(name, {})
            writer.writerow([
                name, value,
                s.get("mean", ""),
                s.get("std", ""),
                s.get("p5", ""),
                s.get("p95", ""),
            ])
    else:
        writer.writerow(["parameter", "value"])
        for name, value in parameters.items():
            writer.writerow([name, value])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="parameters.csv"'},
    )


@router.get("/runs/{run_id}/export/residuals")
async def export_residuals_csv(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export residuals as CSV."""
    results = await _get_pest_results_or_404(project_id, run_id, db)
    residuals = results.get("residuals", [])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["name", "observed", "simulated", "residual"])
    for r in residuals:
        writer.writerow([
            r.get("name", ""),
            r.get("observed", ""),
            r.get("simulated", ""),
            r.get("residual", ""),
        ])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="residuals.csv"'},
    )


@router.get("/runs/{run_id}/output")
async def stream_pest_output(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream PEST++ console output using Server-Sent Events."""
    from app.services.redis_manager import get_async_client

    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    async def generate():
        from app.models.base import get_async_session_factory

        redis_client = await get_async_client()
        pubsub = redis_client.pubsub()
        channel = f"pest:{run_id}:output"
        history_key = f"pest:{run_id}:history"
        db_check_counter = 0

        try:
            # Subscribe first so we don't miss messages during history replay
            await pubsub.subscribe(channel)

            # Replay stored output history
            history = await redis_client.lrange(history_key, 0, -1)
            terminal_status = None
            for line in history:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("__STATUS__:"):
                    status_val = line.split(":", 1)[1]
                    if status_val in ["completed", "failed", "cancelled"]:
                        terminal_status = status_val
                else:
                    yield f"data: {line}\n\n"

            # If the run already finished, send terminal status and stop
            if terminal_status:
                yield f"event: status\ndata: {terminal_status}\n\n"
                return

            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message:
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")

                    if data.startswith("__STATUS__:"):
                        status_val = data.split(":", 1)[1]
                        yield f"event: status\ndata: {status_val}\n\n"
                        if status_val in [
                            "completed",
                            "failed",
                            "cancelled",
                        ]:
                            break
                    else:
                        yield f"data: {data}\n\n"
                    db_check_counter = 0
                else:
                    # No message — periodically check database as fallback
                    db_check_counter += 1
                    if db_check_counter >= 10:
                        db_check_counter = 0
                        session_factory = get_async_session_factory()
                        async with session_factory() as session:
                            stmt = select(Run).where(Run.id == run_id)
                            result = await session.execute(stmt)
                            current_run = result.scalar_one_or_none()
                            if current_run and current_run.status in [
                                RunStatus.COMPLETED,
                                RunStatus.FAILED,
                                RunStatus.CANCELLED,
                            ]:
                                yield f"event: status\ndata: {current_run.status.value}\n\n"
                                break
        finally:
            await pubsub.unsubscribe(channel)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runs/{run_id}/agents")
async def get_agent_status(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get the status of connected PEST++ agents for a network mode run.

    Returns information about:
    - Expected number of agents
    - Currently connected agents
    - Agent status (waiting, running, completed, failed)
    """
    from app.services.redis_manager import get_sync_client

    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    # Get agent info from Redis
    redis_client = get_sync_client()
    agent_key = f"pest:{run_id}:agents"

    agent_data = redis_client.hgetall(agent_key)

    if not agent_data:
        return {
            "network_mode": False,
            "local_containers": False,
            "expected": 0,
            "connected": 0,
            "local_agents_expected": 0,
            "local_agents_connected": 0,
            "remote_agents_expected": 0,
            "remote_agents_connected": 0,
            "status": "not_network_mode",
            "message": "This run is not using network/container mode",
        }

    local_agents_expected = int(agent_data.get("local_agents_expected", 0))
    local_agents_connected = int(agent_data.get("local_agents_connected", 0))
    remote_agents_expected = int(agent_data.get("remote_agents_expected", 0))
    remote_agents_connected = int(agent_data.get("remote_agents_connected", 0))
    has_local_containers = local_agents_expected > 0
    is_network_mode = remote_agents_expected > 0

    return {
        "network_mode": is_network_mode,
        "local_containers": has_local_containers,
        "expected": int(agent_data.get("expected", 0)),
        "connected": int(agent_data.get("connected", 0)),
        "local_agents_expected": local_agents_expected,
        "local_agents_connected": local_agents_connected,
        "remote_agents_expected": remote_agents_expected,
        "remote_agents_connected": remote_agents_connected,
        "status": agent_data.get("status", "unknown"),
        "run_status": run.status.value,
    }


@router.get("/network-info")
async def get_network_info(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get network configuration info for setting up remote agents.

    Returns the manager host/port and MinIO connection details
    that remote agents need to connect.
    """
    await _get_project_or_404(project_id, db)

    return {
        "manager_port": settings.pest_manager_port,
        "minio_port": settings.minio_port,
        "network_mode_enabled": settings.pest_network_mode,
        "instructions": {
            "step1": "Copy docker-compose.agent.yml and Dockerfile.pest-agent to remote machine",
            "step2": "Set MANAGER_HOST to this server's IP address",
            "step3": f"Set MANAGER_PORT to {settings.pest_manager_port}",
            "step4": "Run: docker compose -f docker-compose.agent.yml up -d --scale pest-agent=4",
        },
    }


@router.get("/current-run")
async def get_current_run_info(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get information about the currently running PEST++ job.

    This is used by remote agents to discover what workspace to download.
    """
    from app.services.redis_manager import get_sync_client

    await _get_project_or_404(project_id, db)

    redis_client = get_sync_client()

    run_info = redis_client.get("pest:current_run")
    if not run_info:
        return {
            "has_active_run": False,
            "message": "No active PEST++ run",
        }

    info = json.loads(run_info)

    # Verify it's for this project
    if info.get("project_id") != str(project_id):
        return {
            "has_active_run": False,
            "message": "No active PEST++ run for this project",
        }

    return {
        "has_active_run": True,
        "run_id": info.get("run_id"),
        "workspace_prefix": info.get("workspace_prefix"),
        "pst_file": info.get("pst_file"),
        "manager_port": info.get("manager_port"),
    }
