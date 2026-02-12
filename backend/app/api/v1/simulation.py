"""Simulation execution API endpoints."""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.rate_limit import limiter
from app.models.base import get_db
from app.models.project import Project, Run, RunStatus, RunType
from app.tasks.simulate import run_forward_model, cancel_run

router = APIRouter(prefix="/projects/{project_id}/simulation", tags=["simulation"])
settings = get_settings()


class RunCreate(BaseModel):
    """Request model for creating a new run."""

    name: Optional[str] = None
    save_budget: bool = False  # Enable cell-by-cell budget output (CBC file)


class RunResponse(BaseModel):
    """Response model for a run."""

    id: str
    project_id: str
    name: Optional[str]
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    exit_code: Optional[int]
    error_message: Optional[str]
    results_path: Optional[str]
    created_at: str


async def get_project_or_404(
    project_id: UUID,
    db: AsyncSession,
) -> Project:
    """Get project by ID or raise 404."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    return project


@router.post("/run")
@limiter.limit("10/minute")
async def start_simulation(
    request: Request,
    project_id: UUID,
    run_config: RunCreate = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Start a new forward simulation.

    Creates a run record and queues the simulation task.
    """
    project = await get_project_or_404(project_id, db)

    # Allow running if files are uploaded, even if FloPy validation failed
    # (MODFLOW can still run models that FloPy can't fully parse)
    if not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model files uploaded",
        )

    # Check for existing running simulation
    stmt = select(Run).where(
        Run.project_id == project_id,
        Run.status.in_([RunStatus.PENDING, RunStatus.RUNNING, RunStatus.QUEUED]),
    )
    result = await db.execute(stmt)
    existing_run = result.scalar_one_or_none()

    if existing_run:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A simulation is already {existing_run.status.value}",
        )

    # Create run record with config
    run_options = {}
    if run_config and run_config.save_budget:
        run_options["save_budget"] = True

    run = Run(
        project_id=project_id,
        name=run_config.name if run_config else f"Run {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        run_type=RunType.FORWARD,
        status=RunStatus.QUEUED,
        config=run_options if run_options else None,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    # Queue the Celery task with options
    save_budget = run_config.save_budget if run_config else False
    task = run_forward_model.delay(str(run.id), str(project_id), save_budget=save_budget)

    # Update with task ID
    run.celery_task_id = task.id
    await db.commit()

    return {
        "run_id": str(run.id),
        "task_id": task.id,
        "status": "queued",
        "message": "Simulation queued for execution",
    }


@router.get("/runs")
async def list_runs(
    project_id: UUID,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> list:
    """List recent simulation runs for a project."""
    await get_project_or_404(project_id, db)

    stmt = (
        select(Run)
        .where(Run.project_id == project_id)
        .order_by(Run.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()

    return [
        {
            "id": str(run.id),
            "name": run.name,
            "status": run.status.value,
            "run_type": run.run_type.value,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "exit_code": run.exit_code,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "convergence_info": run.convergence_info,
        }
        for run in runs
    ]


@router.get("/runs/{run_id}")
async def get_run(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get details of a specific run."""
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    return {
        "id": str(run.id),
        "project_id": str(run.project_id),
        "name": run.name,
        "status": run.status.value,
        "run_type": run.run_type.value,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "exit_code": run.exit_code,
        "error_message": run.error_message,
        "results_path": run.results_path,
        "celery_task_id": run.celery_task_id,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "convergence_info": run.convergence_info,
    }


@router.post("/runs/{run_id}/cancel")
async def cancel_simulation(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Cancel a running simulation."""
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    if run.status not in [RunStatus.PENDING, RunStatus.QUEUED, RunStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel run with status {run.status.value}",
        )

    # Queue cancel task
    cancel_run.delay(str(run_id))

    return {
        "run_id": str(run_id),
        "status": "cancelling",
        "message": "Cancel request sent",
    }


@router.get("/runs/{run_id}/output")
async def stream_output(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Stream simulation output using Server-Sent Events.

    Connect to this endpoint to receive live console output.
    """
    from app.services.redis_manager import get_async_client

    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    async def generate() -> AsyncGenerator[str, None]:
        """Generate SSE events from Redis pub/sub, replaying history first."""
        from app.models.base import get_async_session_factory

        redis_client = await get_async_client()
        pubsub = redis_client.pubsub()
        channel = f"simulation:{run_id}:output"
        history_key = f"simulation:{run_id}:history"
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

            # If the run already finished (history contains terminal status), send it and stop
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

                    # Check for status message
                    if data.startswith("__STATUS__:"):
                        status_val = data.split(":", 1)[1]
                        yield f"event: status\ndata: {status_val}\n\n"
                        if status_val in ["completed", "failed", "cancelled"]:
                            break
                    else:
                        yield f"data: {data}\n\n"
                    db_check_counter = 0
                else:
                    # No message — periodically check database as fallback
                    # (every ~10 seconds of silence, not every second)
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


@router.get("/status")
async def get_simulation_status(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get the current simulation status for a project."""
    await get_project_or_404(project_id, db)

    # Get the most recent run
    stmt = (
        select(Run)
        .where(Run.project_id == project_id)
        .order_by(Run.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        return {
            "has_runs": False,
            "latest_run": None,
        }

    return {
        "has_runs": True,
        "latest_run": {
            "id": str(run.id),
            "status": run.status.value,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        },
    }
