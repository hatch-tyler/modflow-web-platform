"""Shared dependencies for API v1 endpoints."""
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project, Run, RunStatus


async def get_project_or_404(
    project_id: UUID,
    db: AsyncSession,
    *,
    require_valid: bool = False,
    require_storage: bool = False,
) -> Project:
    """Get project by ID or raise 404.

    Args:
        require_valid: If True, also check project.is_valid.
        require_storage: If True, also check project.storage_path is set.
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if require_valid and not project.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project does not have a valid model uploaded",
        )

    if require_storage and not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project has no model files",
        )

    return project


async def get_run_or_404(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession,
    *,
    allowed_statuses: list[RunStatus] | None = None,
    require_results: bool = False,
) -> Run:
    """Get run by ID within a project, or raise 404.

    Args:
        allowed_statuses: If provided, run.status must be in this list.
        require_results: If True, also check run.results_path is set.
    """
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found in project {project_id}",
        )

    if allowed_statuses and run.status not in allowed_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run is not in expected state (status: {run.status.value})",
        )

    if require_results and not run.results_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results available for this run",
        )

    return run


async def get_completed_run_with_project(
    project_id: UUID,
    run_id: UUID,
    db: AsyncSession,
) -> tuple[Run, Project]:
    """Get a completed run and its project. Convenience for results endpoints."""
    project = await get_project_or_404(project_id, db)
    run = await get_run_or_404(
        project_id, run_id, db,
        allowed_statuses=[RunStatus.COMPLETED],
        require_results=True,
    )
    return run, project
