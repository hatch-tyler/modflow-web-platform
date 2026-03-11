"""Convergence analysis API endpoints."""

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.base import get_db
from app.models.project import Run
from app.services.storage import get_storage_service
from app.api.v1.dependencies import get_completed_run_with_project

router = APIRouter(
    prefix="/projects/{project_id}/runs/{run_id}/results/convergence",
    tags=["convergence"],
)

settings = get_settings()


@router.get("/detail")
async def get_convergence_detail(
    project_id: str,
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return detailed convergence data from convergence_detail.json."""
    run, _project = await get_completed_run_with_project(UUID(project_id), UUID(run_id), db)

    if not run.results_path:
        raise HTTPException(status_code=404, detail="No results available")

    storage = get_storage_service()
    obj_path = f"{run.results_path}/processed/convergence_detail.json"

    try:
        data = storage.download_file(settings.minio_bucket_models, obj_path)
        return json.loads(data)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Convergence detail not available. Re-run post-processing.",
        )


@router.get("/stress-data")
async def get_stress_data(
    project_id: str,
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return stress data summary from stress_summary.json."""
    run, _project = await get_completed_run_with_project(UUID(project_id), UUID(run_id), db)

    if not run.results_path:
        raise HTTPException(status_code=404, detail="No results available")

    storage = get_storage_service()
    obj_path = f"{run.results_path}/processed/stress_summary.json"

    try:
        data = storage.download_file(settings.minio_bucket_models, obj_path)
        return json.loads(data)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Stress data not available. Re-run post-processing.",
        )


@router.get("/recommendations")
async def get_recommendations(
    project_id: str,
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Generate refinement recommendations from convergence + stress data."""
    run, project = await get_completed_run_with_project(UUID(project_id), UUID(run_id), db)

    if not run.results_path:
        raise HTTPException(status_code=404, detail="No results available")

    storage = get_storage_service()

    # Load convergence detail
    try:
        conv_data = json.loads(
            storage.download_file(
                settings.minio_bucket_models,
                f"{run.results_path}/processed/convergence_detail.json",
            )
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Convergence detail not available")

    # Load stress summary (optional)
    stress_data = {}
    try:
        stress_data = json.loads(
            storage.download_file(
                settings.minio_bucket_models,
                f"{run.results_path}/processed/stress_summary.json",
            )
        )
    except Exception:
        pass  # Stress data is optional for recommendations

    from app.services.refinement_service import generate_recommendations
    recommendations = generate_recommendations(conv_data, stress_data)

    return {"recommendations": recommendations}


class ApplyRefinementsRequest(BaseModel):
    refinement_ids: list[str]


@router.post("/apply-refinements")
async def apply_refinements(
    project_id: str,
    run_id: str,
    body: ApplyRefinementsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Apply selected refinements to model files."""
    run, project = await get_completed_run_with_project(UUID(project_id), UUID(run_id), db)

    if not project.storage_path:
        raise HTTPException(status_code=400, detail="Project has no model files")

    if not run.results_path:
        raise HTTPException(status_code=404, detail="No results available")

    storage = get_storage_service()

    # Load convergence detail for recommendations
    try:
        conv_data = json.loads(
            storage.download_file(
                settings.minio_bucket_models,
                f"{run.results_path}/processed/convergence_detail.json",
            )
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Convergence detail not available")

    stress_data = {}
    try:
        stress_data = json.loads(
            storage.download_file(
                settings.minio_bucket_models,
                f"{run.results_path}/processed/stress_summary.json",
            )
        )
    except Exception:
        pass

    from app.services.refinement_service import generate_recommendations, apply_refinements as do_apply
    recommendations = generate_recommendations(conv_data, stress_data)

    model_type = project.model_type.value if project.model_type else "mf6"
    result = do_apply(
        project_id,
        project.storage_path,
        body.refinement_ids,
        recommendations,
        model_type,
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


class RevertRefinementsRequest(BaseModel):
    backup_timestamp: str


@router.post("/revert-refinements")
async def revert_refinements(
    project_id: str,
    run_id: str,
    body: RevertRefinementsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Revert model files from a backup."""
    run, project = await get_completed_run_with_project(UUID(project_id), UUID(run_id), db)

    if not project.storage_path:
        raise HTTPException(status_code=400, detail="Project has no model files")

    from app.services.refinement_service import revert_refinements as do_revert
    result = do_revert(project_id, project.storage_path, body.backup_timestamp)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
