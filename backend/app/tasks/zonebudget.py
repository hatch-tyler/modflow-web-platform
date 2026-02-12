"""Zone budget Celery task with progress tracking."""

import json
import logging
from uuid import uuid4

from app.config import get_settings
from app.models.base import SessionLocal
from app.models.project import Project, Run, RunStatus
from app.services.redis_manager import get_sync_client
from app.services.storage import get_storage_service
from celery_app import celery_app
from sqlalchemy import select

settings = get_settings()
logger = logging.getLogger(__name__)

# Redis hash TTL for progress tracking
_PROGRESS_TTL = 3600  # 1 hour


def _update_progress(
    redis_client, task_id: str, status: str, progress: int, message: str,
    result_path: str = "", error: str = "",
):
    """Update zone budget progress in Redis hash."""
    key = f"zb:progress:{task_id}"
    redis_client.hset(key, mapping={
        "status": status,
        "progress": str(progress),
        "message": message,
        "result_path": result_path,
        "error": error,
    })
    redis_client.expire(key, _PROGRESS_TTL)


@celery_app.task(bind=True, name="app.tasks.zonebudget.compute_zone_budget_task")
def compute_zone_budget_task(
    self,
    task_id: str,
    run_id: str,
    project_id: str,
    zone_layers: dict,
    quick_mode: bool = False,
):
    """
    Compute zone budget on the Celery worker (12GB container).

    Progress is tracked in Redis hash `zb:progress:{task_id}`.
    Result is uploaded to MinIO and path stored in the progress hash.
    """
    rc = get_sync_client()
    _update_progress(rc, task_id, "downloading", 5, "Downloading CBC file...")

    try:
        with SessionLocal() as db:
            run = db.execute(
                select(Run).where(Run.id == run_id)
            ).scalar_one_or_none()

            if not run or run.status != RunStatus.COMPLETED or not run.results_path:
                _update_progress(rc, task_id, "failed", 0, "", error="Run not found or not completed")
                return

            project = db.execute(
                select(Project).where(Project.id == project_id)
            ).scalar_one_or_none()

            model_type = project.model_type.value if project and project.model_type else "mf6"
            results_path = run.results_path

        storage = get_storage_service()

        # Import here to avoid circular imports
        from app.api.v1.zonebudget import (
            _find_cbc_object,
            _load_grid_dimensions,
            _sync_compute_zone_budget,
            compute_zone_hash,
        )

        cbc_obj = _find_cbc_object(storage, results_path)
        if not cbc_obj:
            _update_progress(rc, task_id, "failed", 0, "", error="CBC file not found in results")
            return

        dims = _load_grid_dimensions(storage, results_path)

        _update_progress(rc, task_id, "computing", 10, "Starting zone budget computation...")

        def progress_cb(current: int, total: int, message: str):
            # Map computation progress to 10-90% range
            pct = 10 + int(80 * current / max(total, 1))
            _update_progress(rc, task_id, "computing", pct, message)

        result = _sync_compute_zone_budget(
            cbc_obj=cbc_obj,
            model_type=model_type,
            zone_layers=zone_layers,
            nlay=dims["nlay"],
            nrow=dims["nrow"],
            ncol=dims["ncol"],
            quick_mode=quick_mode,
            progress_callback=progress_cb,
        )

        _update_progress(rc, task_id, "computing", 92, "Uploading results...")

        # Upload result to MinIO
        zone_hash = compute_zone_hash(zone_layers, quick_mode)
        result_obj = f"{results_path}/processed/zone_budget_{zone_hash}.json"
        storage.upload_bytes(
            settings.minio_bucket_models,
            result_obj,
            json.dumps(result).encode("utf-8"),
            content_type="application/json",
        )

        _update_progress(rc, task_id, "completed", 100, "Zone budget computed successfully", result_path=result_obj)
        logger.info(f"Zone budget task {task_id} completed, result at {result_obj}")

    except Exception as e:
        logger.exception(f"Zone budget task {task_id} failed: {e}")
        _update_progress(rc, task_id, "failed", 0, "", error=str(e))
