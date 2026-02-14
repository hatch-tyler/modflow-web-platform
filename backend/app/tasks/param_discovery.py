"""PEST parameter discovery Celery task with progress tracking."""

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.services.redis_manager import get_sync_client
from app.services.storage import get_storage_service
from celery_app import celery_app

settings = get_settings()
logger = logging.getLogger(__name__)

# Redis key TTLs
_PROGRESS_TTL = 3600  # 1 hour for progress hash
_ACTIVE_SCAN_TTL = 600  # 10 min safety TTL for active-scan marker


def _update_progress(
    redis_client, task_id: str, status: str, progress: int, message: str,
    error: str = "",
):
    """Update parameter scan progress in Redis hash."""
    key = f"pest:params:{task_id}"
    redis_client.hset(key, mapping={
        "status": status,
        "progress": str(progress),
        "message": message,
        "error": error,
    })
    redis_client.expire(key, _PROGRESS_TTL)


@celery_app.task(bind=True, name="app.tasks.param_discovery.discover_parameters_task")
def discover_parameters_task(
    self,
    task_id: str,
    project_id: str,
    storage_path: str,
):
    """
    Discover PEST parameters on the Celery worker (3GB container).

    Progress is tracked in Redis hash `pest:params:{task_id}`.
    Result is cached in MinIO at `projects/{project_id}/pest/parameters_cache.json`.
    """
    rc = get_sync_client()
    active_key = f"pest:params:active:{project_id}"

    try:
        _update_progress(rc, task_id, "downloading", 5, "Downloading model files...")

        storage = get_storage_service()
        from app.services.mesh import _is_model_input_file

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            # Download model input files (skip output/results)
            files = storage.list_objects(
                settings.minio_bucket_models,
                prefix=storage_path,
                recursive=True,
            )
            file_count = 0
            for obj_name in files:
                rel_path = obj_name[len(storage_path):].lstrip("/")
                if not rel_path or not _is_model_input_file(rel_path):
                    continue
                local_path = model_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                storage.download_to_file(
                    settings.minio_bucket_models, obj_name, local_path
                )
                file_count += 1
                if file_count % 100 == 0:
                    pct = min(5 + int(45 * file_count / max(file_count + 50, 1)), 49)
                    _update_progress(
                        rc, task_id, "downloading", pct,
                        f"Downloaded {file_count} files..."
                    )

            _update_progress(rc, task_id, "loading", 50, "Loading model...")

            from app.services.pest_setup import discover_parameters
            params = discover_parameters(model_dir)

        _update_progress(rc, task_id, "caching", 90, "Saving results...")

        # Cache the discovered parameters in MinIO
        from app.api.v1.pest import PARAM_CACHE_VERSION
        cache_path = f"projects/{project_id}/pest/parameters_cache.json"
        cache_data = {
            "storage_path": storage_path,
            "cache_version": PARAM_CACHE_VERSION,
            "parameters": params,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        storage.upload_bytes(
            settings.minio_bucket_models,
            cache_path,
            json.dumps(cache_data).encode("utf-8"),
            content_type="application/json",
        )

        # Clear active-scan marker
        rc.delete(active_key)

        _update_progress(rc, task_id, "completed", 100, "Done")
        logger.info(
            "Parameter discovery task %s completed for project %s (%d params found)",
            task_id, project_id, len(params),
        )

    except Exception as e:
        logger.exception("Parameter discovery task %s failed: %s", task_id, e)
        _update_progress(rc, task_id, "failed", 0, "", error=str(e))
        # Always clear active-scan marker on failure
        rc.delete(active_key)
