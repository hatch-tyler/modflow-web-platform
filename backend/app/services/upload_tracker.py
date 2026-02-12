"""Upload job tracking service using Redis."""

import json
from typing import Optional
from uuid import uuid4

from app.schemas.upload import UploadStage, UploadStatus
from app.services.redis_manager import get_sync_client

# Redis key prefix for upload jobs
UPLOAD_KEY_PREFIX = "upload_job:"
# TTL for upload job data (1 hour)
UPLOAD_TTL = 3600


def create_upload_job(project_id: str) -> str:
    """Create a new upload job and return its ID."""
    job_id = str(uuid4())
    status = UploadStatus(
        job_id=job_id,
        project_id=project_id,
        stage=UploadStage.RECEIVING,
        progress=0,
        message="Waiting for file upload...",
    )

    r = get_sync_client()
    r.setex(
        f"{UPLOAD_KEY_PREFIX}{job_id}",
        UPLOAD_TTL,
        status.model_dump_json(),
    )
    # Also store mapping from project_id to current job_id
    r.setex(f"upload_project:{project_id}", UPLOAD_TTL, job_id)

    return job_id


def update_upload_status(
    job_id: str,
    stage: Optional[UploadStage] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    file_count: Optional[int] = None,
    files_processed: Optional[int] = None,
    is_valid: Optional[bool] = None,
    error: Optional[str] = None,
) -> None:
    """Update the status of an upload job."""
    r = get_sync_client()
    key = f"{UPLOAD_KEY_PREFIX}{job_id}"

    data = r.get(key)
    if not data:
        return

    status_dict = json.loads(data)

    if stage is not None:
        status_dict["stage"] = stage.value
    if progress is not None:
        status_dict["progress"] = progress
    if message is not None:
        status_dict["message"] = message
    if file_count is not None:
        status_dict["file_count"] = file_count
    if files_processed is not None:
        status_dict["files_processed"] = files_processed
    if is_valid is not None:
        status_dict["is_valid"] = is_valid
    if error is not None:
        status_dict["error"] = error

    r.setex(key, UPLOAD_TTL, json.dumps(status_dict))


def get_upload_status(job_id: str) -> Optional[UploadStatus]:
    """Get the current status of an upload job."""
    r = get_sync_client()
    data = r.get(f"{UPLOAD_KEY_PREFIX}{job_id}")

    if not data:
        return None

    return UploadStatus.model_validate_json(data)


def get_project_upload_job(project_id: str) -> Optional[str]:
    """Get the current upload job ID for a project."""
    r = get_sync_client()
    return r.get(f"upload_project:{project_id}")


def delete_upload_job(job_id: str) -> None:
    """Delete an upload job."""
    r = get_sync_client()

    # Get project_id first to clean up the mapping
    data = r.get(f"{UPLOAD_KEY_PREFIX}{job_id}")
    if data:
        status_dict = json.loads(data)
        project_id = status_dict.get("project_id")
        if project_id:
            r.delete(f"upload_project:{project_id}")

    r.delete(f"{UPLOAD_KEY_PREFIX}{job_id}")
