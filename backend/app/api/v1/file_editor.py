"""File content read/write/backup API endpoints."""

import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.base import get_db
from app.services.storage import get_storage_service
from app.api.v1.dependencies import get_project_or_404

router = APIRouter(
    prefix="/projects/{project_id}/files",
    tags=["file-editor"],
)

settings = get_settings()

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

BINARY_EXTENSIONS = {
    ".hds", ".hed", ".cbc", ".bud", ".cbb", ".grb",
    ".exe", ".dll", ".so", ".zip", ".gz", ".tar",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp",
}


def _validate_file_path(file_path: str) -> None:
    """Validate file path against traversal attacks.

    Uses PurePosixPath resolution to catch encoded traversal sequences,
    and rejects absolute/UNC paths.
    """
    # Reject absolute paths (Unix and Windows UNC)
    if file_path.startswith("/") or file_path.startswith("\\") or ":" in file_path:
        raise HTTPException(status_code=400, detail="Invalid file path")

    # Resolve and check for traversal (.. anywhere in resolved path)
    resolved = PurePosixPath(file_path)
    try:
        # .resolve() on PurePosixPath doesn't touch filesystem but
        # iterating parts catches ".." components
        for part in resolved.parts:
            if part == "..":
                raise HTTPException(status_code=400, detail="Invalid file path")
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid file path")


@router.get("/{file_path:path}/content")
async def get_file_content(
    project_id: str,
    file_path: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the text content of a model file."""
    project = await get_project_or_404(UUID(project_id), db, require_storage=True)

    # Security: prevent path traversal
    _validate_file_path(file_path)

    # Check for binary files
    ext = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    if ext in BINARY_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Binary files cannot be viewed as text")

    storage = get_storage_service()
    obj_path = f"{project.storage_path}/{file_path}"

    try:
        # Check file size first
        file_size = storage.get_object_size(settings.minio_bucket_models, obj_path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size / 1024 / 1024:.1f}MB). Maximum is 5MB.",
            )

        data = storage.download_file(settings.minio_bucket_models, obj_path)
        content = data.decode("utf-8", errors="replace")

        return {
            "content": content,
            "size": file_size,
            "encoding": "utf-8",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")


class SaveContentRequest(BaseModel):
    content: str
    create_backup: bool = True


@router.put("/{file_path:path}/content")
async def save_file_content(
    project_id: str,
    file_path: str,
    body: SaveContentRequest,
    db: AsyncSession = Depends(get_db),
):
    """Save modified content to a model file, optionally creating a backup."""
    project = await get_project_or_404(UUID(project_id), db, require_storage=True)

    _validate_file_path(file_path)

    storage = get_storage_service()
    obj_path = f"{project.storage_path}/{file_path}"

    # Create backup if requested
    backup_timestamp = None
    if body.create_backup:
        try:
            existing = storage.download_file(settings.minio_bucket_models, obj_path)
            backup_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = f"projects/{project_id}/file_backups/{file_path}.{backup_timestamp}"
            storage.upload_bytes(
                settings.minio_bucket_models,
                backup_path,
                existing,
                content_type="application/octet-stream",
            )
        except Exception:
            pass  # File might not exist yet

    # Upload new content
    try:
        content_bytes = body.content.encode("utf-8")
        storage.upload_bytes(
            settings.minio_bucket_models,
            obj_path,
            content_bytes,
            content_type="text/plain",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    return {
        "saved": True,
        "size": len(content_bytes),
        "backup_timestamp": backup_timestamp,
    }


@router.get("/{file_path:path}/backups")
async def list_file_backups(
    project_id: str,
    file_path: str,
    db: AsyncSession = Depends(get_db),
):
    """List backup versions of a file."""
    project = await get_project_or_404(UUID(project_id), db, require_storage=True)

    storage = get_storage_service()
    backup_prefix = f"projects/{project_id}/file_backups/{file_path}."

    backups = []
    try:
        objects = storage.list_objects(
            settings.minio_bucket_models,
            prefix=backup_prefix,
            recursive=False,
        )
        for obj_name in objects:
            # Extract timestamp from filename
            ts = obj_name.rsplit(".", 1)[-1]
            try:
                size = storage.get_object_size(settings.minio_bucket_models, obj_name)
                backups.append({
                    "timestamp": ts,
                    "size": size,
                    "path": obj_name,
                })
            except Exception:
                backups.append({"timestamp": ts, "size": 0, "path": obj_name})
    except Exception:
        pass

    # Sort by timestamp descending
    backups.sort(key=lambda b: b["timestamp"], reverse=True)

    return {"backups": backups}


class RevertRequest(BaseModel):
    backup_timestamp: str = Field(
        ..., pattern=r"^\d{8}_\d{6}$",
        description="Timestamp in YYYYMMDD_HHMMSS format",
    )


@router.post("/{file_path:path}/revert")
async def revert_file(
    project_id: str,
    file_path: str,
    body: RevertRequest,
    db: AsyncSession = Depends(get_db),
):
    """Revert a file to a backup version."""
    project = await get_project_or_404(UUID(project_id), db, require_storage=True)

    storage = get_storage_service()
    backup_path = f"projects/{project_id}/file_backups/{file_path}.{body.backup_timestamp}"
    obj_path = f"{project.storage_path}/{file_path}"

    try:
        backup_data = storage.download_file(settings.minio_bucket_models, backup_path)
    except Exception:
        raise HTTPException(status_code=404, detail="Backup not found")

    try:
        storage.upload_bytes(
            settings.minio_bucket_models,
            obj_path,
            backup_data,
            content_type="text/plain",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore file: {e}")

    content = backup_data.decode("utf-8", errors="replace")
    return {
        "reverted": True,
        "content": content,
        "size": len(backup_data),
    }
