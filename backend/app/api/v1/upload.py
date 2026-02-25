"""Model upload and validation endpoints."""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.rate_limit import limiter
from app.models.base import get_db, SessionLocal
from app.models.project import ModelType, Project
from app.schemas.project import ValidationReport
from app.schemas.upload import UploadStage, UploadStatus
from app.schemas.observation import CategorizedFiles, FileInfo, ObservationSetSummary
from app.services.file_classifier import (
    get_blocked_files,
    classify_directory,
    get_categorized_summary,
    detect_observation_csv,
)
from app.services.grid_cache import (
    delete_grid_cache,
    delete_array_caches,
    generate_and_cache_grid,
    generate_and_cache_arrays,
)
from app.services.modflow import extract_and_validate_zip, validate_model, find_model_directory
from app.services.path_normalizer import (
    extract_zip_with_normalized_paths,
    normalize_all_model_files,
    normalize_path,
)
from app.services.storage import get_storage_service
from app.services.upload_tracker import (
    create_upload_job,
    update_upload_status,
    get_upload_status,
    get_project_upload_job,
    delete_upload_job,
)

router = APIRouter(tags=["upload"])
settings = get_settings()


def process_upload_sync(
    job_id: str,
    project_id: str,
    contents: bytes,
) -> None:
    """
    Process the upload synchronously (runs in background thread).
    Updates status in Redis at each stage.
    """
    import time

    storage = get_storage_service()
    storage_path = f"projects/{project_id}/model"

    try:
        # Stage 1: Extracting
        update_upload_status(
            job_id,
            stage=UploadStage.EXTRACTING,
            progress=0,
            message="Extracting ZIP contents...",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            extracted_path = temp_path / "extracted"
            extracted_path.mkdir(parents=True, exist_ok=True)

            # Extract ZIP with normalized paths
            file_count, extracted_files = extract_zip_with_normalized_paths(
                contents, extracted_path
            )

            update_upload_status(
                job_id,
                progress=30,
                message=f"Extracted {file_count} files, checking for blocked files...",
                file_count=file_count,
            )

            # Check for blocked files (executables, scripts, nested archives)
            blocked_files = get_blocked_files(extracted_files)
            if blocked_files:
                blocked_list = ", ".join(blocked_files[:5])
                if len(blocked_files) > 5:
                    blocked_list += f" and {len(blocked_files) - 5} more"
                update_upload_status(
                    job_id,
                    stage=UploadStage.FAILED,
                    error=f"Blocked files detected: {blocked_list}. "
                          f"Executables, scripts, and nested archives are not allowed.",
                    message="Upload rejected due to blocked files",
                )
                return

            update_upload_status(
                job_id,
                progress=50,
                message=f"Extracted {file_count} files",
                file_count=file_count,
            )

            if file_count > settings.max_model_files:
                update_upload_status(
                    job_id,
                    stage=UploadStage.FAILED,
                    error=f"Too many files in ZIP (max {settings.max_model_files})",
                )
                return

            # Normalize paths inside MODFLOW input files
            files_modified, paths_normalized = normalize_all_model_files(extracted_path)

            update_upload_status(
                job_id,
                progress=100,
                message=f"Extracted {file_count} files, normalized {paths_normalized} paths",
            )

            # Stage 2: Validating
            update_upload_status(
                job_id,
                stage=UploadStage.VALIDATING,
                progress=0,
                message="Validating MODFLOW model with FloPy...",
            )

            validation_result, subdir = extract_and_validate_zip(contents)

            update_upload_status(
                job_id,
                progress=100,
                message=f"Validation {'passed' if validation_result.is_valid else 'failed'}",
                is_valid=validation_result.is_valid,
            )

            # Stage 3: Storing in MinIO
            update_upload_status(
                job_id,
                stage=UploadStage.STORING,
                progress=0,
                message="Clearing old files...",
            )

            # Delete any existing files and caches
            try:
                storage.delete_prefix(settings.minio_bucket_models, storage_path)
                delete_grid_cache(project_id)
                delete_array_caches(project_id)
            except Exception as e:
                logger.warning(
                    "Failed to clean old files for project %s: %s", project_id, e
                )

            # Get list of files to upload
            files_to_upload = [f for f in extracted_path.rglob("*") if f.is_file()]
            total_files = len(files_to_upload)

            update_upload_status(
                job_id,
                message=f"Uploading {total_files} files to storage...",
                file_count=total_files,
                files_processed=0,
            )

            # Upload all extracted files with progress
            for i, file_path in enumerate(files_to_upload):
                relative_path = file_path.relative_to(extracted_path)
                object_name = f"{storage_path}/{normalize_path(str(relative_path))}"

                with open(file_path, "rb") as f:
                    file_data = f.read()
                    storage.upload_bytes(
                        settings.minio_bucket_models,
                        object_name,
                        file_data,
                    )

                # Update progress every 10 files or at the end
                if (i + 1) % 10 == 0 or i == total_files - 1:
                    progress = int(((i + 1) / total_files) * 100)
                    update_upload_status(
                        job_id,
                        progress=progress,
                        message=f"Uploaded {i + 1} of {total_files} files",
                        files_processed=i + 1,
                    )

            # Stage 4: Caching (only if valid)
            if validation_result.is_valid:
                update_upload_status(
                    job_id,
                    stage=UploadStage.CACHING,
                    progress=0,
                    message="Generating grid cache for 3D viewer...",
                )

                try:
                    generate_and_cache_grid(project_id, extracted_path)
                    update_upload_status(
                        job_id,
                        progress=40,
                        message="Grid cached, generating array caches...",
                    )

                    generate_and_cache_arrays(project_id, extracted_path)
                    update_upload_status(
                        job_id,
                        progress=70,
                        message="Array caches generated, classifying files...",
                    )
                except Exception as e:
                    # Caching failure is non-fatal
                    update_upload_status(
                        job_id,
                        progress=70,
                        message=f"Cache generation skipped: {str(e)[:100]}",
                    )

                # Classify files and store categorization
                try:
                    categories = classify_directory(extracted_path)
                    summary = get_categorized_summary(categories)

                    # Store categorization as JSON in MinIO
                    categorization_data = {
                        "categories": categories,
                        "summary": summary,
                    }
                    storage.upload_bytes(
                        settings.minio_bucket_models,
                        f"projects/{project_id}/categorization.json",
                        json.dumps(categorization_data).encode("utf-8"),
                        content_type="application/json",
                    )

                    # Auto-detect observation CSVs
                    obs_files = categories.get("observation", [])
                    detected_obs = []
                    for obs_file in obs_files:
                        obs_path = extracted_path / obs_file["path"]
                        if obs_path.exists():
                            obs_info = detect_observation_csv(obs_path)
                            if obs_info:
                                detected_obs.append({
                                    "file_path": obs_file["path"],
                                    **obs_info,
                                })

                    if detected_obs:
                        # Store detected observations metadata
                        storage.upload_bytes(
                            settings.minio_bucket_models,
                            f"projects/{project_id}/detected_observations.json",
                            json.dumps(detected_obs).encode("utf-8"),
                            content_type="application/json",
                        )

                    update_upload_status(
                        job_id,
                        progress=100,
                        message=f"All caches generated, {len(obs_files)} observation files detected",
                    )
                except Exception as e:
                    update_upload_status(
                        job_id,
                        progress=100,
                        message=f"Caching complete, file classification skipped: {str(e)[:100]}",
                    )

            # Update project in database (sync — runs in background thread)
            update_project_after_upload(
                project_id, storage_path, validation_result
            )

            # Stage 5: Complete
            update_upload_status(
                job_id,
                stage=UploadStage.COMPLETE,
                progress=100,
                message="Upload complete!",
                is_valid=validation_result.is_valid,
            )

    except Exception as e:
        update_upload_status(
            job_id,
            stage=UploadStage.FAILED,
            error=str(e),
            message=f"Upload failed: {str(e)[:200]}",
        )


def update_project_after_upload(
    project_id: str,
    storage_path: str,
    validation_result,
) -> None:
    """Update project record after upload processing (sync, runs in background thread)."""
    with SessionLocal() as db:
        stmt = select(Project).where(Project.id == UUID(project_id))
        result = db.execute(stmt)
        project = result.scalar_one_or_none()

        if project is None:
            return

        project.storage_path = storage_path
        project.is_valid = validation_result.is_valid

        if validation_result.is_valid:
            model_type_map = {
                "mf2005": ModelType.MODFLOW_2005,
                "mfnwt": ModelType.MODFLOW_NWT,
                "mf6": ModelType.MODFLOW_6,
            }
            project.model_type = model_type_map.get(
                validation_result.model_type, ModelType.UNKNOWN
            )
            project.nlay = validation_result.nlay
            project.nrow = validation_result.nrow
            project.ncol = validation_result.ncol
            project.nper = validation_result.nper
            project.grid_type = validation_result.grid_type
            project.xoff = validation_result.xoff
            project.yoff = validation_result.yoff
            project.angrot = validation_result.angrot
            project.epsg = validation_result.epsg
            project.length_unit = validation_result.length_unit
            project.stress_period_data = validation_result.stress_period_data
            project.delr = validation_result.delr
            project.delc = validation_result.delc
            project.time_unit = validation_result.time_unit
            project.packages = validation_result.packages
            project.validation_errors = None
        else:
            project.model_type = ModelType.UNKNOWN
            project.validation_errors = {"errors": validation_result.errors}

        db.commit()


@router.post(
    "/projects/{project_id}/upload",
    response_model=ValidationReport,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("5/minute")
async def upload_model(
    request: Request,
    project_id: UUID,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    async_mode: bool = False,
):
    """
    Upload a ZIP file containing MODFLOW model input files.

    The ZIP file will be extracted, validated using FloPy, and stored.
    Model metadata (grid dimensions, packages, etc.) will be extracted
    and stored in the project record.

    Query params:
    - async_mode: If true, returns immediately with job_id for polling (default: false)
    """
    # Verify project exists
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a ZIP archive",
        )

    # Check file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb:.1f} MB) exceeds maximum ({settings.max_upload_size_mb} MB)",
        )

    # Create upload job for tracking
    job_id = create_upload_job(str(project_id))

    if async_mode:
        # Run processing in background thread via FastAPI's BackgroundTasks
        # (avoids leaking a ThreadPoolExecutor on every async upload)
        background_tasks.add_task(process_upload_sync, job_id, str(project_id), contents)

        return {"job_id": job_id, "message": "Upload started, poll /upload/status/{job_id} for progress"}

    # Synchronous mode: process and wait for completion
    update_upload_status(
        job_id,
        stage=UploadStage.RECEIVING,
        progress=100,
        message=f"Received {size_mb:.1f} MB file",
    )

    # Run the processing in a thread to avoid blocking
    await asyncio.to_thread(
        process_upload_sync,
        job_id,
        str(project_id),
        contents,
    )

    # Get final status
    final_status = get_upload_status(job_id)

    if final_status and final_status.stage == UploadStage.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=final_status.error or "Upload failed",
        )

    # Refresh project data
    await db.refresh(project)

    return ValidationReport(
        is_valid=project.is_valid,
        model_type=project.model_type,
        grid_info={
            "nlay": project.nlay or 0,
            "nrow": project.nrow or 0,
            "ncol": project.ncol or 0,
            "nper": project.nper or 0,
            "grid_type": project.grid_type,
        }
        if project.is_valid
        else None,
        packages_found=list(project.packages.keys()) if project.packages else [],
        packages_missing=[],
        errors=project.validation_errors.get("errors", []) if project.validation_errors else [],
        warnings=[],
    )


@router.get(
    "/upload/status/{job_id}",
    response_model=UploadStatus,
)
async def get_upload_job_status(job_id: str) -> UploadStatus:
    """Get the current status of an upload job."""
    upload_status = get_upload_status(job_id)

    if upload_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload job {job_id} not found",
        )

    return upload_status


@router.get(
    "/projects/{project_id}/upload/status",
    response_model=Optional[UploadStatus],
)
async def get_project_upload_status(project_id: UUID) -> Optional[UploadStatus]:
    """Get the current upload status for a project (if any)."""
    job_id = get_project_upload_job(str(project_id))

    if job_id is None:
        return None

    return get_upload_status(job_id)


@router.get(
    "/projects/{project_id}/validation",
    response_model=ValidationReport,
)
async def get_validation(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ValidationReport:
    """Get the current validation status for a project."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    errors = []
    if project.validation_errors:
        errors = project.validation_errors.get("errors", [])

    return ValidationReport(
        is_valid=project.is_valid,
        model_type=project.model_type,
        grid_info={
            "nlay": project.nlay or 0,
            "nrow": project.nrow or 0,
            "ncol": project.ncol or 0,
            "nper": project.nper or 0,
            "grid_type": project.grid_type,
        }
        if project.is_valid
        else None,
        packages_found=list(project.packages.keys()) if project.packages else [],
        packages_missing=[],
        errors=errors,
        warnings=[],
    )


@router.get("/projects/{project_id}/files")
async def list_model_files(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List all files in the project's model storage."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if not project.storage_path:
        return {"files": []}

    storage = get_storage_service()
    files = storage.list_objects(
        settings.minio_bucket_models,
        prefix=project.storage_path,
        recursive=True,
    )

    # Remove storage path prefix for cleaner display
    prefix_len = len(project.storage_path) + 1
    file_list = [f[prefix_len:] for f in files if len(f) > prefix_len]

    return {"files": sorted(file_list)}


@router.get("/projects/{project_id}/files/categorized")
async def get_categorized_files(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> CategorizedFiles:
    """
    Get categorized list of files for a project.

    Categories:
    - model_required: Files FloPy identified as essential (NAM, DIS, BAS, solver, etc.)
    - model_optional: Other MODFLOW input files (arrays, external refs)
    - pest: PEST/pyEMU files (.pst, .tpl, .ins, .par, etc.)
    - observation: Auto-detected observation CSVs
    - other: Unrecognized files
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    storage = get_storage_service()

    # Try to load cached categorization
    categorization_path = f"projects/{project_id}/categorization.json"
    try:
        if storage.object_exists(settings.minio_bucket_models, categorization_path):
            data = storage.download_file(settings.minio_bucket_models, categorization_path)
            cached = json.loads(data)

            # Convert to response format
            categories = {}
            for cat_name, files in cached.get("categories", {}).items():
                categories[cat_name] = [
                    FileInfo(
                        path=f["path"],
                        name=f["name"],
                        extension=f.get("extension", ""),
                        description=f.get("description", ""),
                        size=f.get("size", 0),
                    )
                    for f in files
                ]

            summary = cached.get("summary", {})

            # Load detected observations
            detected_obs = []
            detected_path = f"projects/{project_id}/detected_observations.json"
            if storage.object_exists(settings.minio_bucket_models, detected_path):
                obs_data = storage.download_file(settings.minio_bucket_models, detected_path)
                detected_list = json.loads(obs_data)
                # Convert to ObservationSetSummary (simplified)
                for i, obs in enumerate(detected_list):
                    file_path = obs.get("file_path", "")
                    detected_obs.append(
                        ObservationSetSummary(
                            id=f"detected_{i}",
                            name=Path(file_path).stem if file_path else f"observation_{i}",
                            source="zip_detected",
                            format=obs.get("format", "long"),
                            wells=obs.get("wells", []),
                            n_observations=obs.get("n_observations", 0),
                            created_at=project.updated_at,
                            file_path=file_path,
                        )
                    )

            return CategorizedFiles(
                categories=categories,
                blocked_rejected=[],
                total_files=summary.get("total_files", 0),
                total_size_mb=summary.get("total_size_mb", 0.0),
                detected_observations=detected_obs,
            )
    except Exception:
        pass

    # Fallback: generate categorization from storage listing
    if not project.storage_path:
        return CategorizedFiles()

    files = storage.list_objects(
        settings.minio_bucket_models,
        prefix=project.storage_path,
        recursive=True,
    )

    prefix_len = len(project.storage_path) + 1
    file_list = [f[prefix_len:] for f in files if len(f) > prefix_len]

    # Simple categorization without detailed info
    from app.services.file_classifier import classify_files

    categories_raw = classify_files(file_list)
    categories = {}
    total_files = 0

    for cat_name, files_in_cat in categories_raw.items():
        categories[cat_name] = [
            FileInfo(
                path=f["path"],
                name=f["name"],
                extension=f.get("extension", ""),
                description=f.get("description", ""),
                size=0,  # Size not available without downloading
            )
            for f in files_in_cat
        ]
        total_files += len(files_in_cat)

    return CategorizedFiles(
        categories=categories,
        total_files=total_files,
    )


@router.delete("/projects/{project_id}/files/{file_path:path}")
async def delete_project_file(
    project_id: UUID,
    file_path: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Delete a specific file from the project's storage.

    Returns warning if the file is a required model file.
    Recalculates categorization after deletion.
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model files uploaded for this project",
        )

    storage = get_storage_service()
    object_name = f"{project.storage_path}/{normalize_path(file_path)}"

    # Check if file exists
    if not storage.object_exists(settings.minio_bucket_models, object_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_path} not found",
        )

    # Check if it's a core model file
    from app.services.file_classifier import classify_file

    category = classify_file(file_path)
    is_core = category == "model_core"

    # Delete the file
    storage.delete_object(settings.minio_bucket_models, object_name)

    # Invalidate categorization cache
    categorization_path = f"projects/{project_id}/categorization.json"
    try:
        storage.delete_object(settings.minio_bucket_models, categorization_path)
    except Exception:
        pass

    # Clear grid cache if model file deleted
    if category in ("model_core", "model_input"):
        try:
            delete_grid_cache(str(project_id))
            delete_array_caches(str(project_id))
        except Exception:
            pass

    response = {
        "message": f"File {file_path} deleted successfully",
        "file_path": file_path,
        "category": category,
    }

    if is_core:
        response["warning"] = (
            "This was a core model file. "
            "The model may no longer be valid for simulation."
        )
    elif category == "model_input":
        response["warning"] = (
            "This was a model input file. "
            "The model may fail if this file is referenced in the NAM file."
        )

    return response


@router.get("/projects/{project_id}/files/{file_path:path}/preview")
async def preview_csv_file(
    project_id: UUID,
    file_path: str,
    rows: int = 10,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Preview a CSV file from the project's storage.

    Returns the headers and first N rows of data.
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model files uploaded for this project",
        )

    # Check file extension
    if not file_path.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files can be previewed",
        )

    storage = get_storage_service()
    object_name = f"{project.storage_path}/{normalize_path(file_path)}"

    # Check if file exists
    if not storage.object_exists(settings.minio_bucket_models, object_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_path} not found",
        )

    try:
        # Download and parse CSV
        file_data = storage.download_file(settings.minio_bucket_models, object_name)
        content = file_data.decode('utf-8')
        lines = content.strip().split('\n')

        if not lines:
            return {"headers": [], "rows": [], "total_rows": 0}

        # Parse headers
        headers = [h.strip().strip('"') for h in lines[0].split(',')]

        # Parse data rows (limit to requested number)
        data_rows = []
        for line in lines[1:rows + 1]:
            if line.strip():
                # Handle quoted values properly
                values = []
                in_quotes = False
                current_value = ""
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        values.append(current_value.strip().strip('"'))
                        current_value = ""
                    else:
                        current_value += char
                values.append(current_value.strip().strip('"'))
                data_rows.append(values)

        return {
            "headers": headers,
            "rows": data_rows,
            "total_rows": len(lines) - 1,  # Exclude header
            "file_path": file_path,
        }

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not a valid UTF-8 encoded CSV",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to preview file: {str(e)}",
        )


@router.post(
    "/projects/{project_id}/revalidate",
    response_model=ValidationReport,
    status_code=status.HTTP_200_OK,
)
async def revalidate_model(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Re-run FloPy validation on an already-uploaded model.

    Downloads model files from MinIO to a temp directory, re-validates,
    and updates the project record with refreshed metadata (grid info,
    spatial reference, packages, etc.).
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    if not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model files uploaded for this project",
        )

    storage = get_storage_service()
    storage_prefix = f"{project.storage_path}/"

    # Download model files from MinIO to temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            objects = storage.list_objects(
                settings.minio_bucket_models,
                prefix=storage_prefix,
                recursive=True,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list model files: {str(e)}",
            )

        file_count = 0
        for obj_name in objects:
            # Skip non-model files (caches, categorization, etc.)
            rel_path = obj_name[len(storage_prefix):]
            if not rel_path or rel_path.endswith("/"):
                continue
            if rel_path.startswith("runs/") or rel_path == "categorization.json":
                continue

            local_path = temp_path / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                storage.download_to_file(
                    settings.minio_bucket_models,
                    obj_name,
                    local_path,
                )
                file_count += 1
            except Exception:
                continue

        if file_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No model files found in storage",
            )

        # Find the model directory (might be root or a subdirectory)
        model_dir = find_model_directory(temp_path)
        if model_dir is None:
            model_dir = temp_path

        # Run validation
        validation_result = validate_model(model_dir)

        # Update project record
        if validation_result.is_valid:
            model_type_map = {
                "mf2005": ModelType.MODFLOW_2005,
                "mfnwt": ModelType.MODFLOW_NWT,
                "mf6": ModelType.MODFLOW_6,
            }
            project.model_type = model_type_map.get(
                validation_result.model_type, ModelType.UNKNOWN
            )
            project.nlay = validation_result.nlay
            project.nrow = validation_result.nrow
            project.ncol = validation_result.ncol
            project.nper = validation_result.nper
            project.grid_type = validation_result.grid_type
            if validation_result.xoff is not None:
                project.xoff = validation_result.xoff
            if validation_result.yoff is not None:
                project.yoff = validation_result.yoff
            if validation_result.angrot is not None:
                project.angrot = validation_result.angrot
            if validation_result.epsg is not None:
                project.epsg = validation_result.epsg
            if validation_result.length_unit:
                project.length_unit = validation_result.length_unit
            if validation_result.time_unit:
                project.time_unit = validation_result.time_unit
            if validation_result.stress_period_data:
                project.stress_period_data = validation_result.stress_period_data
            if validation_result.delr:
                project.delr = validation_result.delr
            if validation_result.delc:
                project.delc = validation_result.delc
            if validation_result.packages:
                project.packages = validation_result.packages
            project.is_valid = True
            project.validation_errors = None
        else:
            project.validation_errors = {"errors": validation_result.errors}

        await db.commit()
        await db.refresh(project)

    return ValidationReport(
        is_valid=validation_result.is_valid,
        model_type=validation_result.model_type,
        grid_info={
            "nlay": validation_result.nlay,
            "nrow": validation_result.nrow,
            "ncol": validation_result.ncol,
            "nper": validation_result.nper,
            "grid_type": validation_result.grid_type,
        } if validation_result.nlay else None,
        packages_found=list(validation_result.packages.keys()),
        packages_missing=[],
        errors=validation_result.errors,
        warnings=validation_result.warnings,
    )
