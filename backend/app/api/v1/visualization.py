"""3D visualization endpoints for MODFLOW grids."""

import logging
import struct
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.base import get_db
from app.models.project import Project
from app.services.storage import get_storage_service
from app.services.grid_cache import (
    get_cached_grid,
    grid_cache_exists,
    regenerate_grid_cache_from_storage,
    get_cached_array,
    array_cache_exists,
    cache_array,
    list_cached_arrays,
)
from app.services.mesh import (
    generate_grid_mesh,
    get_array_data,
    load_and_extract_array,
    load_and_list_arrays,
    load_model_from_storage,
)
from app.services.boundaries import (
    get_boundary_conditions,
    boundary_package_to_dict,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects/{project_id}", tags=["visualization"])
settings = get_settings()


async def get_project_or_404(
    project_id: UUID,
    db: AsyncSession,
    require_valid: bool = True,
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

    if require_valid and not project.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project does not have a valid model uploaded",
        )

    return project


@router.get("/grid")
async def get_grid_mesh(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
    regenerate: bool = False,
) -> Response:
    """
    Get 3D grid mesh data for visualization.

    Uses cached grid data for fast response (~1s instead of ~100s for large models).
    Cache is generated automatically during model upload.

    Query params:
    - regenerate: Force regeneration of grid cache (default: false)

    Returns binary data containing:
    - Header: grid_type, nlay, nrow, ncol (4 x int32)
    - Cell centers: (ncells * 3) x float32
    - delr: ncol x float32 (structured) or vertices: (ncells * 8 * 3) x float32 (USG)
    - delc: nrow x float32 (structured only)
    - top: (nrow * ncol) x float32
    - botm: (nlay * nrow * ncol) x float32
    """
    import asyncio

    project = await get_project_or_404(project_id, db)

    if not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model files found",
        )

    project_id_str = str(project_id)
    storage_path = project.storage_path

    # Try to use cached grid data first (fast path)
    if not regenerate and grid_cache_exists(project_id_str):
        cached_data = get_cached_grid(project_id_str)
        if cached_data:
            # Parse header to get grid info for response headers
            import struct
            grid_type, nlay, nrow, ncol = struct.unpack("<iiii", cached_data[:16])
            return Response(
                content=cached_data,
                media_type="application/octet-stream",
                headers={
                    "X-Grid-Type": "structured" if grid_type == 0 else "unstructured",
                    "X-Grid-Nlay": str(nlay),
                    "X-Grid-Nrow": str(nrow),
                    "X-Grid-Ncol": str(ncol),
                    "X-Grid-Cached": "true",
                },
            )

    # Cache miss or regeneration requested - generate from model (slow path)
    # Run in thread pool to avoid blocking the async event loop
    mesh = await asyncio.to_thread(
        regenerate_grid_cache_from_storage, project_id_str, storage_path
    )
    if mesh is None:
        # Fall back to direct generation without caching
        model = await asyncio.to_thread(
            load_model_from_storage, project_id_str, storage_path
        )
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load model",
            )

        mesh = await asyncio.to_thread(generate_grid_mesh, model)
        if mesh is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate grid mesh",
            )

    return Response(
        content=mesh.to_binary(),
        media_type="application/octet-stream",
        headers={
            "X-Grid-Type": "structured" if mesh.grid_type == 0 else "unstructured",
            "X-Grid-Nlay": str(mesh.nlay),
            "X-Grid-Nrow": str(mesh.nrow),
            "X-Grid-Ncol": str(mesh.ncol),
            "X-Grid-Cached": "false",
        },
    )


@router.get("/grid/info")
async def get_grid_info(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get grid metadata without loading full mesh."""
    project = await get_project_or_404(project_id, db)

    return {
        "nlay": project.nlay,
        "nrow": project.nrow,
        "ncol": project.ncol,
        "nper": project.nper,
        "ncells": (project.nlay or 0) * (project.nrow or 0) * (project.ncol or 0),
        "model_type": project.model_type.value if project.model_type else None,
        "packages": list(project.packages.keys()) if project.packages else [],
    }


@router.get("/arrays/{array_name}")
async def get_array(
    project_id: UUID,
    array_name: str,
    layer: int = None,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """
    Get a cell property array as binary float32 data.

    Uses cached array data for fast response. Cache is generated during upload.

    Supported arrays:
    - ibound/idomain: Active cell indicator
    - top: Top elevation
    - botm: Bottom elevation
    - hk/k: Horizontal hydraulic conductivity
    - vka/k33: Vertical hydraulic conductivity
    - ss: Specific storage
    - sy: Specific yield
    - strt/ic: Starting heads

    Query params:
    - layer: Optional layer index (0-based) to get single layer
    """
    import asyncio

    project = await get_project_or_404(project_id, db)

    if not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model files found",
        )

    project_id_str = str(project_id)
    storage_path = project.storage_path

    # Normalize array name for cache lookup
    cache_name = array_name.lower()

    # Try cache first (fast path) - only for full array requests
    if layer is None and array_cache_exists(project_id_str, cache_name):
        cached_data = get_cached_array(project_id_str, cache_name)
        if cached_data:
            # Parse header to get shape for response headers
            ndim = struct.unpack("<i", cached_data[:4])[0]
            shape = struct.unpack(f"<{ndim}i", cached_data[4:4 + ndim * 4])
            return Response(
                content=cached_data,
                media_type="application/octet-stream",
                headers={
                    "X-Array-Name": array_name,
                    "X-Array-Shape": ",".join(str(d) for d in shape),
                    "X-Array-Dtype": "float32",
                    "X-Array-Cached": "true",
                },
            )

    # Cache miss - load model and extract array in same temp dir
    # (FloPy lazy-loads OPEN/CLOSE external arrays, so files must stay on disk)
    try:
        arr = await asyncio.to_thread(
            load_and_extract_array, project_id_str, storage_path, array_name
        )
    except MemoryError:
        logger.error(f"OOM loading model for project {project_id_str} (arrays)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insufficient memory to load model. Try again shortly.",
        )
    except Exception as e:
        logger.error(f"Failed to load model for project {project_id_str}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )
    if arr is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Array '{array_name}' not found in model",
        )

    # Cache the full array for future requests (async, non-blocking)
    if layer is None:
        try:
            await asyncio.to_thread(cache_array, project_id_str, cache_name, arr)
        except Exception:
            pass  # Caching failure is non-fatal

    # Extract single layer if requested
    if layer is not None:
        if arr.ndim == 3:
            if layer < 0 or layer >= arr.shape[0]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Layer {layer} out of range (0-{arr.shape[0]-1})",
                )
            arr = arr[layer]
        elif arr.ndim == 2 and layer != 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Array is 2D, layer parameter only valid for 3D arrays",
            )

    # Convert to float32 for consistent frontend handling
    arr_f32 = arr.astype(np.float32)

    # Build header with shape info
    shape = arr_f32.shape
    ndim = len(shape)
    header = struct.pack("<i", ndim)  # number of dimensions
    for dim in shape:
        header += struct.pack("<i", dim)

    return Response(
        content=header + arr_f32.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Array-Name": array_name,
            "X-Array-Shape": ",".join(str(d) for d in shape),
            "X-Array-Dtype": "float32",
            "X-Array-Cached": "false",
        },
    )


# Set of projects currently being backfilled (prevents duplicate work)
_backfill_in_progress: set[str] = set()


def _backfill_missing_arrays(
    project_id: str, storage_path: str, missing_names: list[str]
) -> None:
    """
    Cache missing viewer arrays in the background (fire-and-forget).

    Downloads model once and extracts all missing arrays in a single
    pass to avoid redundant multi-GB downloads.
    """
    if project_id in _backfill_in_progress:
        return
    _backfill_in_progress.add(project_id)
    try:
        import tempfile
        from pathlib import Path
        from app.services.mesh import (
            _get_project_lock,
            load_model_from_directory,
            get_array_data,
        )

        lock = _get_project_lock(project_id)
        with lock:
            storage = get_storage_service()

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                files = storage.list_objects(
                    settings.minio_bucket_models,
                    prefix=storage_path,
                    recursive=True,
                )
                for obj_name in files:
                    rel_path = obj_name[len(storage_path):].lstrip("/")
                    if not rel_path:
                        continue
                    local_path = temp_path / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    file_data = storage.download_file(
                        settings.minio_bucket_models, obj_name
                    )
                    local_path.write_bytes(file_data)

                model = load_model_from_directory(temp_path)
                if model is None:
                    logger.warning(
                        f"Backfill: could not load model for {project_id}"
                    )
                    return

                for name in missing_names:
                    try:
                        arr = get_array_data(model, name)
                        if arr is not None:
                            cache_array(project_id, name, arr)
                            logger.info(
                                f"Backfilled array cache '{name}' for "
                                f"project {project_id} (shape: {arr.shape})"
                            )
                    except Exception as e:
                        logger.debug(
                            f"Backfill: {name} not available for "
                            f"{project_id}: {e}"
                        )
    except Exception as e:
        logger.error(f"Backfill failed for project {project_id}: {e}")
    finally:
        _backfill_in_progress.discard(project_id)


@router.get("/arrays")
async def list_arrays(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List available property arrays for the 3D viewer dropdown.

    Returns only material-property arrays suitable for cell colouring
    (hk, vka, ss, sy).  Grid-geometry arrays (top, botm), the active-cell
    mask (ibound) and initial conditions (strt) are still fetchable via
    GET /arrays/{name} but are excluded from this listing because they
    are either redundant with existing UI controls or don't make sense
    as per-cell colour values.

    Uses the MinIO array cache for instant response.  If any viewer
    arrays are missing from the cache (e.g. project uploaded before
    cache expansion), a background task backfills them so the next
    request picks them up.
    """
    import asyncio

    project = await get_project_or_404(project_id, db)

    if not project.storage_path:
        return {"arrays": []}

    project_id_str = str(project_id)

    # Property arrays shown in the viewer dropdown, in display order.
    viewer_arrays = [
        ("hk", "Horizontal hydraulic conductivity"),
        ("vka", "Vertical hydraulic conductivity"),
        ("ss", "Specific storage"),
        ("sy", "Specific yield"),
    ]
    viewer_names = {name for name, _ in viewer_arrays}
    descriptions = dict(viewer_arrays)

    # Read from MinIO cache — populated at upload time, instant response
    try:
        cached_names = await asyncio.to_thread(list_cached_arrays, project_id_str)
    except Exception:
        cached_names = []

    cached_set = set(cached_names)

    available = []
    for name, description in viewer_arrays:
        if name not in cached_set:
            continue
        try:
            cached_data = await asyncio.to_thread(
                get_cached_array, project_id_str, name
            )
            if cached_data is None:
                continue
            # Parse binary: ndim(i4) + shape(ndim*i4) + float32 data
            ndim = struct.unpack("<i", cached_data[:4])[0]
            shape = list(struct.unpack(
                f"<{ndim}i", cached_data[4:4 + ndim * 4]
            ))
            data_offset = 4 + ndim * 4
            arr = np.frombuffer(cached_data[data_offset:], dtype=np.float32)
            min_val = float(np.nanmin(arr))
            max_val = float(np.nanmax(arr))
            available.append({
                "name": name,
                "description": description,
                "shape": shape,
                "min": min_val if np.isfinite(min_val) else 0,
                "max": max_val if np.isfinite(max_val) else 1,
            })
        except Exception:
            pass

    # If any viewer arrays are not yet cached, backfill them in the
    # background so the *next* request picks them up.  This handles
    # projects uploaded before we expanded the default cache list.
    missing = viewer_names - cached_set
    if missing and project.storage_path:
        import threading
        threading.Thread(
            target=_backfill_missing_arrays,
            args=(project_id_str, project.storage_path, list(missing)),
            daemon=True,
        ).start()

    return {"arrays": available}


@router.get("/boundaries")
async def list_boundaries(
    project_id: UUID,
    stress_period: int = 0,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List all boundary condition packages and their cell counts.

    Query params:
    - stress_period: Stress period index (0-based), defaults to 0
    """
    import asyncio

    project = await get_project_or_404(project_id, db)

    if not project.storage_path:
        return {"boundaries": {}, "stress_period": stress_period, "nper": project.nper}

    # Load model - run in thread pool to avoid blocking
    try:
        model = await asyncio.to_thread(
            load_model_from_storage, str(project_id), project.storage_path
        )
    except MemoryError:
        logger.error(f"OOM loading model for project {project_id} (boundaries)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insufficient memory to load model. Try again shortly.",
        )
    except Exception as e:
        logger.error(f"Failed to load model for project {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )
    if model is None:
        return {"boundaries": {}, "stress_period": stress_period, "nper": project.nper}

    boundaries = await asyncio.to_thread(get_boundary_conditions, model, stress_period)

    # Return summary without full cell data
    summary = {}
    for pkg_type, pkg in boundaries.items():
        summary[pkg_type] = {
            "name": pkg.name,
            "description": pkg.description,
            "cell_count": len(pkg.cells),
            "value_names": pkg.value_names,
        }

    return {
        "boundaries": summary,
        "stress_period": stress_period,
        "nper": project.nper,
    }


@router.get("/boundaries/{package_type}")
async def get_boundary_package(
    project_id: UUID,
    package_type: str,
    stress_period: int = 0,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get detailed boundary condition data for a specific package.

    Path params:
    - package_type: Package type (CHD, WEL, RIV, DRN, GHB, RCH, EVT)

    Query params:
    - stress_period: Stress period index (0-based), defaults to 0

    Returns cell locations and values for visualization.
    """
    import asyncio

    project = await get_project_or_404(project_id, db)

    if not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model files found",
        )

    # Load model - run in thread pool to avoid blocking
    try:
        model = await asyncio.to_thread(
            load_model_from_storage, str(project_id), project.storage_path
        )
    except MemoryError:
        logger.error(f"OOM loading model for project {project_id} (boundary package)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Insufficient memory to load model. Try again shortly.",
        )
    except Exception as e:
        logger.error(f"Failed to load model for project {project_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load model",
        )

    boundaries = await asyncio.to_thread(get_boundary_conditions, model, stress_period)
    package_type_upper = package_type.upper()

    if package_type_upper not in boundaries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Boundary package '{package_type}' not found in model",
        )

    return boundary_package_to_dict(boundaries[package_type_upper])


@router.post("/grid/regenerate-cache")
async def regenerate_grid_cache(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Regenerate the cached grid mesh for a project.

    Useful for migrating existing projects that were uploaded
    before grid caching was implemented.

    Returns information about the regenerated cache.
    """
    import asyncio

    project = await get_project_or_404(project_id, db)

    if not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model files found",
        )

    project_id_str = str(project_id)
    storage_path = project.storage_path

    # Generate and cache - run in thread pool to avoid blocking
    mesh = await asyncio.to_thread(
        regenerate_grid_cache_from_storage, project_id_str, storage_path
    )
    if mesh is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate grid cache",
        )

    return {
        "status": "success",
        "project_id": project_id_str,
        "grid_info": {
            "grid_type": "structured" if mesh.grid_type == 0 else "unstructured",
            "nlay": mesh.nlay,
            "nrow": mesh.nrow,
            "ncol": mesh.ncol,
            "ncells": mesh.nlay * mesh.nrow * mesh.ncol,
        },
        "cache_size_bytes": len(mesh.to_binary()),
    }


@router.post("/cache/regenerate-all")
async def regenerate_all_caches(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Regenerate all cached data (grid mesh and arrays) for a project.

    Useful for migrating existing projects that were uploaded
    before caching was implemented, or to refresh stale caches.

    Uses a single temp directory for both grid and array generation
    so FloPy's lazy-loaded OPEN/CLOSE external arrays remain accessible.

    Returns information about the regenerated caches.
    """
    import asyncio
    from app.services.grid_cache import regenerate_all_from_storage

    project = await get_project_or_404(project_id, db)

    if not project.storage_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model files found",
        )

    project_id_str = str(project_id)
    storage_path = project.storage_path

    result = await asyncio.to_thread(
        regenerate_all_from_storage, project_id_str, storage_path
    )

    return {
        "status": "success",
        "project_id": project_id_str,
        "grid_cached": result["grid_info"] is not None,
        "grid_info": result["grid_info"],
        "arrays_cached": result["arrays_cached"],
    }
