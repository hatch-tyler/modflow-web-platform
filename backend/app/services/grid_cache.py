"""Grid mesh and array caching service for improved 3D viewer performance.

Pre-generates and caches grid mesh data and property arrays in MinIO to avoid
expensive model loading on each request.
"""

import struct
import tempfile
from pathlib import Path
from typing import Optional, List

import numpy as np

from app.config import get_settings
from app.services.mesh import (
    generate_grid_mesh,
    get_array_data,
    load_model_from_directory,
    GridMesh,
)
from app.services.storage import get_storage_service

settings = get_settings()

# Cache paths within project storage
GRID_CACHE_FILENAME = "cache/grid.bin"
ARRAY_CACHE_PREFIX = "cache/arrays"

# Arrays to cache during upload for instant viewing
DEFAULT_ARRAYS_TO_CACHE = ["ibound", "top", "botm", "hk", "strt"]


def get_grid_cache_path(project_id: str) -> str:
    """Get the MinIO object path for a project's cached grid mesh."""
    return f"projects/{project_id}/{GRID_CACHE_FILENAME}"


def grid_cache_exists(project_id: str) -> bool:
    """Check if a cached grid mesh exists for a project."""
    storage = get_storage_service()
    cache_path = get_grid_cache_path(project_id)
    return storage.object_exists(settings.minio_bucket_models, cache_path)


def get_cached_grid(project_id: str) -> Optional[bytes]:
    """
    Retrieve cached grid mesh data from MinIO.

    Returns:
        Binary grid mesh data if cache exists, None otherwise.
    """
    storage = get_storage_service()
    cache_path = get_grid_cache_path(project_id)

    try:
        return storage.download_file(settings.minio_bucket_models, cache_path)
    except Exception:
        return None


def cache_grid_mesh(project_id: str, grid_mesh: GridMesh) -> bool:
    """
    Cache a GridMesh object to MinIO.

    Args:
        project_id: Project UUID as string
        grid_mesh: GridMesh object to cache

    Returns:
        True if caching succeeded, False otherwise.
    """
    storage = get_storage_service()
    cache_path = get_grid_cache_path(project_id)

    try:
        binary_data = grid_mesh.to_binary()
        storage.upload_bytes(
            settings.minio_bucket_models,
            cache_path,
            binary_data,
            content_type="application/octet-stream",
        )
        return True
    except Exception as e:
        print(f"Failed to cache grid mesh for project {project_id}: {e}")
        return False


def delete_grid_cache(project_id: str) -> bool:
    """
    Delete cached grid mesh for a project.

    Args:
        project_id: Project UUID as string

    Returns:
        True if deletion succeeded or cache didn't exist.
    """
    storage = get_storage_service()
    cache_path = get_grid_cache_path(project_id)

    try:
        if storage.object_exists(settings.minio_bucket_models, cache_path):
            storage.delete_object(settings.minio_bucket_models, cache_path)
        return True
    except Exception as e:
        print(f"Failed to delete grid cache for project {project_id}: {e}")
        return False


def generate_and_cache_grid(project_id: str, model_dir: Path) -> Optional[GridMesh]:
    """
    Generate grid mesh from a model directory and cache it.

    This is called during model upload after files are extracted.

    Args:
        project_id: Project UUID as string
        model_dir: Path to directory containing model files

    Returns:
        GridMesh if generation succeeded, None otherwise.
    """
    try:
        # Load model from directory
        model = load_model_from_directory(model_dir)
        if model is None:
            print(f"Failed to load model from {model_dir}")
            return None

        # Generate mesh
        mesh = generate_grid_mesh(model)
        if mesh is None:
            print(f"Failed to generate grid mesh for project {project_id}")
            return None

        # Cache to MinIO
        if cache_grid_mesh(project_id, mesh):
            print(f"Cached grid mesh for project {project_id} "
                  f"({mesh.nlay}x{mesh.nrow}x{mesh.ncol} = "
                  f"{mesh.nlay * mesh.nrow * mesh.ncol:,} cells)")

        return mesh

    except Exception as e:
        print(f"Error generating/caching grid for project {project_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def regenerate_grid_cache_from_storage(project_id: str, storage_path: str) -> Optional[GridMesh]:
    """
    Regenerate grid cache for an existing project by downloading model files.

    Used for migrating existing projects that don't have cached grids.

    Args:
        project_id: Project UUID as string
        storage_path: MinIO storage path for the project's model files

    Returns:
        GridMesh if regeneration succeeded, None otherwise.
    """
    storage = get_storage_service()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download all model files
        files = storage.list_objects(
            settings.minio_bucket_models,
            prefix=storage_path,
            recursive=True,
        )

        for obj_name in files:
            # Get relative path within model directory
            rel_path = obj_name[len(storage_path):].lstrip("/")
            if not rel_path:
                continue

            local_path = temp_path / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            file_data = storage.download_file(settings.minio_bucket_models, obj_name)
            local_path.write_bytes(file_data)

        # Generate and cache
        return generate_and_cache_grid(project_id, temp_path)


def regenerate_all_from_storage(
    project_id: str, storage_path: str
) -> dict:
    """
    Regenerate grid cache AND array caches in a single temp directory.

    FloPy uses lazy loading for OPEN/CLOSE external arrays, so the model
    files must remain on disk while arrays are extracted. This function
    keeps a single temp directory alive for both grid and array operations.

    Returns:
        Dict with grid_info, arrays_cached keys.
    """
    storage = get_storage_service()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download all model files
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
            file_data = storage.download_file(settings.minio_bucket_models, obj_name)
            local_path.write_bytes(file_data)

        # Load model once (path normalization happens inside load_model_from_directory)
        model = load_model_from_directory(temp_path)
        if model is None:
            return {"grid_info": None, "arrays_cached": {}}

        # Generate and cache grid mesh
        mesh = generate_grid_mesh(model)
        grid_info = None
        if mesh is not None:
            if cache_grid_mesh(project_id, mesh):
                print(f"Cached grid mesh for project {project_id} "
                      f"({mesh.nlay}x{mesh.nrow}x{mesh.ncol} = "
                      f"{mesh.nlay * mesh.nrow * mesh.ncol:,} cells)")
            grid_info = {
                "grid_type": "structured" if mesh.grid_type == 0 else "unstructured",
                "nlay": mesh.nlay,
                "nrow": mesh.nrow,
                "ncol": mesh.ncol,
                "ncells": mesh.nlay * mesh.nrow * mesh.ncol,
            }

        # Cache arrays (model files still available in temp_path)
        arrays_cached = cache_arrays_from_model(project_id, model)

        return {"grid_info": grid_info, "arrays_cached": arrays_cached}


# ============================================================================
# Array Caching
# ============================================================================

def get_array_cache_path(project_id: str, array_name: str) -> str:
    """Get the MinIO object path for a cached array."""
    return f"projects/{project_id}/{ARRAY_CACHE_PREFIX}/{array_name}.bin"


def array_cache_exists(project_id: str, array_name: str) -> bool:
    """Check if a cached array exists for a project."""
    storage = get_storage_service()
    cache_path = get_array_cache_path(project_id, array_name)
    return storage.object_exists(settings.minio_bucket_models, cache_path)


def get_cached_array(project_id: str, array_name: str) -> Optional[bytes]:
    """
    Retrieve cached array data from MinIO.

    Returns:
        Binary array data if cache exists, None otherwise.
    """
    storage = get_storage_service()
    cache_path = get_array_cache_path(project_id, array_name)

    try:
        return storage.download_file(settings.minio_bucket_models, cache_path)
    except Exception:
        return None


def cache_array(project_id: str, array_name: str, arr: np.ndarray) -> bool:
    """
    Cache a numpy array to MinIO.

    Format: ndim (int32) + shape (ndim x int32) + data (float32)

    Args:
        project_id: Project UUID as string
        array_name: Name of the array (e.g., 'hk', 'ibound')
        arr: Numpy array to cache

    Returns:
        True if caching succeeded, False otherwise.
    """
    storage = get_storage_service()
    cache_path = get_array_cache_path(project_id, array_name)

    try:
        # Convert to float32 for consistent handling
        arr_f32 = arr.astype(np.float32)

        # Build binary: ndim + shape + data
        shape = arr_f32.shape
        ndim = len(shape)
        header = struct.pack("<i", ndim)
        for dim in shape:
            header += struct.pack("<i", dim)

        binary_data = header + arr_f32.tobytes()

        storage.upload_bytes(
            settings.minio_bucket_models,
            cache_path,
            binary_data,
            content_type="application/octet-stream",
        )
        return True
    except Exception as e:
        print(f"Failed to cache array {array_name} for project {project_id}: {e}")
        return False


def list_cached_arrays(project_id: str) -> List[str]:
    """List all cached arrays for a project."""
    storage = get_storage_service()
    prefix = f"projects/{project_id}/{ARRAY_CACHE_PREFIX}/"

    try:
        objects = storage.list_objects(
            settings.minio_bucket_models,
            prefix=prefix,
            recursive=True,
        )
        # Extract array names from paths like "cache/arrays/hk.bin"
        arrays = []
        for obj in objects:
            if obj.endswith(".bin"):
                name = obj.split("/")[-1].replace(".bin", "")
                arrays.append(name)
        return arrays
    except Exception:
        return []


def cache_arrays_from_model(project_id: str, model, array_names: List[str] = None) -> dict:
    """
    Cache multiple arrays from a loaded model.

    Args:
        project_id: Project UUID as string
        model: Loaded FloPy model object
        array_names: List of array names to cache (default: DEFAULT_ARRAYS_TO_CACHE)

    Returns:
        Dict with array names as keys and success status as values.
    """
    if array_names is None:
        array_names = DEFAULT_ARRAYS_TO_CACHE

    results = {}
    for name in array_names:
        try:
            arr = get_array_data(model, name)
            if arr is not None and hasattr(arr, 'shape'):
                success = cache_array(project_id, name, arr)
                results[name] = success
                if success:
                    print(f"Cached array '{name}' for project {project_id} (shape: {arr.shape})")
            else:
                results[name] = False
        except Exception as e:
            print(f"Failed to cache array '{name}': {e}")
            results[name] = False

    return results


def generate_and_cache_arrays(project_id: str, model_dir: Path, array_names: List[str] = None) -> dict:
    """
    Generate and cache arrays from a model directory.

    Args:
        project_id: Project UUID as string
        model_dir: Path to directory containing model files
        array_names: List of array names to cache

    Returns:
        Dict with array names as keys and success status as values.
    """
    try:
        model = load_model_from_directory(model_dir)
        if model is None:
            return {}

        return cache_arrays_from_model(project_id, model, array_names)
    except Exception as e:
        print(f"Error caching arrays for project {project_id}: {e}")
        return {}


def delete_array_caches(project_id: str) -> bool:
    """Delete all cached arrays for a project."""
    storage = get_storage_service()
    prefix = f"projects/{project_id}/{ARRAY_CACHE_PREFIX}/"

    try:
        storage.delete_prefix(settings.minio_bucket_models, prefix)
        return True
    except Exception as e:
        print(f"Failed to delete array caches for project {project_id}: {e}")
        return False
