"""On-demand head slice extraction from binary HDS files."""

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import get_settings
from app.services.slice_cache import (
    cache_slice,
    cache_timestep_index,
    get_cached_slice,
    get_cached_timestep_index,
)
from app.services.storage import get_storage_service

settings = get_settings()


def _find_hds_file(storage, results_path: str) -> Optional[str]:
    """Find the binary head file in the results path.

    Looks for common MODFLOW head file extensions:
    - .hds (MODFLOW 6, MF2005)
    - .hed (older convention)
    """
    output_objects = storage.list_objects(
        settings.minio_bucket_models,
        prefix=results_path,
        recursive=True,
    )

    # Look for both .hds and .hed extensions, preferring main model file
    best = None
    for obj_name in output_objects:
        fname = obj_name.rsplit("/", 1)[-1].lower()
        if fname.endswith(".hds") or fname.endswith(".hed"):
            if best is None:
                best = obj_name
            elif fname.count(".") < best.rsplit("/", 1)[-1].count("."):
                best = obj_name
    return best


def get_timestep_index(
    project_id: str,
    run_id: str,
    results_path: str,
    model_type: str,
) -> dict:
    """
    Get or build the timestep index for a run.

    The index contains:
    - kstpkper_list: List of [kstp, kper] pairs
    - times: List of simulation times
    - nlay: Number of layers
    - grid_shape: [nlay, nrow, ncol] or [nlay, ncpl] for unstructured

    This is cached in Redis for fast subsequent access.
    Tries to populate from results_summary.json before downloading the HDS file.
    """
    import json

    # Check cache first
    cached = get_cached_timestep_index(project_id, run_id)
    if cached:
        return cached

    storage = get_storage_service()
    is_unstructured = model_type == "mfusg"

    # Try to build index from results_summary.json (avoids downloading multi-GB HDS file)
    summary_obj = f"{results_path}/results_summary.json"
    try:
        if storage.object_exists(settings.minio_bucket_models, summary_obj):
            summary_data = json.loads(
                storage.download_file(settings.minio_bucket_models, summary_obj)
            )
            hs = summary_data.get("heads_summary", {})
            meta = summary_data.get("metadata", {})
            kstpkper_list = hs.get("kstpkper_list", [])
            times = hs.get("times", [])

            if kstpkper_list:
                nlay = meta.get("nlay", 1)
                nrow = meta.get("nrow", 1)
                ncol = meta.get("ncol", 1)

                if is_unstructured or meta.get("grid_type") in ("vertex", "unstructured"):
                    grid_shape = [nlay, ncol]
                else:
                    grid_shape = [nlay, nrow, ncol]

                hds_obj = _find_hds_file(storage, results_path)

                index_data = {
                    "kstpkper_list": kstpkper_list,
                    "times": times,
                    "nlay": nlay,
                    "grid_shape": grid_shape,
                    "is_unstructured": is_unstructured,
                    "hds_path": hds_obj,
                }

                cache_timestep_index(project_id, run_id, index_data)
                return index_data
    except Exception:
        pass  # Fall through to HDS file download

    # Fallback: download HDS file and read index directly
    import flopy.utils

    hds_obj = _find_hds_file(storage, results_path)

    if not hds_obj:
        return {"error": "HDS file not found", "kstpkper_list": [], "times": []}

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "output.hds"
        storage.download_to_file(settings.minio_bucket_models, hds_obj, local_path)

        hds = None
        try:
            if is_unstructured:
                hds = flopy.utils.HeadUFile(str(local_path))
            else:
                hds = flopy.utils.HeadFile(str(local_path))

            kstpkper_list = hds.get_kstpkper()
            times = hds.get_times()

            # Get grid shape from first timestep
            first_data = hds.get_data(kstpkper=kstpkper_list[0])
            if is_unstructured:
                layers_data = first_data if isinstance(first_data, list) else [first_data]
                nlay = len(layers_data)
                ncpl = len(np.asarray(layers_data[0]).flatten())
                grid_shape = [nlay, ncpl]
            else:
                grid_shape = list(first_data.shape)  # [nlay, nrow, ncol]
                nlay = grid_shape[0]

            index_data = {
                "kstpkper_list": [[int(ks), int(kp)] for ks, kp in kstpkper_list],
                "times": [float(t) for t in times] if times else [],
                "nlay": nlay,
                "grid_shape": grid_shape,
                "is_unstructured": is_unstructured,
                "hds_path": hds_obj,
            }

            # Cache for future requests
            cache_timestep_index(project_id, run_id, index_data)

            return index_data

        except Exception as e:
            return {
                "error": str(e),
                "kstpkper_list": [],
                "times": [],
            }
        finally:
            if hds is not None:
                try:
                    hds.close()
                except Exception:
                    pass


def extract_head_slice_on_demand(
    project_id: str,
    run_id: str,
    results_path: str,
    model_type: str,
    layer: int,
    kper: int,
    kstp: int,
) -> dict:
    """
    Extract a single head slice on-demand from the binary HDS file.

    This function:
    1. Checks Redis cache for the slice
    2. If not cached, downloads HDS file and extracts the specific timestep
    3. Caches the result for subsequent requests

    Args:
        project_id: Project UUID
        run_id: Run UUID
        results_path: Path to results in MinIO
        model_type: MODFLOW model type (mf6, mf2005, mfnwt, mfusg)
        layer: Layer index (0-based)
        kper: Stress period (0-based for MF6, 1-based for older versions)
        kstp: Time step within stress period

    Returns:
        Dict with shape and data arrays, or error message
    """
    import flopy.utils

    # Check cache first
    cached = get_cached_slice(project_id, run_id, layer, kper, kstp)
    if cached:
        return cached

    storage = get_storage_service()
    hds_obj = _find_hds_file(storage, results_path)

    if not hds_obj:
        return {"error": "HDS file not found in results"}

    is_unstructured = model_type == "mfusg"

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "output.hds"
        storage.download_to_file(settings.minio_bucket_models, hds_obj, local_path)

        hds = None
        try:
            if is_unstructured:
                hds = flopy.utils.HeadUFile(str(local_path))
            else:
                hds = flopy.utils.HeadFile(str(local_path))

            # Verify the requested timestep exists
            kstpkper_list = hds.get_kstpkper()
            if (kstp, kper) not in kstpkper_list:
                return {
                    "error": f"Timestep (kstp={kstp}, kper={kper}) not found",
                    "available_timesteps": [[int(ks), int(kp)] for ks, kp in kstpkper_list],
                }

            # Extract the data
            data = hds.get_data(kstpkper=(kstp, kper))

            if is_unstructured:
                # HeadUFile returns list of arrays per layer
                layers_data = data if isinstance(data, list) else [data]
                if layer >= len(layers_data):
                    return {"error": f"Layer {layer} not found (max: {len(layers_data) - 1})"}
                arr = np.asarray(layers_data[layer])
            else:
                # HeadFile returns 3D array
                if layer >= data.shape[0]:
                    return {"error": f"Layer {layer} not found (max: {data.shape[0] - 1})"}
                arr = data[layer]

            # Mask dry/inactive cells
            masked = np.where(
                (np.abs(arr) > 1e20) | (arr < -880) | (np.isclose(arr, 999.0)),
                np.nan,
                arr,
            )

            # Convert to JSON-serializable format
            if arr.ndim == 1:
                flat = [None if (v != v) else float(v) for v in masked.tolist()]
                data_list = [flat]
                shape = [1, len(flat)]
            else:
                data_list = []
                for row in masked.tolist():
                    data_list.append([None if (v != v) else float(v) for v in row])
                shape = list(arr.shape)

            slice_data = {
                "layer": layer,
                "kper": kper,
                "kstp": kstp,
                "shape": shape,
                "data": data_list,
            }

            # Cache for future requests
            cache_slice(project_id, run_id, layer, kper, kstp, slice_data)

            return slice_data

        except Exception as e:
            return {"error": f"Error extracting head slice: {str(e)}"}
        finally:
            if hds is not None:
                try:
                    hds.close()
                except Exception:
                    pass


def get_head_statistics(
    project_id: str,
    run_id: str,
    results_path: str,
    model_type: str,
) -> dict:
    """
    Get head statistics without loading full data.

    Returns min/max heads across all timesteps by scanning the file.
    This is useful for color scaling in visualizations.
    Checks results_summary.json first to avoid downloading large HDS files.
    """
    import json

    storage = get_storage_service()

    # Try results_summary.json first (avoids downloading multi-GB HDS file)
    summary_obj = f"{results_path}/results_summary.json"
    try:
        if storage.object_exists(settings.minio_bucket_models, summary_obj):
            summary_data = json.loads(
                storage.download_file(settings.minio_bucket_models, summary_obj)
            )
            hs = summary_data.get("heads_summary", {})
            min_head = hs.get("min_head")
            max_head = hs.get("max_head")
            nstp = hs.get("nstp_total", 0)
            if min_head is not None and max_head is not None:
                return {
                    "min_head": min_head,
                    "max_head": max_head,
                    "timesteps_sampled": 3,
                    "total_timesteps": nstp,
                }
    except Exception:
        pass

    # Fallback: download HDS file and scan
    import flopy.utils

    hds_obj = _find_hds_file(storage, results_path)

    if not hds_obj:
        return {"error": "HDS file not found"}

    is_unstructured = model_type == "mfusg"

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = Path(temp_dir) / "output.hds"
        storage.download_to_file(settings.minio_bucket_models, hds_obj, local_path)

        hds = None
        try:
            if is_unstructured:
                hds = flopy.utils.HeadUFile(str(local_path))
            else:
                hds = flopy.utils.HeadFile(str(local_path))

            kstpkper_list = hds.get_kstpkper()
            global_max = None
            global_min = None

            # Sample a subset of timesteps for statistics (first, middle, last)
            if len(kstpkper_list) > 3:
                indices = [0, len(kstpkper_list) // 2, len(kstpkper_list) - 1]
                sample_timesteps = [kstpkper_list[i] for i in indices]
            else:
                sample_timesteps = kstpkper_list

            for kstp, kper in sample_timesteps:
                try:
                    data = hds.get_data(kstpkper=(kstp, kper))
                    if is_unstructured:
                        layers_data = data if isinstance(data, list) else [data]
                    else:
                        layers_data = [data[i] for i in range(data.shape[0])]

                    for arr in layers_data:
                        arr = np.asarray(arr)
                        masked = np.where(
                            (np.abs(arr) > 1e20) | (arr < -880) | (np.isclose(arr, 999.0)),
                            np.nan,
                            arr,
                        )
                        valid = masked[~np.isnan(masked)]
                        if valid.size > 0:
                            layer_max = float(np.max(valid))
                            layer_min = float(np.min(valid))
                            if global_max is None or layer_max > global_max:
                                global_max = layer_max
                            if global_min is None or layer_min < global_min:
                                global_min = layer_min
                except Exception:
                    continue

            return {
                "min_head": global_min,
                "max_head": global_max,
                "timesteps_sampled": len(sample_timesteps),
                "total_timesteps": len(kstpkper_list),
            }

        except Exception as e:
            return {"error": str(e)}
        finally:
            if hds is not None:
                try:
                    hds.close()
                except Exception:
                    pass
