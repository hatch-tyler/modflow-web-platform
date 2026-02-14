"""HDS file streaming service using HTTP range requests.

This module provides efficient access to MODFLOW binary head files without
downloading the entire file. It reads only the specific bytes needed for
each request by:

1. Building a record index that maps (layer, kper, kstp) to byte offsets
2. Using HTTP Range requests to fetch only the needed data bytes
3. Caching the index in Redis for fast subsequent access

For a 1GB HDS file with 100 timesteps, extracting one slice downloads ~10MB
instead of 1GB (99% reduction in data transfer).

HDS File Format (MODFLOW 6 / MF2005):
- Each record has a 44-byte header:
  - kstp: 4 bytes (int32)
  - kper: 4 bytes (int32)
  - pertim: 4 bytes (float32) or 8 bytes (float64) depending on precision
  - totim: 4 bytes (float32) or 8 bytes (float64)
  - text: 16 bytes (character array, e.g. "HEAD")
  - ncol: 4 bytes (int32)
  - nrow: 4 bytes (int32)
  - ilay: 4 bytes (int32, negative indicates 3D array)
- Data follows header: nrow * ncol * 4 bytes (float32) or 8 bytes (float64)
"""

import logging
import struct
import time
from typing import Optional

import numpy as np

from app.config import get_settings
from app.services.cache_service import get_cache_service
from app.services.storage import get_storage_service

logger = logging.getLogger(__name__)
settings = get_settings()

# HDS header structure (single precision)
HDS_HEADER_SIZE = 44
HDS_HEADER_FORMAT = "<2i2f16s3i"  # kstp, kper, pertim, totim, text, ncol, nrow, ilay

# Double precision header
HDS_HEADER_DP_SIZE = 52
HDS_HEADER_DP_FORMAT = "<2i2d16s3i"


def _find_hds_file(storage, results_path: str) -> Optional[str]:
    """Find the main binary head file in the results path.

    Prefers the file with fewest dots in its name (e.g. flow.hds over
    flow.sfr.hds) to select the main model output.
    """
    output_objects = storage.list_objects(
        settings.minio_bucket_models,
        prefix=results_path,
        recursive=True,
    )
    best = None
    for obj_name in output_objects:
        fname = obj_name.rsplit("/", 1)[-1].lower()
        if fname.endswith(".hds") or fname.endswith(".hed"):
            if best is None:
                best = obj_name
            elif fname.count(".") < best.rsplit("/", 1)[-1].count("."):
                best = obj_name
    return best


def _detect_precision(storage, bucket: str, hds_path: str) -> tuple[str, int, int]:
    """
    Detect if HDS file uses single or double precision.

    Returns:
        Tuple of (format_string, header_size, data_bytes_per_value)
    """
    # Read first header to detect precision
    header_bytes = storage.download_range(bucket, hds_path, 0, HDS_HEADER_DP_SIZE)

    # Try single precision first
    try:
        kstp, kper, pertim, totim, text, ncol, nrow, ilay = struct.unpack(
            HDS_HEADER_FORMAT, header_bytes[:HDS_HEADER_SIZE]
        )
        text_str = text.decode("ascii", errors="ignore").strip()

        # Validate header values are reasonable
        if (
            0 < kstp < 10000
            and 0 < kper < 10000
            and 0 < ncol < 100000
            and 0 < nrow < 100000
            and "HEAD" in text_str.upper()
        ):
            return HDS_HEADER_FORMAT, HDS_HEADER_SIZE, 4
    except Exception:
        pass

    # Try double precision
    try:
        kstp, kper, pertim, totim, text, ncol, nrow, ilay = struct.unpack(
            HDS_HEADER_DP_FORMAT, header_bytes[:HDS_HEADER_DP_SIZE]
        )
        text_str = text.decode("ascii", errors="ignore").strip()

        if (
            0 < kstp < 10000
            and 0 < kper < 10000
            and 0 < ncol < 100000
            and 0 < nrow < 100000
            and "HEAD" in text_str.upper()
        ):
            return HDS_HEADER_DP_FORMAT, HDS_HEADER_DP_SIZE, 8
    except Exception:
        pass

    # Default to single precision
    return HDS_HEADER_FORMAT, HDS_HEADER_SIZE, 4


def build_hds_index(
    project_id: str,
    run_id: str,
    results_path: str,
) -> dict:
    """
    Build or retrieve the HDS record index.

    The index maps each (layer, kper, kstp) combination to:
    - byte_offset: Starting position of the data (after header)
    - data_size: Size of data in bytes

    This allows fetching specific slices using HTTP Range requests.

    Performance optimization: For files under 200MB, downloads the entire file
    once and parses locally. For larger files, uses calculated record positions
    to minimize network round-trips.

    Returns:
        Dict with:
        - records: List of {layer, kper, kstp, byte_offset, data_size}
        - nrow, ncol, nlay: Grid dimensions
        - precision: 'single' or 'double'
        - times: List of simulation times
    """
    total_start = time.time()
    cache = get_cache_service()

    # Check cache first
    t0 = time.time()
    cached = cache.get_hds_index(project_id, run_id)
    cache_time = time.time() - t0
    if cached:
        logger.info(f"[PERF] HDS index cache HIT in {cache_time:.3f}s")
        return cached
    logger.info(f"[PERF] HDS index cache MISS (check took {cache_time:.3f}s)")

    storage = get_storage_service()
    hds_path = _find_hds_file(storage, results_path)

    if not hds_path:
        return {"error": "HDS file not found"}

    bucket = settings.minio_bucket_models
    file_size = storage.get_object_size(bucket, hds_path)
    logger.info(f"[PERF] HDS file size: {file_size / 1024 / 1024:.1f} MB")

    # Detect precision
    header_format, header_size, data_bytes = _detect_precision(storage, bucket, hds_path)
    precision = "double" if data_bytes == 8 else "single"

    records = []
    times = []
    nrow = ncol = nlay = 0
    layers_seen = set()

    # Performance optimization: download entire file for files < 200 MB
    # This uses 1 network request instead of potentially thousands
    MAX_FULL_DOWNLOAD_SIZE = 200 * 1024 * 1024  # 200 MB

    if file_size <= MAX_FULL_DOWNLOAD_SIZE:
        # Download entire file and parse locally - much faster for most models
        t0 = time.time()
        file_data = storage.download_file(bucket, hds_path)
        download_time = time.time() - t0
        logger.info(f"[PERF] Full HDS download took {download_time:.3f}s ({len(file_data)} bytes)")
        offset = 0

        while offset + header_size <= len(file_data):
            try:
                header_bytes = file_data[offset:offset + header_size]
                values = struct.unpack(header_format, header_bytes)
                kstp, kper = values[0], values[1]
                totim = values[3]
                ncol_r, nrow_r, ilay = values[5], values[6], values[7]

                # Validate header
                if not (0 < ncol_r < 100000 and 0 < nrow_r < 100000):
                    break

                if ncol_r > 0:
                    ncol = ncol_r
                if nrow_r > 0:
                    nrow = nrow_r

                layer = abs(ilay) - 1 if ilay != 0 else 0
                layers_seen.add(layer)

                data_size = ncol * nrow * data_bytes
                data_offset = offset + header_size

                records.append({
                    "layer": layer,
                    "kper": kper,
                    "kstp": kstp,
                    "byte_offset": data_offset,
                    "data_size": data_size,
                    "totim": float(totim),
                })

                if totim not in times:
                    times.append(float(totim))

                offset = data_offset + data_size
            except Exception:
                break
    else:
        # For very large files, use calculated positions
        # Read first header to get dimensions
        first_header = storage.download_range(bucket, hds_path, 0, header_size)
        first_values = struct.unpack(header_format, first_header)
        ncol = first_values[5]
        nrow = first_values[6]

        data_size = ncol * nrow * data_bytes
        record_size = header_size + data_size

        # Estimate total records
        estimated_records = file_size // record_size

        # Batch header reads: request multiple headers in one Range request
        # Headers are small (44-52 bytes), so we can fit many in one request
        HEADERS_PER_BATCH = 100
        batch_data_size = HEADERS_PER_BATCH * record_size

        offset = 0
        while offset < file_size:
            # Calculate batch boundaries
            batch_end = min(offset + batch_data_size, file_size)
            batch_bytes = storage.download_range(bucket, hds_path, offset, batch_end - offset)

            local_offset = 0
            while local_offset + header_size <= len(batch_bytes):
                try:
                    header_bytes = batch_bytes[local_offset:local_offset + header_size]
                    values = struct.unpack(header_format, header_bytes)
                    kstp, kper = values[0], values[1]
                    totim = values[3]
                    ncol_r, nrow_r, ilay = values[5], values[6], values[7]

                    if not (0 < ncol_r < 100000 and 0 < nrow_r < 100000):
                        local_offset = len(batch_bytes)  # Exit inner loop
                        break

                    if ncol_r > 0:
                        ncol = ncol_r
                    if nrow_r > 0:
                        nrow = nrow_r

                    layer = abs(ilay) - 1 if ilay != 0 else 0
                    layers_seen.add(layer)

                    current_data_size = ncol * nrow * data_bytes
                    absolute_data_offset = offset + local_offset + header_size

                    records.append({
                        "layer": layer,
                        "kper": kper,
                        "kstp": kstp,
                        "byte_offset": absolute_data_offset,
                        "data_size": current_data_size,
                        "totim": float(totim),
                    })

                    if totim not in times:
                        times.append(float(totim))

                    # Skip to next record
                    local_offset += header_size + current_data_size
                except Exception:
                    local_offset = len(batch_bytes)
                    break

            offset += local_offset
            if local_offset == 0:
                break

    nlay = len(layers_seen) if layers_seen else 1

    index_data = {
        "records": records,
        "nrow": nrow,
        "ncol": ncol,
        "nlay": nlay,
        "precision": precision,
        "times": sorted(times),
        "hds_path": hds_path,
        "file_size": file_size,
    }

    # Cache the index
    t0 = time.time()
    cache.set_hds_index(project_id, run_id, index_data)
    cache_set_time = time.time() - t0

    total_time = time.time() - total_start
    logger.info(f"[PERF] HDS index built in {total_time:.3f}s ({len(records)} records, cache_set: {cache_set_time:.3f}s)")

    return index_data


def get_head_slice_streaming(
    project_id: str,
    run_id: str,
    results_path: str,
    layer: int,
    kper: int,
    kstp: int,
) -> dict:
    """
    Extract a single head slice using streaming (HTTP Range requests).

    Only downloads the specific bytes for the requested timestep/layer,
    dramatically reducing data transfer for large HDS files.

    Args:
        project_id: Project UUID
        run_id: Run UUID
        results_path: Path to results in MinIO
        layer: Layer index (0-based)
        kper: Stress period (1-based as stored in file)
        kstp: Time step within stress period (1-based)

    Returns:
        Dict with shape and data arrays, or error message
    """
    total_start = time.time()
    cache = get_cache_service()

    # Check slice cache first
    t0 = time.time()
    cached = cache.get_slice(project_id, run_id, layer, kper, kstp)
    cache_check_time = time.time() - t0
    if cached:
        logger.info(f"[PERF] Slice cache HIT in {cache_check_time:.3f}s")
        return cached
    logger.info(f"[PERF] Slice cache MISS (check took {cache_check_time:.3f}s)")

    # Get or build HDS index
    t0 = time.time()
    index = build_hds_index(project_id, run_id, results_path)
    index_time = time.time() - t0
    logger.info(f"[PERF] HDS index took {index_time:.3f}s (records: {len(index.get('records', []))})")

    if "error" in index:
        return index

    # Find the matching record
    # Note: HDS files use 1-based kper/kstp, but FloPy (and our API) use 0-based
    # Convert from 0-based (API) to 1-based (file) for lookup
    file_kper = kper + 1
    file_kstp = kstp + 1

    target_record = None
    for rec in index["records"]:
        if rec["layer"] == layer and rec["kper"] == file_kper and rec["kstp"] == file_kstp:
            target_record = rec
            break

    if not target_record:
        # Build available timesteps list (convert back to 0-based for API consistency)
        available = sorted(set(
            (rec["kstp"] - 1, rec["kper"] - 1)
            for rec in index["records"]
            if rec["layer"] == layer
        ))
        return {
            "error": f"Timestep (kstp={kstp}, kper={kper}) not found for layer {layer}",
            "available_timesteps": [[ks, kp] for ks, kp in available],
        }

    # Fetch only the data bytes for this slice
    storage = get_storage_service()
    bucket = settings.minio_bucket_models

    t0 = time.time()
    data_bytes = storage.download_range(
        bucket,
        index["hds_path"],
        target_record["byte_offset"],
        target_record["data_size"],
    )
    download_time = time.time() - t0
    logger.info(f"[PERF] Slice download took {download_time:.3f}s ({len(data_bytes)} bytes)")

    # Convert to numpy array
    t0 = time.time()
    dtype = np.float64 if index["precision"] == "double" else np.float32
    arr = np.frombuffer(data_bytes, dtype=dtype)
    arr = arr.reshape((index["nrow"], index["ncol"]))

    # Mask dry/inactive cells
    masked = np.where(
        (np.abs(arr) > 1e20) | (arr < -880) | (np.isclose(arr, 999.0)),
        np.nan,
        arr,
    )

    # Convert to JSON-serializable format
    data_list = []
    for row in masked.tolist():
        data_list.append([None if (v != v) else float(v) for v in row])

    processing_time = time.time() - t0
    logger.info(f"[PERF] Array processing took {processing_time:.3f}s")

    slice_data = {
        "layer": layer,
        "kper": kper,
        "kstp": kstp,
        "shape": [index["nrow"], index["ncol"]],
        "data": data_list,
        "totim": target_record["totim"],
    }

    # Cache for future requests
    t0 = time.time()
    cache.set_slice(project_id, run_id, layer, kper, kstp, slice_data)
    cache_set_time = time.time() - t0

    total_time = time.time() - total_start
    logger.info(f"[PERF] Total slice extraction: {total_time:.3f}s (cache_set: {cache_set_time:.3f}s)")

    return slice_data


def get_timeseries_streaming(
    project_id: str,
    run_id: str,
    results_path: str,
    layer: int,
    row: int,
    col: int,
) -> dict:
    """
    Extract a head time series for a specific cell using streaming.

    Only downloads 4-8 bytes per timestep instead of the entire array.

    Args:
        project_id: Project UUID
        run_id: Run UUID
        results_path: Path to results in MinIO
        layer: Layer index (0-based)
        row: Row index (0-based)
        col: Column index (0-based)

    Returns:
        Dict with times and heads arrays
    """
    cache = get_cache_service()

    # Check timeseries cache first
    cached = cache.get_timeseries(project_id, run_id, layer, row=row, col=col)
    if cached:
        return cached

    # Get or build HDS index
    index = build_hds_index(project_id, run_id, results_path)
    if "error" in index:
        return index

    # Filter records for this layer
    layer_records = [r for r in index["records"] if r["layer"] == layer]
    layer_records.sort(key=lambda r: (r["kper"], r["kstp"]))

    if not layer_records:
        return {"error": f"No data found for layer {layer}"}

    # Calculate byte offset within each array for the target cell
    bytes_per_value = 8 if index["precision"] == "double" else 4
    cell_offset_in_array = (row * index["ncol"] + col) * bytes_per_value

    storage = get_storage_service()
    bucket = settings.minio_bucket_models

    times = []
    heads = []

    for rec in layer_records:
        # Fetch just the bytes for this cell
        cell_offset = rec["byte_offset"] + cell_offset_in_array

        try:
            cell_bytes = storage.download_range(bucket, index["hds_path"], cell_offset, bytes_per_value)
            dtype = np.float64 if index["precision"] == "double" else np.float32
            value = np.frombuffer(cell_bytes, dtype=dtype)[0]

            # Mask dry/inactive values
            if abs(value) > 1e20 or value < -880 or abs(value - 999.0) < 0.01:
                heads.append(None)
            else:
                heads.append(float(value))

            times.append(rec["totim"])
        except Exception:
            heads.append(None)
            times.append(rec["totim"])

    result = {
        "layer": layer,
        "row": row,
        "col": col,
        "times": times,
        "heads": heads,
    }

    # Cache for future requests
    cache.set_timeseries(project_id, run_id, layer, result, row=row, col=col)

    return result


def get_kstpkper_list(
    project_id: str,
    run_id: str,
    results_path: str,
) -> list[tuple[int, int]]:
    """
    Get list of available (kstp, kper) pairs from HDS index.

    Returns:
        List of (kstp, kper) tuples for available timesteps
    """
    index = build_hds_index(project_id, run_id, results_path)
    if "error" in index:
        return []

    # Get unique kstpkper pairs across all layers
    kstpkper = sorted(set(
        (rec["kstp"], rec["kper"])
        for rec in index["records"]
    ))

    return kstpkper
