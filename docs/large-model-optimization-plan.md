# Large Model Post-Processing Optimization Plan

## Problem Statement

For large MODFLOW models, the output files (HDS for heads, CBC/CBB for cell budgets) can be **multiple gigabytes** in size. Current post-processing:
- Downloads entire files to temporary storage
- Reads all data into memory via FloPy
- Processes every timestep sequentially
- Converts arrays to JSON (memory-intensive)

For a model like BBK Las Posas with ~55 minute runtime, the output files can easily be 2-10+ GB, causing post-processing to take 10-30+ minutes.

---

## Optimization 1: Memory-Mapped File Reading

### What It Does
Instead of loading entire binary files into memory, memory mapping creates a virtual view of the file that reads data on-demand from disk. Only the bytes actually accessed are loaded into RAM.

### Implementation

**File: `backend/app/tasks/postprocess.py`**

```python
import mmap
import struct

def _process_heads_mmap(hds_path: Path, model_type: str, project: Project,
                        timesteps_to_process: list = None) -> dict:
    """
    Process binary head file using memory-mapped I/O for large files.

    For files > 500MB, uses mmap instead of FloPy's full file loading.
    """
    file_size = hds_path.stat().st_size
    use_mmap = file_size > 500 * 1024 * 1024  # 500 MB threshold

    if not use_mmap:
        # Fall back to standard FloPy processing for smaller files
        return _process_heads_standard(hds_path, model_type, project, timesteps_to_process)

    # Memory-mapped approach for large files
    with open(hds_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Parse header to get timestep offsets without loading data
            header_info = _parse_hds_header(mm, model_type)

            # Build index of (kstp, kper) -> file offset
            timestep_index = _build_timestep_index(mm, header_info)

            # Only read requested timesteps
            if timesteps_to_process is None:
                timesteps_to_process = [list(timestep_index.keys())[-1]]  # Last only

            head_arrays = {}
            for kstp, kper in timesteps_to_process:
                offset = timestep_index.get((kstp, kper))
                if offset:
                    # Seek to offset and read only this timestep's data
                    data = _read_timestep_at_offset(mm, offset, header_info)
                    head_arrays[f"L0_SP{kper}_TS{kstp}"] = data

            return {"summary": header_info, "head_arrays": head_arrays}
```

**New utility functions needed:**

```python
def _parse_hds_header(mm: mmap.mmap, model_type: str) -> dict:
    """Parse MODFLOW binary head file header to get grid dimensions."""
    # MODFLOW binary format: each record has header with kstp, kper, pertim, totim,
    # text (16 bytes), ncol, nrow, ilay
    mm.seek(0)

    # Read first record header (44 bytes for most MODFLOW versions)
    header = mm.read(44)
    kstp, kper, pertim, totim = struct.unpack('<2i2f', header[:16])
    text = header[16:32].decode('ascii').strip()
    ncol, nrow, ilay = struct.unpack('<3i', header[32:44])

    return {
        'ncol': ncol,
        'nrow': nrow,
        'record_size': ncol * nrow * 4,  # 4 bytes per float32
        'header_size': 44,
    }

def _build_timestep_index(mm: mmap.mmap, header_info: dict) -> dict:
    """Scan file to build index of timestep -> file offset."""
    index = {}
    offset = 0
    record_size = header_info['record_size']
    header_size = header_info['header_size']
    file_size = mm.size()

    while offset < file_size - header_size:
        mm.seek(offset)
        header = mm.read(header_size)
        if len(header) < header_size:
            break

        kstp, kper = struct.unpack('<2i', header[:8])
        index[(kstp, kper)] = offset

        # Skip to next record
        offset += header_size + record_size + 8  # +8 for Fortran record markers

    return index

def _read_timestep_at_offset(mm: mmap.mmap, offset: int, header_info: dict) -> dict:
    """Read a single timestep's head data at the given file offset."""
    mm.seek(offset + header_info['header_size'])

    ncol, nrow = header_info['ncol'], header_info['nrow']
    data_bytes = mm.read(ncol * nrow * 4)

    # Convert to numpy array
    arr = np.frombuffer(data_bytes, dtype=np.float32).reshape(nrow, ncol)

    # Mask inactive cells
    masked = np.where(np.abs(arr) > 1e20, np.nan, arr)

    return {
        'shape': [nrow, ncol],
        'data': masked.tolist(),
    }
```

### Benefits
| Metric | Before | After |
|--------|--------|-------|
| Memory usage for 5GB HDS | ~5-10 GB RAM | ~50-100 MB RAM |
| File open time | 30-60 seconds | <1 second |
| Random timestep access | Must scan from start | Direct seek |

### Caveats
- More complex code to maintain
- Need to handle different MODFLOW versions (MF2005, MF6, MFUSG have slightly different formats)
- FloPy already handles format variations; custom code needs same handling

---

## Optimization 2: Parallel Processing

### What It Does
Process heads and budget files concurrently using Python's `concurrent.futures` or Celery subtasks. Currently they run sequentially.

### Implementation

**Option A: ThreadPoolExecutor (simpler, same process)**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_results_parallel(self, run_id: str, project_id: str, quick_mode: bool = True):
    """Extract results with parallel head/budget processing."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # ... download files ...

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            heads_future = executor.submit(
                _process_heads, local_files, model_type, project, quick_mode
            )
            budget_future = executor.submit(
                _process_budget, local_files, model_type, quick_mode
            )

            # Update progress as each completes
            for future in as_completed([heads_future, budget_future]):
                if future == heads_future:
                    heads_result = future.result()
                    update_progress(50, 100, "Heads processing complete")
                else:
                    budget_result = future.result()
                    update_progress(70, 100, "Budget processing complete")
```

**Option B: Celery Subtasks (distributed, better for very large files)**

```python
from celery import group

@celery_app.task(bind=True, name="app.tasks.postprocess.process_heads_task")
def process_heads_task(self, run_id: str, temp_path: str, model_type: str, quick_mode: bool):
    """Celery task for processing heads file."""
    # ... processing logic ...
    return heads_result

@celery_app.task(bind=True, name="app.tasks.postprocess.process_budget_task")
def process_budget_task(self, run_id: str, temp_path: str, model_type: str, quick_mode: bool):
    """Celery task for processing budget file."""
    # ... processing logic ...
    return budget_result

@celery_app.task(bind=True, name="app.tasks.postprocess.extract_results")
def extract_results(self, run_id: str, project_id: str, quick_mode: bool = True):
    """Main task that orchestrates parallel subtasks."""

    # ... download files to shared storage (not temp dir) ...

    # Launch parallel subtasks
    job = group(
        process_heads_task.s(run_id, shared_path, model_type, quick_mode),
        process_budget_task.s(run_id, shared_path, model_type, quick_mode),
    )

    result = job.apply_async()
    heads_result, budget_result = result.get()  # Wait for both

    # ... combine and upload results ...
```

### Benefits
| Metric | Before (Sequential) | After (Parallel) |
|--------|---------------------|------------------|
| Total time (HDS: 2min, CBC: 3min) | 5 minutes | ~3 minutes |
| CPU utilization | Single core | Multi-core |
| Progress visibility | One at a time | Both in progress |

### Caveats
- ThreadPoolExecutor: Still limited by Python GIL for CPU-bound work (but I/O is the bottleneck anyway)
- Celery subtasks: Requires shared filesystem between workers, more complex error handling
- Memory usage doubles if both files loaded simultaneously

---

## Optimization 3: On-Demand Timestep Processing

### What It Does
Instead of processing all (or selected) timesteps upfront, only process timesteps when the user actually requests them via the UI. Store the raw HDS/CBC files and process slices on-demand.

### Implementation

**New API Endpoint: `GET /results/heads/slice`**

```python
@router.get("/projects/{project_id}/runs/{run_id}/results/heads/slice")
async def get_head_slice_on_demand(
    project_id: UUID,
    run_id: UUID,
    layer: int,
    kper: int,
    kstp: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Extract a single head slice on-demand from the binary HDS file.

    Caches the result for subsequent requests.
    """
    # Check cache first
    cache_key = f"heads_L{layer}_SP{kper}_TS{kstp}"
    cached = await get_cached_slice(project_id, run_id, cache_key)
    if cached:
        return cached

    # Not cached - extract from HDS file
    storage = get_storage_service()
    run = await get_run(db, run_id)

    # Stream HDS file (or use range requests if storage supports it)
    hds_path = f"{run.results_path}/output.hds"

    # Option A: Download to temp and use mmap
    with tempfile.NamedTemporaryFile() as tmp:
        file_data = storage.download_file(settings.minio_bucket_models, hds_path)
        tmp.write(file_data)
        tmp.flush()

        slice_data = extract_single_slice(tmp.name, layer, kper, kstp)

    # Cache the result
    await cache_slice(project_id, run_id, cache_key, slice_data)

    return slice_data
```

**Frontend Changes:**

```typescript
// Instead of loading all timesteps upfront, fetch on-demand
const { data: headSlice, isLoading } = useQuery({
  queryKey: ['head-slice', projectId, runId, layer, kper, kstp],
  queryFn: () => resultsApi.getHeadSlice(projectId, runId, layer, kper, kstp),
  staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  enabled: !!runId && kper !== undefined,
})
```

**Caching Strategy:**

```python
# Redis cache for frequently accessed slices
async def cache_slice(project_id: str, run_id: str, key: str, data: dict, ttl: int = 3600):
    """Cache a head slice in Redis with 1-hour TTL."""
    redis_key = f"slice:{project_id}:{run_id}:{key}"
    await redis.setex(redis_key, ttl, json.dumps(data))

async def get_cached_slice(project_id: str, run_id: str, key: str) -> Optional[dict]:
    """Retrieve cached slice from Redis."""
    redis_key = f"slice:{project_id}:{run_id}:{key}"
    data = await redis.get(redis_key)
    return json.loads(data) if data else None
```

### Benefits
| Metric | Before (Pre-process all) | After (On-demand) |
|--------|--------------------------|-------------------|
| Initial post-processing time | 10-30 minutes | 1-2 minutes |
| First visualization available | After all processing | Immediately |
| Storage for processed data | All timesteps as JSON | Only viewed timesteps |
| Time to view timestep 50 | Same (already processed) | 2-5 seconds (first), instant (cached) |

### Caveats
- First access to each timestep has latency (2-5 seconds)
- Need to keep raw HDS/CBC files in storage longer
- More complex caching logic
- Range requests on MinIO would be ideal but adds complexity

---

## Optimization 4: Streaming/Chunked Downloads

### What It Does
Instead of downloading entire GB files before processing, stream them in chunks and process as data arrives.

### Implementation

```python
import io

def stream_and_process_hds(storage, hds_object_path: str, timesteps_to_process: list):
    """
    Stream HDS file in chunks, building timestep index and extracting
    requested timesteps without loading entire file.
    """
    CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB chunks

    # First pass: scan headers to build timestep index
    timestep_index = {}
    offset = 0

    for chunk in storage.stream_object(settings.minio_bucket_models, hds_object_path, CHUNK_SIZE):
        # Process chunk to find timestep headers
        # ... header parsing logic ...
        pass

    # Second pass: fetch only the chunks containing requested timesteps
    for kstp, kper in timesteps_to_process:
        target_offset = timestep_index[(kstp, kper)]

        # Use range request to fetch just this timestep's data
        data = storage.download_range(
            settings.minio_bucket_models,
            hds_object_path,
            start=target_offset,
            end=target_offset + record_size + header_size
        )

        # Process single timestep
        yield process_timestep_data(data, kstp, kper)
```

**MinIO Range Request Support:**

```python
def download_range(self, bucket: str, object_name: str, start: int, end: int) -> bytes:
    """Download a byte range from an object (HTTP Range header)."""
    response = self.client.get_object(
        bucket,
        object_name,
        offset=start,
        length=end - start
    )
    return response.read()
```

### Benefits
| Metric | Before | After |
|--------|--------|-------|
| Memory for 5GB file | 5GB+ | 64MB (chunk size) |
| Network transfer | Entire file | Only needed chunks |
| Time to first result | After full download | After first chunk |

---

## Implementation Priority

Based on effort vs. impact:

| Priority | Optimization | Effort | Impact | Recommended |
|----------|--------------|--------|--------|-------------|
| 1 | **On-Demand Processing** | Medium | High | Yes - biggest UX improvement |
| 2 | **Parallel Processing** | Low | Medium | Yes - easy win |
| 3 | **Memory-Mapped Reading** | Medium | Medium | Yes - for very large files |
| 4 | **Streaming Downloads** | High | Medium | Later - complex, MinIO-specific |

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. ✅ Quick mode (already implemented)
2. Parallel processing with ThreadPoolExecutor
3. Better progress reporting

### Phase 2: On-Demand Processing (3-5 days)
1. New API endpoint for on-demand slice extraction
2. Redis caching for extracted slices
3. Frontend changes to fetch slices on-demand
4. Keep raw HDS/CBC in storage for on-demand access

### Phase 3: Large File Support (2-3 days)
1. Memory-mapped file reading for files > 500MB
2. File size detection to choose processing strategy
3. Timestep index caching to speed up subsequent access

### Phase 4: Advanced (Future)
1. Streaming/chunked downloads
2. Range requests for MinIO
3. Background full-processing after quick results available

---

## Quick Win: Add Parallel Processing Now

Here's a minimal change to add parallel processing immediately:

```python
# In backend/app/tasks/postprocess.py

from concurrent.futures import ThreadPoolExecutor

# Inside extract_results(), replace sequential processing:

# BEFORE:
# heads_result = _process_heads(...)
# budget_result = _process_budget(...)

# AFTER:
with ThreadPoolExecutor(max_workers=2) as executor:
    heads_future = executor.submit(
        _process_heads, local_files, model_type, project, quick_mode, None
    )
    budget_future = executor.submit(
        _process_budget, local_files, model_type, quick_mode, None
    )

    heads_result = heads_future.result()
    update_progress(50, 100, "Head data processed")

    budget_result = budget_future.result()
    update_progress(70, 100, "Budget data processed")
```

This alone can reduce post-processing time by 30-40% with minimal code changes.
