"""Zone budget API endpoint using FloPy ZoneBudget."""

import asyncio
import hashlib
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.base import get_db
from app.models.project import Project, Run, RunStatus
from app.services.storage import get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/projects/{project_id}/runs/{run_id}/results",
    tags=["results"],
)
settings = get_settings()


IGNORED_BUDGET_TERMS = {
    'PERCENT DISCREPANCY', 'PERCENT_DISCREPANCY',
    'IN-OUT', 'IN - OUT', 'IN_-_OUT',
    'NUMBER OF TIME STEPS', 'MULTIPLIER FOR DELT', 'INITIAL TIME STEP SIZE',
    'TOTAL', 'TOTAL_IN', 'TOTAL_OUT', 'TOTAL IN', 'TOTAL OUT',
    # Internal cell-to-cell flows (not real budget sources/sinks)
    'FLOW_RIGHT_FACE', 'FLOW RIGHT FACE',
    'FLOW_FRONT_FACE', 'FLOW FRONT FACE',
    'FLOW_LOWER_FACE', 'FLOW LOWER FACE',
    'FLOW_JA_FACE', 'FLOW JA FACE',
}

# MF6 CBC record types to skip (internal flows / ancillary data, not budget terms)
_MF6_SKIP_RECORDS = {'FLOW-JA-FACE', 'DATA-SPDIS', 'DATA-SAT', 'DATA-STOSS',
                     'DATA-STOSY'}


def _compute_mf6_zone_budget(
    cbc_path: str,
    zone_array: np.ndarray,
    zone_name_to_num: dict[str, int],
    nlay: int,
    nrow: int,
    ncol: int,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    kstpkper_filter: list | None = None,
) -> dict:
    """
    Compute zone budget for MF6 CBC files manually.

    FloPy's classic ZoneBudget doesn't handle MF6's mixed CBC format
    (3D arrays + sparse node-based records). This function processes
    each record type directly.

    Args:
        cbc_path: Path to the CBC file.
        zone_array: 3D numpy array (nlay, nrow, ncol) with zone numbers.
        zone_name_to_num: Mapping of zone names to zone numbers.
        nlay, nrow, ncol: Grid dimensions.

    Returns:
        Dict with zone_names, columns, and records.
    """
    import flopy.utils

    num_to_name = {v: k for k, v in zone_name_to_num.items()}
    zone_nums = sorted(zone_name_to_num.values())
    zone_cols = [f"ZONE_{zn}" for zn in zone_nums]

    # Flatten zone_array for node-based lookups (1-based node indexing)
    zone_flat = zone_array.ravel()  # shape (nlay * nrow * ncol,)
    num_cells = len(zone_flat)
    max_zone_num = max(zone_nums) if zone_nums else 0

    cbc = flopy.utils.CellBudgetFile(str(cbc_path))
    record_names = [n.decode().strip() for n in cbc.get_unique_record_names()]
    kstpkper_list = cbc.get_kstpkper()

    if kstpkper_filter is not None:
        kstpkper_list = [k for k in kstpkper_list if k in kstpkper_filter]

    records = []
    total_steps = len(kstpkper_list)
    for step_idx, kstpkper in enumerate(kstpkper_list):
        if progress_callback:
            progress_callback(step_idx, total_steps, f"Processing timestep {step_idx + 1}/{total_steps}")
        kstp, kper = int(kstpkper[0]), int(kstpkper[1])

        for recname in record_names:
            if recname in _MF6_SKIP_RECORDS:
                continue

            try:
                data_list = cbc.get_data(text=recname, kstpkper=kstpkper)
            except Exception:
                continue

            # Accumulate inflow/outflow per zone using vectorized numpy ops.
            # bincount length must cover all zone numbers (0..max_zone_num).
            bc_len = max_zone_num + 1
            zone_in_arr = np.zeros(bc_len, dtype=np.float64)
            zone_out_arr = np.zeros(bc_len, dtype=np.float64)

            for data in data_list:
                if data.dtype.names and 'q' in data.dtype.names:
                    # Sparse/compact record: (node, node2, q, ...)
                    nodes = np.asarray(data['node'], dtype=np.intp) - 1  # 1-based → 0-based
                    q = np.asarray(data['q'], dtype=np.float64)
                    valid = (nodes >= 0) & (nodes < num_cells)
                    nodes_v = nodes[valid]
                    q_v = q[valid]
                    zones_v = zone_flat[nodes_v]
                    zoned = zones_v > 0
                    zones_z = zones_v[zoned]
                    q_z = q_v[zoned]
                    pos = q_z > 0
                    if np.any(pos):
                        zone_in_arr += np.bincount(zones_z[pos], weights=q_z[pos], minlength=bc_len)
                    neg = q_z < 0
                    if np.any(neg):
                        zone_out_arr += np.bincount(zones_z[neg], weights=np.abs(q_z[neg]), minlength=bc_len)
                else:
                    # Full 3D array record
                    arr = np.asarray(data, dtype=np.float64)
                    if arr.ndim == 3 and arr.shape == (nlay, nrow, ncol):
                        flat_data = arr.ravel()
                    elif arr.ndim == 3 and arr.shape[0] == nlay:
                        flat_data = arr.ravel()
                        if len(flat_data) != num_cells:
                            continue
                    else:
                        continue
                    zoned = zone_flat > 0
                    zones_z = zone_flat[zoned]
                    vals = flat_data[zoned]
                    pos = vals > 0
                    if np.any(pos):
                        zone_in_arr += np.bincount(zones_z[pos], weights=vals[pos], minlength=bc_len)
                    neg = vals < 0
                    if np.any(neg):
                        zone_out_arr += np.bincount(zones_z[neg], weights=np.abs(vals[neg]), minlength=bc_len)

            cleaned = recname.upper().replace(' ', '_').replace('-', '_')

            # Build FROM_ / TO_ records from vectorized accumulators
            from_rec = {'name': f'FROM_{cleaned}', 'kper': kper, 'kstp': kstp}
            to_rec = {'name': f'TO_{cleaned}', 'kper': kper, 'kstp': kstp}
            for zn in zone_nums:
                from_rec[f'ZONE_{zn}'] = float(zone_in_arr[zn])
                to_rec[f'ZONE_{zn}'] = float(zone_out_arr[zn])
            records.append(from_rec)
            records.append(to_rec)

    cbc.close()

    all_zone_names = [num_to_name[zn] for zn in zone_nums]
    columns = ['name', 'kper', 'kstp'] + zone_cols

    return {
        'zone_names': all_zone_names,
        'columns': columns,
        'records': records,
    }


def _parse_budget_lines_two_col(text: str) -> list[tuple[str, float, float]]:
    """
    Parse budget lines from a two-column (cumulative | rates) block.

    Each line has the form:
        NAME = cumulative_value          NAME = rate_value
    or single-column:
        NAME = value

    Returns list of (name, cumulative_value, rate_value).
    If only one column, rate_value == cumulative_value.
    """
    results = []
    # Match lines like:  NAME = 1234.0   NAME = 5678.0
    # or single:         NAME = 1234.0
    two_col = re.compile(
        r'^\s+([A-Z][A-Z0-9_ -]+?)\s*=\s*([+-]?\d+\.?\d*(?:[Ee][+-]?\d+)?)'
        r'(?:\s+([A-Z][A-Z0-9_ -]+?)\s*=\s*([+-]?\d+\.?\d*(?:[Ee][+-]?\d+)?))?',
        re.MULTILINE
    )
    for m in two_col.finditer(text):
        name1 = m.group(1).strip()
        val1 = float(m.group(2))
        if m.group(3) and m.group(4):
            val2 = float(m.group(4))
        else:
            val2 = val1
        results.append((name1, val1, val2))
    return results


def parse_budget_from_listing(listing_content: str) -> dict:
    """
    Parse water budget summary from MODFLOW listing file.

    Handles both MF6 ("VOLUME BUDGET") and MF2005/NWT ("VOLUMETRIC BUDGET")
    header formats, and two-column layouts (cumulative | rates).
    Prefers RATES column when available (more useful for time-series).

    Returns dict with records in zone budget format, or empty dict if parsing fails.
    """
    records = []

    # Find all budget sections — match both "VOLUME BUDGET" (MF6) and "VOLUMETRIC BUDGET" (MF2005/NWT)
    sections = list(re.finditer(
        r'VOLUM\w*\s+BUDGET FOR ENTIRE MODEL.*?'
        r'(?=VOLUM\w*\s+BUDGET FOR ENTIRE MODEL|HEAD WILL BE SAVED|$)',
        listing_content,
        re.IGNORECASE | re.DOTALL
    ))

    for section in sections:
        section_text = section.group()

        # Extract stress period and time step
        sp_match = re.search(r'STRESS PERIOD\s+(\d+)', section_text, re.IGNORECASE)
        ts_match = re.search(r'TIME STEP\s+(\d+)', section_text, re.IGNORECASE)

        if not sp_match or not ts_match:
            continue

        kper = int(sp_match.group(1))
        kstp = int(ts_match.group(1))

        # Detect if "RATES" column header is present (two-column format)
        has_rates = bool(re.search(r'RATES\s+FOR\s+THIS\s+TIME\s+STEP', section_text, re.IGNORECASE))

        # Find IN section: from "IN:" line, past dashes, to "TOTAL IN" or "OUT:" line
        # Use [^\n]* to consume rest of two-column header line (handles second IN: on same line)
        in_match = re.search(
            r'\bIN:[^\n]*\n[^\n]*-+[^\n]*\n(.*?)(?=TOTAL\s+IN|OUT:)',
            section_text,
            re.IGNORECASE | re.DOTALL
        )

        # Find OUT section: from "OUT:" line, past dashes, to "TOTAL OUT" or "IN - OUT" or "PERCENT"
        out_match = re.search(
            r'\bOUT:[^\n]*\n[^\n]*-+[^\n]*\n(.*?)(?=TOTAL\s+OUT|\bIN\s*-\s*OUT\b|PERCENT)',
            section_text,
            re.IGNORECASE | re.DOTALL
        )

        if in_match:
            for name, cumul, rate in _parse_budget_lines_two_col(in_match.group(1)):
                cleaned = name.upper().replace(' ', '_').replace('-', '_')
                if cleaned in IGNORED_BUDGET_TERMS or name.upper() in IGNORED_BUDGET_TERMS:
                    continue
                # Prefer rates when two-column format is present
                value = rate if has_rates else cumul
                records.append({
                    'name': f'FROM_{cleaned}',
                    'kper': kper,
                    'kstp': kstp,
                    'ZONE_1': abs(value)
                })

        if out_match:
            for name, cumul, rate in _parse_budget_lines_two_col(out_match.group(1)):
                cleaned = name.upper().replace(' ', '_').replace('-', '_')
                if cleaned in IGNORED_BUDGET_TERMS or name.upper() in IGNORED_BUDGET_TERMS:
                    continue
                value = rate if has_rates else cumul
                records.append({
                    'name': f'TO_{cleaned}',
                    'kper': kper,
                    'kstp': kstp,
                    'ZONE_1': abs(value)
                })

    if not records:
        return {}

    return {
        'zone_names': ['Entire Model'],
        'columns': ['name', 'kper', 'kstp', 'ZONE_1'],
        'records': records,
        'source': 'listing_file',
        'warning': 'Budget parsed from listing file (flow rates). '
                   'Enable "Save Water Budget" when running simulation for detailed cell-by-cell budget data.'
    }


class ZoneBudgetRequest(BaseModel):
    zone_layers: dict[str, dict[str, list[int]]]
    # layer_index (str) -> { zone_name -> [cell_indices (0-based)] }
    # e.g. {"0": {"Zone 1": [0,1,2], "Zone 2": [10,11]}, "1": {"Zone 1": [0,1,2]}}


def _budget_json_to_zone_format(budget: dict) -> dict | None:
    """Convert post-processed budget.json into zone-budget record format."""
    periods = budget.get("periods", {})
    if not periods:
        return None

    records = []
    for _period_key, period_data in periods.items():
        kper = period_data.get("kper", 0)
        kstp = period_data.get("kstp", 0)

        in_flows = period_data.get("in", {})
        for name, value in in_flows.items():
            cleaned = name.upper().replace(" ", "_").replace("-", "_")
            if cleaned in IGNORED_BUDGET_TERMS or name.upper() in IGNORED_BUDGET_TERMS:
                continue
            records.append({
                "name": f"FROM_{cleaned}",
                "kper": kper,
                "kstp": kstp,
                "ZONE_1": value,
            })

        out_flows = period_data.get("out", {})
        for name, value in out_flows.items():
            cleaned = name.upper().replace(" ", "_").replace("-", "_")
            if cleaned in IGNORED_BUDGET_TERMS or name.upper() in IGNORED_BUDGET_TERMS:
                continue
            records.append({
                "name": f"TO_{cleaned}",
                "kper": kper,
                "kstp": kstp,
                "ZONE_1": abs(value) if value else 0,
            })

    if not records:
        return None

    return {
        "zone_names": ["Entire Model"],
        "columns": ["name", "kper", "kstp", "ZONE_1"],
        "records": records,
    }


# Maximum CBC file size (in bytes) to attempt downloading for FloPy ZoneBudget.
# Files larger than this are too big to process in the API container.
_MAX_CBC_SIZE_FOR_ZONEBUDGET = 2 * 1024 * 1024 * 1024  # 2 GB


@router.get("/zone-budget/model")
async def get_model_zone_budget(
    project_id: UUID,
    run_id: UUID,
    refresh: bool = False,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Compute zone budget for the entire model as a single zone.

    Uses pre-processed budget.json from post-processing when available.
    Falls back to FloPy ZoneBudget on the CBC file for small files,
    or listing file parsing as a last resort.

    Query params:
    - refresh: If true, ignore cached result and recompute from source data.
    """
    # Validate run
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found in project {project_id}",
        )

    if run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run is not completed (status: {run.status.value})",
        )

    if not run.results_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results available for this run",
        )

    storage = get_storage_service()

    # Check for cached zone budget result (skip if refresh requested)
    cache_obj = f"{run.results_path}/processed/zone_budget_model.json"
    if not refresh and storage.object_exists(settings.minio_bucket_models, cache_obj):
        data = storage.download_file(settings.minio_bucket_models, cache_obj)
        return json.loads(data)

    # Try pre-processed budget.json first (generated during post-processing)
    budget_obj = f"{run.results_path}/processed/budget.json"
    if storage.object_exists(settings.minio_bucket_models, budget_obj):
        try:
            data = storage.download_file(settings.minio_bucket_models, budget_obj)
            budget = json.loads(data)
            zone_result = _budget_json_to_zone_format(budget)
            if zone_result and zone_result.get("records"):
                # Cache for future requests
                storage.upload_bytes(
                    settings.minio_bucket_models,
                    cache_obj,
                    json.dumps(zone_result).encode("utf-8"),
                    content_type="application/json",
                )
                return zone_result
        except Exception as e:
            logger.debug(f"Failed to convert budget.json to zone format: {e}")

    # Find CBC file and listing file
    output_objects = list(storage.list_objects(
        settings.minio_bucket_models,
        prefix=run.results_path,
        recursive=True,
    ))

    cbc_obj = None
    lst_obj = None
    for obj_name in output_objects:
        lower = obj_name.lower()
        if lower.endswith(".cbc") or lower.endswith(".bud") or lower.endswith(".cbb"):
            cbc_obj = obj_name
        if lower.endswith(".lst") or lower.endswith(".list"):
            lst_obj = obj_name

    # Try FloPy ZoneBudget on CBC file if it's small enough
    if cbc_obj:
        try:
            stat = storage.client.stat_object(settings.minio_bucket_models, cbc_obj)
            cbc_size = stat.size
        except Exception:
            cbc_size = None

        if cbc_size and cbc_size <= _MAX_CBC_SIZE_FOR_ZONEBUDGET:
            import flopy.utils

            # Get project for model type
            stmt_proj = select(Project).where(Project.id == project_id)
            result_proj = await db.execute(stmt_proj)
            project = result_proj.scalar_one_or_none()
            model_type = project.model_type.value if project and project.model_type else "mf6"
            is_usg = model_type == "mfusg"

            summary_obj = f"{run.results_path}/results_summary.json"
            if storage.object_exists(settings.minio_bucket_models, summary_obj):
                summary = json.loads(
                    storage.download_file(settings.minio_bucket_models, summary_obj)
                )
                meta = summary.get("metadata", {})
                nlay = meta.get("nlay", 1)
                nrow = meta.get("nrow", 1)
                ncol = meta.get("ncol", 1)

                with tempfile.TemporaryDirectory() as temp_dir:
                    local_cbc = Path(temp_dir) / "output.cbc"
                    storage.download_to_file(settings.minio_bucket_models, cbc_obj, local_cbc)

                    try:
                        is_mf6 = model_type == "mf6"

                        if is_mf6:
                            zone_array = np.ones((nlay, nrow, ncol), dtype=int)
                            response = _compute_mf6_zone_budget(
                                str(local_cbc), zone_array,
                                {"Entire Model": 1}, nlay, nrow, ncol,
                            )
                        elif is_usg:
                            total_cells = ncol
                            zone_array = np.ones((nlay, total_cells), dtype=int)
                            cbc = flopy.utils.CellBudgetFile(
                                str(local_cbc), precision="double"
                            )
                            zb = flopy.utils.ZoneBudget(cbc, zone_array)
                            df = zb.get_budget()
                            cbc.close()
                            records = df.to_dict(orient="records")
                            for rec in records:
                                for k, v in rec.items():
                                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                                        rec[k] = None
                            response = {
                                "zone_names": ["Entire Model"],
                                "columns": list(df.columns) if len(records) > 0 else [],
                                "records": records,
                            }
                        else:
                            zone_array = np.ones((nlay, nrow, ncol), dtype=int)
                            cbc = flopy.utils.CellBudgetFile(str(local_cbc))
                            zb = flopy.utils.ZoneBudget(cbc, zone_array)
                            df = zb.get_budget()
                            cbc.close()
                            records = df.to_dict(orient="records")
                            for rec in records:
                                for k, v in rec.items():
                                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                                        rec[k] = None
                            response = {
                                "zone_names": ["Entire Model"],
                                "columns": list(df.columns) if len(records) > 0 else [],
                                "records": records,
                            }

                        # Cache result
                        storage.upload_bytes(
                            settings.minio_bucket_models,
                            cache_obj,
                            json.dumps(response).encode("utf-8"),
                            content_type="application/json",
                        )
                        return response

                    except Exception as e:
                        logger.debug(f"FloPy ZoneBudget failed, falling through to listing file: {e}")

    # Listing file fallback
    if lst_obj:
        try:
            with tempfile.TemporaryDirectory() as lst_tmp:
                local_lst = Path(lst_tmp) / "output.lst"
                storage.download_to_file(
                    settings.minio_bucket_models, lst_obj, local_lst
                )
                lst_content = local_lst.read_text(
                    encoding='utf-8', errors='ignore'
                )
            listing_budget = parse_budget_from_listing(lst_content)

            if listing_budget and listing_budget.get('records'):
                storage.upload_bytes(
                    settings.minio_bucket_models,
                    cache_obj,
                    json.dumps(listing_budget).encode("utf-8"),
                    content_type="application/json",
                )
                return listing_budget
        except Exception as e:
            logger.debug(f"Listing file budget parse failed: {e}")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="No budget data available. "
               "Enable 'Save Water Budget' option when running simulation to generate budget data.",
    )


def compute_zone_hash(zone_layers: dict, quick_mode: bool = False) -> str:
    """Compute a stable hash for zone configuration + mode for cache keying."""
    canonical = json.dumps(zone_layers, sort_keys=True, separators=(',', ':'))
    prefix = "quick:" if quick_mode else "full:"
    return hashlib.sha256((prefix + canonical).encode()).hexdigest()[:16]


def _sync_compute_zone_budget(
    cbc_obj: str,
    model_type: str,
    zone_layers: dict[str, dict[str, list[int]]],
    nlay: int,
    nrow: int,
    ncol: int,
    quick_mode: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Synchronous zone budget computation — safe to call from thread or Celery task.

    Downloads CBC from MinIO, builds zone array, runs FloPy/custom MF6 computation.
    Returns zone budget result dict.
    """
    import flopy.utils

    storage = get_storage_service()
    is_mf6 = model_type == "mf6"
    is_usg = model_type == "mfusg"

    # Collect all zone names across layers to assign consistent zone numbers
    all_zone_names: list[str] = []
    for layer_zones_data in zone_layers.values():
        for zn in layer_zones_data:
            if zn not in all_zone_names:
                all_zone_names.append(zn)
    zone_name_to_num = {name: idx + 1 for idx, name in enumerate(all_zone_names)}

    # Build zone array from request
    if is_usg:
        total_cells = ncol
        zone_array = np.zeros((nlay, total_cells), dtype=int)
        for layer_str, layer_zones_data in zone_layers.items():
            lay = int(layer_str)
            if lay < 0 or lay >= nlay:
                continue
            for zone_name, cell_indices in layer_zones_data.items():
                znum = zone_name_to_num[zone_name]
                for ci in cell_indices:
                    if 0 <= ci < total_cells:
                        zone_array[lay, ci] = znum
    else:
        zone_array = np.zeros((nlay, nrow, ncol), dtype=int)
        for layer_str, layer_zones_data in zone_layers.items():
            lay = int(layer_str)
            if lay < 0 or lay >= nlay:
                continue
            for zone_name, cell_indices in layer_zones_data.items():
                znum = zone_name_to_num[zone_name]
                for ci in cell_indices:
                    r = ci // ncol
                    c = ci % ncol
                    if 0 <= r < nrow and 0 <= c < ncol:
                        zone_array[lay, r, c] = znum

    with tempfile.TemporaryDirectory() as temp_dir:
        local_cbc = Path(temp_dir) / "output.cbc"
        storage.download_to_file(settings.minio_bucket_models, cbc_obj, local_cbc)

        if is_mf6:
            # Determine kstpkper filter for quick mode
            kstpkper_filter = None
            if quick_mode:
                cbc_file = flopy.utils.CellBudgetFile(str(local_cbc))
                all_kstpkper = cbc_file.get_kstpkper()
                cbc_file.close()
                if all_kstpkper:
                    kstpkper_filter = [all_kstpkper[-1]]

            return _compute_mf6_zone_budget(
                str(local_cbc), zone_array, zone_name_to_num,
                nlay, nrow, ncol,
                progress_callback=progress_callback,
                kstpkper_filter=kstpkper_filter,
            )

        if is_usg:
            cbc = flopy.utils.CellBudgetFile(
                str(local_cbc), precision="double"
            )
        else:
            cbc = flopy.utils.CellBudgetFile(str(local_cbc))

        zb = flopy.utils.ZoneBudget(cbc, zone_array)
        budget_array = zb.get_budget()
        cbc.close()

        # Convert numpy record array to list of dicts
        col_names = list(budget_array.dtype.names) if budget_array.dtype.names else []
        records = []
        for row in budget_array:
            rec = {}
            for col in col_names:
                val = row[col]
                if isinstance(val, (np.integer,)):
                    rec[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    fval = float(val)
                    rec[col] = None if (np.isnan(fval) or np.isinf(fval)) else fval
                elif isinstance(val, (bytes,)):
                    rec[col] = val.decode("utf-8", errors="replace").strip()
                else:
                    rec[col] = str(val).strip() if isinstance(val, np.str_) else val
            records.append(rec)

    return {
        "zone_names": all_zone_names,
        "columns": col_names,
        "records": records,
    }


def _find_cbc_object(storage, results_path: str) -> str | None:
    """Find the CBC file object in MinIO results."""
    output_objects = storage.list_objects(
        settings.minio_bucket_models,
        prefix=results_path,
        recursive=True,
    )
    for obj_name in output_objects:
        lower = obj_name.lower()
        if lower.endswith(".cbc") or lower.endswith(".bud") or lower.endswith(".cbb"):
            return obj_name
    return None


def _load_grid_dimensions(storage, results_path: str) -> dict:
    """Load grid dimensions from results summary in MinIO."""
    summary_obj = f"{results_path}/results_summary.json"
    if not storage.object_exists(settings.minio_bucket_models, summary_obj):
        raise ValueError("Results summary not available")
    summary = json.loads(
        storage.download_file(settings.minio_bucket_models, summary_obj)
    )
    meta = summary.get("metadata", {})
    return {
        "nlay": meta.get("nlay", 1),
        "nrow": meta.get("nrow", 1),
        "ncol": meta.get("ncol", 1),
    }


@router.post("/zone-budget")
async def compute_zone_budget(
    project_id: UUID,
    run_id: UUID,
    request: ZoneBudgetRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Compute zone budget from CBC file using FloPy ZoneBudget (synchronous fallback).

    Request body contains per-layer zone assignments as named lists of
    cell indices. Returns budget DataFrame as JSON records.
    Runs computation in a thread to avoid blocking the event loop.
    """
    # Validate run
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found in project {project_id}",
        )

    if run.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run is not completed (status: {run.status.value})",
        )

    if not run.results_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results available for this run",
        )

    # Get project for model type
    stmt_proj = select(Project).where(Project.id == project_id)
    result_proj = await db.execute(stmt_proj)
    project = result_proj.scalar_one_or_none()
    model_type = project.model_type.value if project and project.model_type else "mf6"

    storage = get_storage_service()

    cbc_obj = _find_cbc_object(storage, run.results_path)
    if not cbc_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cell budget file (.cbc/.bud/.cbb) not found in results",
        )

    try:
        dims = _load_grid_dimensions(storage, run.results_path)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    try:
        result_data = await asyncio.to_thread(
            _sync_compute_zone_budget,
            cbc_obj,
            model_type,
            request.zone_layers,
            dims["nlay"],
            dims["nrow"],
            dims["ncol"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error computing zone budget: {e}",
        )

    return result_data


# ---------------------------------------------------------------------------
# Async compute / status / result endpoints (Phase 2)
# ---------------------------------------------------------------------------

class ZoneBudgetComputeRequest(BaseModel):
    zone_layers: dict[str, dict[str, list[int]]]
    quick_mode: bool = False


@router.post("/zone-budget/compute")
async def compute_zone_budget_async(
    project_id: UUID,
    run_id: UUID,
    request: ZoneBudgetComputeRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Dispatch zone budget computation to Celery worker.

    Returns cached result immediately on cache hit, or a task_id for polling.
    """
    from uuid import uuid4

    # Validate run
    stmt = select(Run).where(Run.id == run_id, Run.project_id == project_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Run {run_id} not found in project {project_id}")
    if run.status != RunStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Run is not completed (status: {run.status.value})")
    if not run.results_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No results available for this run")

    storage = get_storage_service()

    # Check MinIO cache
    zone_hash = compute_zone_hash(request.zone_layers, request.quick_mode)
    cache_obj = f"{run.results_path}/processed/zone_budget_{zone_hash}.json"
    if storage.object_exists(settings.minio_bucket_models, cache_obj):
        try:
            data = json.loads(storage.download_file(settings.minio_bucket_models, cache_obj))
            return {"status": "completed", "result": data, "cached": True}
        except Exception as e:
            logger.debug(f"Cache corrupt for zone budget {zone_hash}, recomputing: {e}")

    # Dispatch Celery task
    task_id = str(uuid4())

    # Set initial progress in Redis
    try:
        import time
        from app.services.redis_manager import get_sync_client
        rc = get_sync_client()
        rc.hset(f"zb:progress:{task_id}", mapping={
            "status": "queued",
            "progress": "0",
            "message": "Queued for computation...",
            "result_path": "",
            "error": "",
            "created_at": str(time.time()),
        })
        rc.expire(f"zb:progress:{task_id}", 3600)
    except Exception as e:
        logger.debug(f"Failed to set initial zone budget progress in Redis: {e}")

    from app.tasks.zonebudget import compute_zone_budget_task
    compute_zone_budget_task.delay(
        task_id=task_id,
        run_id=str(run_id),
        project_id=str(project_id),
        zone_layers=request.zone_layers,
        quick_mode=request.quick_mode,
    )

    return {"status": "queued", "task_id": task_id}


@router.get("/zone-budget/status/{task_id}")
async def get_zone_budget_status(
    project_id: UUID,
    run_id: UUID,
    task_id: str,
) -> dict:
    """
    Poll zone budget computation progress.

    Reads from Redis hash `zb:progress:{task_id}`.
    """
    from app.services.redis_manager import get_sync_client

    try:
        rc = get_sync_client()
        key = f"zb:progress:{task_id}"
        data = rc.hgetall(key)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Redis unavailable: {e}")

    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No progress found for task {task_id}")

    task_status = data.get("status", "unknown")

    # Detect stale "queued" tasks — if queued for >60s, the worker likely
    # never received the task (e.g. unregistered task, broker issue)
    if task_status == "queued":
        import time
        created_at = data.get("created_at", "")
        is_stale = False
        if created_at:
            try:
                is_stale = (time.time() - float(created_at)) > 60
            except (ValueError, TypeError):
                pass
        else:
            # No created_at means pre-fix task — treat as stale
            is_stale = True
        if is_stale:
            error_msg = "Task was not picked up by the worker. The worker may need to be restarted."
            rc.hset(f"zb:progress:{task_id}", mapping={
                "status": "failed",
                "error": error_msg,
            })
            return {
                "status": "failed",
                "progress": 0,
                "message": "",
                "error": error_msg,
            }

    return {
        "status": task_status,
        "progress": int(data.get("progress", "0")),
        "message": data.get("message", ""),
        "error": data.get("error", "") or None,
    }


@router.get("/zone-budget/result/{task_id}")
async def get_zone_budget_result(
    project_id: UUID,
    run_id: UUID,
    task_id: str,
) -> dict:
    """
    Retrieve completed zone budget result.

    Reads `result_path` from Redis progress hash, downloads JSON from MinIO.
    """
    from app.services.redis_manager import get_sync_client

    try:
        rc = get_sync_client()
        key = f"zb:progress:{task_id}"
        data = rc.hgetall(key)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Redis unavailable: {e}")

    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No progress found for task {task_id}")

    task_status = data.get("status", "")
    if task_status != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Task is not completed (status: {task_status})")

    result_path = data.get("result_path", "")
    if not result_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Result path not found in progress data")

    storage = get_storage_service()
    try:
        result_json = json.loads(
            storage.download_file(settings.minio_bucket_models, result_path)
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Failed to retrieve result: {e}")

    return result_json


# ---------------------------------------------------------------------------
# Zone definition persistence (Phase 3)
# ---------------------------------------------------------------------------

zone_def_router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["zone-definitions"],
)


class ZoneDefinitionSave(BaseModel):
    name: str
    zone_layers: dict[str, dict[str, list[int]]]
    num_zones: int = 6


@zone_def_router.get("/zone-definitions")
async def list_zone_definitions(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """List saved zone definitions for a project."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Project {project_id} not found")
    if not project.storage_path:
        return []

    storage = get_storage_service()
    prefix = f"{project.storage_path}/zone_definitions/"
    try:
        objects = list(storage.list_objects(
            settings.minio_bucket_models, prefix=prefix, recursive=True,
        ))
    except Exception:
        return []

    definitions = []
    for obj_name in objects:
        if not obj_name.endswith(".json"):
            continue
        try:
            data = json.loads(storage.download_file(settings.minio_bucket_models, obj_name))
            definitions.append({
                "name": data.get("name", obj_name.split("/")[-1].replace(".json", "")),
                "num_zones": data.get("num_zones", 0),
                "zone_count": sum(len(z) for z in data.get("zone_layers", {}).values()),
            })
        except Exception:
            continue

    return definitions


@zone_def_router.post("/zone-definitions")
async def save_zone_definition(
    project_id: UUID,
    request: ZoneDefinitionSave,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Save a zone definition for a project."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Project {project_id} not found")
    if not project.storage_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Project has no storage path")

    # Sanitize name for file path
    safe_name = re.sub(r'[^\w\s-]', '', request.name).strip().replace(' ', '_')
    if not safe_name:
        safe_name = "unnamed"

    storage = get_storage_service()
    obj_path = f"{project.storage_path}/zone_definitions/{safe_name}.json"
    payload = {
        "name": request.name,
        "zone_layers": request.zone_layers,
        "num_zones": request.num_zones,
    }
    storage.upload_bytes(
        settings.minio_bucket_models,
        obj_path,
        json.dumps(payload).encode("utf-8"),
        content_type="application/json",
    )

    return {"name": request.name, "saved": True}


@zone_def_router.delete("/zone-definitions/{name}")
async def delete_zone_definition(
    project_id: UUID,
    name: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a saved zone definition."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Project {project_id} not found")
    if not project.storage_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Project has no storage path")

    safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
    storage = get_storage_service()
    obj_path = f"{project.storage_path}/zone_definitions/{safe_name}.json"

    if not storage.object_exists(settings.minio_bucket_models, obj_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Zone definition '{name}' not found")
    try:
        storage.delete_object(settings.minio_bucket_models, obj_path)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Zone definition '{name}' not found")

    return {"name": name, "deleted": True}


@zone_def_router.get("/zone-definitions/{name}")
async def get_zone_definition(
    project_id: UUID,
    name: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Retrieve a saved zone definition by name."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Project {project_id} not found")
    if not project.storage_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Project has no storage path")

    safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
    storage = get_storage_service()
    obj_path = f"{project.storage_path}/zone_definitions/{safe_name}.json"

    try:
        data = json.loads(storage.download_file(settings.minio_bucket_models, obj_path))
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Zone definition '{name}' not found")

    return data
