"""Observations API endpoints for managing multiple observation datasets.

Supports multiple observation sets following pyEMU's add_observations pattern:
- Each project can have multiple observation sets
- Sets can be uploaded, auto-detected from ZIP, or marked from existing files
- PEST calibration can select which sets to include with per-set weights
"""

import csv
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.base import get_db
from app.models.project import Project
from app.schemas.observation import (
    ColumnMapping,
    MarkFileAsObservation,
    ObservationSet,
    ObservationSetCreate,
    ObservationSetData,
    ObservationSetSummary,
    ObservationSetUpdate,
    ObservationWellData,
)
from app.services.storage import get_storage_service

router = APIRouter(
    prefix="/projects/{project_id}/observations",
    tags=["observations"],
)
settings = get_settings()


# ─── Storage Paths ─────────────────────────────────────────────────


def _observations_base_path(project_id: UUID) -> str:
    """Base path for all observation data."""
    return f"projects/{project_id}/observations"


def _sets_index_path(project_id: UUID) -> str:
    """Path to the sets index file."""
    return f"{_observations_base_path(project_id)}/sets.json"


def _set_path(project_id: UUID, set_id: str) -> str:
    """Path to a specific observation set directory."""
    return f"{_observations_base_path(project_id)}/{set_id}"


def _set_data_path(project_id: UUID, set_id: str) -> str:
    """Path to the raw CSV data for a set."""
    return f"{_set_path(project_id, set_id)}/data.csv"


def _set_metadata_path(project_id: UUID, set_id: str) -> str:
    """Path to the metadata JSON for a set."""
    return f"{_set_path(project_id, set_id)}/metadata.json"


def _set_parsed_path(project_id: UUID, set_id: str) -> str:
    """Path to the parsed JSON data for a set."""
    return f"{_set_path(project_id, set_id)}/parsed.json"


# ─── Index Management ──────────────────────────────────────────────


def _load_sets_index(storage, project_id: UUID) -> list[dict]:
    """Load the sets index, or return empty list if not exists."""
    index_path = _sets_index_path(project_id)
    try:
        if storage.object_exists(settings.minio_bucket_models, index_path):
            data = storage.download_file(settings.minio_bucket_models, index_path)
            return json.loads(data)
    except Exception:
        pass
    return []


def _save_sets_index(storage, project_id: UUID, sets: list[dict]) -> None:
    """Save the sets index."""
    index_path = _sets_index_path(project_id)
    storage.upload_bytes(
        settings.minio_bucket_models,
        index_path,
        json.dumps(sets, default=str).encode("utf-8"),
        content_type="application/json",
    )


# ─── CSV Parsing ───────────────────────────────────────────────────


def _parse_observation_csv(
    content: str,
    column_mapping: Optional[ColumnMapping] = None,
) -> tuple[dict, str, list[str]]:
    """
    Parse observation CSV in either long or wide format.

    Args:
        content: CSV content as string
        column_mapping: Optional explicit column mapping

    Returns:
        Tuple of (parsed_data, format_type, well_names)

    Long format:
        WellName,Layer,Row,Col,Time,Head
        MW-01,1,15,22,0.0,25.50

    Wide format:
        Time,MW-01,MW-02,MW-03
        0.0,25.50,12.10,8.70
    """
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError("CSV must have a header row and at least one data row")

    header = [h.strip() for h in rows[0]]
    header_lower = [h.lower() for h in header]

    # Determine format
    if column_mapping:
        # Use explicit mapping
        return _parse_with_mapping(header, rows[1:], column_mapping)
    elif header_lower[0] == "wellname":
        return _parse_long_format(header_lower, rows[1:])
    elif header_lower[0] == "time":
        return _parse_wide_format(header, rows[1:])
    else:
        raise ValueError(
            "Unrecognized CSV format. First column must be 'WellName' (long format) "
            "or 'Time' (wide format), or provide explicit column mapping."
        )


def _parse_with_mapping(
    header: list[str],
    data_rows: list[list[str]],
    mapping: ColumnMapping,
) -> tuple[dict, str, list[str]]:
    """Parse CSV using explicit column mapping."""
    header_lower = [h.lower() for h in header]

    # Find column indices
    def get_col_idx(col_spec: str | int | None) -> Optional[int]:
        if col_spec is None:
            return None
        if isinstance(col_spec, int):
            return None  # Fixed value, not a column
        try:
            return header_lower.index(col_spec.lower())
        except ValueError:
            raise ValueError(f"Column '{col_spec}' not found in CSV")

    def get_fixed_or_col(col_spec: str | int | None, row: list[str]) -> Optional[int]:
        if col_spec is None:
            return None
        if isinstance(col_spec, int):
            return col_spec
        idx = header_lower.index(col_spec.lower())
        return int(row[idx])

    well_idx = get_col_idx(mapping.well_name)
    time_idx = get_col_idx(mapping.time if isinstance(mapping.time, str) else None)
    value_idx = get_col_idx(mapping.value)

    if well_idx is None or time_idx is None or value_idx is None:
        raise ValueError("well_name, time, and value must be column names")

    wells: dict[str, dict] = {}
    n_obs = 0

    for row_data in data_rows:
        if not row_data or all(c.strip() == "" for c in row_data):
            continue

        try:
            well_name = row_data[well_idx].strip()
            if not well_name:
                continue

            time_val = float(row_data[time_idx].strip())
            head_val = float(row_data[value_idx].strip())

            # Get location (convert 1-based to 0-based)
            layer = get_fixed_or_col(mapping.layer, row_data)
            if layer is not None and isinstance(mapping.layer, str):
                layer = layer - 1

            if well_name not in wells:
                well_info: dict = {"layer": layer or 0, "times": [], "heads": []}

                if mapping.node is not None:
                    node = get_fixed_or_col(mapping.node, row_data)
                    if node is not None:
                        well_info["node"] = node - 1 if isinstance(mapping.node, str) else node
                else:
                    row = get_fixed_or_col(mapping.row, row_data)
                    col = get_fixed_or_col(mapping.col, row_data)
                    if row is not None:
                        well_info["row"] = row - 1 if isinstance(mapping.row, str) else row
                    if col is not None:
                        well_info["col"] = col - 1 if isinstance(mapping.col, str) else col

                wells[well_name] = well_info

            wells[well_name]["times"].append(time_val)
            wells[well_name]["heads"].append(head_val)
            n_obs += 1

        except (ValueError, IndexError) as e:
            continue

    return {"wells": wells, "n_observations": n_obs}, "long", list(wells.keys())


def _parse_long_format(header: list[str], data_rows: list[list[str]]) -> tuple[dict, str, list[str]]:
    """Parse long-format observation CSV."""
    has_node = "node" in header
    has_row_col = "row" in header and "col" in header

    if not has_node and not has_row_col:
        raise ValueError(
            "Long format CSV must have either Row/Col or Node columns"
        )

    wells: dict[str, dict] = {}
    n_obs = 0

    for row_data in data_rows:
        if not row_data or all(c.strip() == "" for c in row_data):
            continue

        values = {header[i]: row_data[i].strip() for i in range(min(len(header), len(row_data)))}

        well_name = values.get("wellname", "").strip()
        if not well_name:
            continue

        try:
            layer = int(values["layer"]) - 1  # 1-based to 0-based
            time_val = float(values["time"])
            head_val = float(values["head"])
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid data in row: {row_data}. Error: {e}")

        if well_name not in wells:
            well_info: dict = {"layer": layer, "times": [], "heads": []}
            if has_node:
                well_info["node"] = int(values["node"]) - 1
            else:
                well_info["row"] = int(values["row"]) - 1
                well_info["col"] = int(values["col"]) - 1
            wells[well_name] = well_info

        wells[well_name]["times"].append(time_val)
        wells[well_name]["heads"].append(head_val)
        n_obs += 1

    return {"wells": wells, "n_observations": n_obs}, "long", list(wells.keys())


def _parse_wide_format(header: list[str], data_rows: list[list[str]]) -> tuple[dict, str, list[str]]:
    """Parse wide-format observation CSV (no location info)."""
    well_names = [h.strip() for h in header[1:] if h.strip()]
    wells: dict[str, dict] = {}
    n_obs = 0

    for wn in well_names:
        wells[wn] = {"times": [], "heads": []}

    for row_data in data_rows:
        if not row_data or all(c.strip() == "" for c in row_data):
            continue

        try:
            time_val = float(row_data[0].strip())
        except (ValueError, IndexError):
            continue

        for i, wn in enumerate(well_names, start=1):
            if i >= len(row_data):
                continue
            val_str = row_data[i].strip()
            if val_str == "":
                continue
            try:
                head_val = float(val_str)
                wells[wn]["times"].append(time_val)
                wells[wn]["heads"].append(head_val)
                n_obs += 1
            except ValueError:
                continue

    return {"wells": wells, "n_observations": n_obs}, "wide", well_names


# ─── Endpoints ─────────────────────────────────────────────────────


@router.get("/sets", response_model=list[ObservationSetSummary])
async def list_observation_sets(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[ObservationSetSummary]:
    """List all observation sets for a project."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    storage = get_storage_service()
    sets = _load_sets_index(storage, project_id)

    return [
        ObservationSetSummary(
            id=s["id"],
            name=s["name"],
            source=s.get("source", "upload"),
            format=s.get("format", "long"),
            wells=s.get("wells", []),
            n_observations=s.get("n_observations", 0),
            created_at=datetime.fromisoformat(s["created_at"]) if isinstance(s.get("created_at"), str) else s.get("created_at", datetime.now(timezone.utc)),
        )
        for s in sets
    ]


@router.post("/sets", response_model=ObservationSet)
async def create_observation_set(
    project_id: UUID,
    file: UploadFile = File(...),
    name: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> ObservationSet:
    """
    Upload a new observation CSV and create an observation set.

    Accepts long format (WellName,Layer,Row,Col,Time,Head) or
    wide format (Time,MW-01,MW-02,...).
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    content_bytes = await file.read()
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded CSV",
        )

    try:
        parsed, format_type, well_names = _parse_observation_csv(content)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not parsed["wells"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid observation data found in CSV",
        )

    # Generate set ID and metadata
    set_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    set_name = name or file.filename or f"Observation Set {set_id[:8]}"

    storage = get_storage_service()

    # Store raw CSV
    storage.upload_bytes(
        settings.minio_bucket_models,
        _set_data_path(project_id, set_id),
        content_bytes,
        content_type="text/csv",
    )

    # Store parsed JSON
    storage.upload_bytes(
        settings.minio_bucket_models,
        _set_parsed_path(project_id, set_id),
        json.dumps(parsed).encode("utf-8"),
        content_type="application/json",
    )

    # Store metadata
    metadata = {
        "id": set_id,
        "name": set_name,
        "source": "upload",
        "format": format_type,
        "column_mapping": None,
        "wells": well_names,
        "n_observations": parsed["n_observations"],
        "created_at": created_at.isoformat(),
    }

    storage.upload_bytes(
        settings.minio_bucket_models,
        _set_metadata_path(project_id, set_id),
        json.dumps(metadata, default=str).encode("utf-8"),
        content_type="application/json",
    )

    # Update index
    sets = _load_sets_index(storage, project_id)
    sets.append(metadata)
    _save_sets_index(storage, project_id, sets)

    return ObservationSet(
        id=set_id,
        name=set_name,
        source="upload",
        format=format_type,
        column_mapping=None,
        wells=well_names,
        n_observations=parsed["n_observations"],
        created_at=created_at,
    )


@router.get("/sets/{set_id}", response_model=ObservationSetData)
async def get_observation_set(
    project_id: UUID,
    set_id: str,
    db: AsyncSession = Depends(get_db),
) -> ObservationSetData:
    """Get parsed observation data for a specific set."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    storage = get_storage_service()
    parsed_path = _set_parsed_path(project_id, set_id)

    if not storage.object_exists(settings.minio_bucket_models, parsed_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Observation set {set_id} not found",
        )

    data = storage.download_file(settings.minio_bucket_models, parsed_path)
    parsed = json.loads(data)

    wells = {}
    for name, well_data in parsed.get("wells", {}).items():
        wells[name] = ObservationWellData(
            layer=well_data.get("layer", 0),
            row=well_data.get("row"),
            col=well_data.get("col"),
            node=well_data.get("node"),
            times=well_data.get("times", []),
            heads=well_data.get("heads", []),
        )

    return ObservationSetData(
        wells=wells,
        n_observations=parsed.get("n_observations", 0),
    )


@router.patch("/sets/{set_id}", response_model=ObservationSet)
async def update_observation_set(
    project_id: UUID,
    set_id: str,
    update: ObservationSetUpdate,
    db: AsyncSession = Depends(get_db),
) -> ObservationSet:
    """Update observation set name or column mapping."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    storage = get_storage_service()
    metadata_path = _set_metadata_path(project_id, set_id)

    if not storage.object_exists(settings.minio_bucket_models, metadata_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Observation set {set_id} not found",
        )

    # Load current metadata
    data = storage.download_file(settings.minio_bucket_models, metadata_path)
    metadata = json.loads(data)

    # Apply updates
    if update.name is not None:
        metadata["name"] = update.name

    if update.column_mapping is not None:
        metadata["column_mapping"] = update.column_mapping.model_dump()

        # Re-parse the CSV with new mapping
        csv_path = _set_data_path(project_id, set_id)
        if storage.object_exists(settings.minio_bucket_models, csv_path):
            csv_content = storage.download_file(settings.minio_bucket_models, csv_path).decode("utf-8")
            try:
                parsed, format_type, well_names = _parse_observation_csv(
                    csv_content, update.column_mapping
                )
                metadata["format"] = format_type
                metadata["wells"] = well_names
                metadata["n_observations"] = parsed["n_observations"]

                # Update parsed JSON
                storage.upload_bytes(
                    settings.minio_bucket_models,
                    _set_parsed_path(project_id, set_id),
                    json.dumps(parsed).encode("utf-8"),
                    content_type="application/json",
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to parse CSV with new mapping: {e}",
                )

    # Save updated metadata
    storage.upload_bytes(
        settings.minio_bucket_models,
        metadata_path,
        json.dumps(metadata, default=str).encode("utf-8"),
        content_type="application/json",
    )

    # Update index
    sets = _load_sets_index(storage, project_id)
    for i, s in enumerate(sets):
        if s["id"] == set_id:
            sets[i] = metadata
            break
    _save_sets_index(storage, project_id, sets)

    return ObservationSet(
        id=metadata["id"],
        name=metadata["name"],
        source=metadata.get("source", "upload"),
        format=metadata.get("format", "long"),
        column_mapping=ColumnMapping(**metadata["column_mapping"]) if metadata.get("column_mapping") else None,
        wells=metadata.get("wells", []),
        n_observations=metadata.get("n_observations", 0),
        created_at=datetime.fromisoformat(metadata["created_at"]) if isinstance(metadata.get("created_at"), str) else datetime.now(timezone.utc),
    )


@router.delete("/sets/{set_id}")
async def delete_observation_set(
    project_id: UUID,
    set_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete an observation set and its data."""
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    storage = get_storage_service()

    # Delete set files
    set_base = _set_path(project_id, set_id)
    storage.delete_prefix(settings.minio_bucket_models, set_base)

    # Update index
    sets = _load_sets_index(storage, project_id)
    sets = [s for s in sets if s["id"] != set_id]
    _save_sets_index(storage, project_id, sets)

    return {"message": f"Observation set {set_id} deleted successfully"}


@router.post("/from-file", response_model=ObservationSet)
async def mark_file_as_observation(
    project_id: UUID,
    request: MarkFileAsObservation,
    db: AsyncSession = Depends(get_db),
) -> ObservationSet:
    """
    Mark an existing file in the project as an observation dataset.

    Use this to convert a CSV file that was uploaded with the model ZIP
    into a proper observation set with column mapping.
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
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

    # Check if file exists in model storage
    file_path = f"{project.storage_path}/{request.file_path}"
    if not storage.object_exists(settings.minio_bucket_models, file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {request.file_path} not found in project",
        )

    # Download and parse the CSV
    content_bytes = storage.download_file(settings.minio_bucket_models, file_path)
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded CSV",
        )

    try:
        parsed, format_type, well_names = _parse_observation_csv(
            content, request.column_mapping
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not parsed["wells"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid observation data found in CSV with the provided column mapping",
        )

    # Create observation set
    set_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    # Store CSV copy in observations directory
    storage.upload_bytes(
        settings.minio_bucket_models,
        _set_data_path(project_id, set_id),
        content_bytes,
        content_type="text/csv",
    )

    # Store parsed JSON
    storage.upload_bytes(
        settings.minio_bucket_models,
        _set_parsed_path(project_id, set_id),
        json.dumps(parsed).encode("utf-8"),
        content_type="application/json",
    )

    # Store metadata
    metadata = {
        "id": set_id,
        "name": request.name,
        "source": "manual_marked",
        "source_file": request.file_path,
        "format": format_type,
        "column_mapping": request.column_mapping.model_dump(),
        "wells": well_names,
        "n_observations": parsed["n_observations"],
        "created_at": created_at.isoformat(),
    }

    storage.upload_bytes(
        settings.minio_bucket_models,
        _set_metadata_path(project_id, set_id),
        json.dumps(metadata, default=str).encode("utf-8"),
        content_type="application/json",
    )

    # Update index
    sets = _load_sets_index(storage, project_id)
    sets.append(metadata)
    _save_sets_index(storage, project_id, sets)

    return ObservationSet(
        id=set_id,
        name=request.name,
        source="manual_marked",
        format=format_type,
        column_mapping=request.column_mapping,
        wells=well_names,
        n_observations=parsed["n_observations"],
        created_at=created_at,
    )


# ─── Legacy Endpoints (backwards compatibility) ────────────────────


@router.post("/upload")
async def upload_observations_legacy(
    project_id: UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Legacy endpoint for uploading observations.
    Creates a new observation set and returns summary.
    """
    result = await create_observation_set(project_id, file, None, db)
    return {
        "message": "Observations uploaded successfully",
        "set_id": result.id,
        "n_wells": len(result.wells),
        "n_observations": result.n_observations,
        "wells": result.wells,
    }


@router.get("")
async def get_observations_legacy(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Legacy endpoint: Return combined observations from all sets.
    For backwards compatibility with existing PEST integration.
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    storage = get_storage_service()
    sets = _load_sets_index(storage, project_id)

    if not sets:
        return {"wells": {}, "n_observations": 0}

    # Merge all observation sets
    merged_wells: dict = {}
    total_obs = 0

    for s in sets:
        parsed_path = _set_parsed_path(project_id, s["id"])
        try:
            if storage.object_exists(settings.minio_bucket_models, parsed_path):
                data = storage.download_file(settings.minio_bucket_models, parsed_path)
                parsed = json.loads(data)

                for well_name, well_data in parsed.get("wells", {}).items():
                    # Prefix well name with set ID to avoid collisions
                    # Actually for backwards compat, use original names
                    if well_name not in merged_wells:
                        merged_wells[well_name] = well_data
                    else:
                        # Merge times/heads
                        merged_wells[well_name]["times"].extend(well_data.get("times", []))
                        merged_wells[well_name]["heads"].extend(well_data.get("heads", []))

                total_obs += parsed.get("n_observations", 0)
        except Exception:
            continue

    return {"wells": merged_wells, "n_observations": total_obs}


@router.delete("")
async def delete_observations_legacy(
    project_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Legacy endpoint: Delete all observation sets for a project.
    """
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    storage = get_storage_service()
    prefix = _observations_base_path(project_id)
    storage.delete_prefix(settings.minio_bucket_models, prefix)

    return {"message": "All observations deleted successfully"}
