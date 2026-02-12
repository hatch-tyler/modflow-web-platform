"""Pydantic schemas for observation sets and file categorization."""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ─── Column Mapping ────────────────────────────────────────────────


class ColumnMapping(BaseModel):
    """Mapping of CSV columns to observation data fields."""

    well_name: str = Field(..., description="Column name for well identifier")
    layer: str | int = Field(..., description="Column name or fixed layer value")
    row: Optional[str | int] = Field(None, description="Column name or fixed row value")
    col: Optional[str | int] = Field(None, description="Column name or fixed column value")
    node: Optional[str | int] = Field(None, description="Column name or fixed node value (for unstructured)")
    time: str = Field(..., description="Column name for time values")
    value: str = Field(..., description="Column name for observation values (e.g., head)")


# ─── Observation Set ───────────────────────────────────────────────


class ObservationSetBase(BaseModel):
    """Base observation set attributes."""

    name: str = Field(..., description="Display name for the observation set")


class ObservationSetCreate(ObservationSetBase):
    """Schema for creating a new observation set."""

    column_mapping: Optional[ColumnMapping] = Field(
        None, description="Column mapping for CSV parsing"
    )


class ObservationSetUpdate(BaseModel):
    """Schema for updating an observation set."""

    name: Optional[str] = None
    column_mapping: Optional[ColumnMapping] = None


class ObservationSet(ObservationSetBase):
    """Full observation set with metadata."""

    id: str = Field(..., description="Unique identifier")
    source: Literal["upload", "zip_detected", "manual_marked"] = Field(
        ..., description="How this set was created"
    )
    format: Literal["long", "wide"] = Field(..., description="CSV format")
    column_mapping: Optional[ColumnMapping] = Field(
        None, description="Column mapping used to parse CSV"
    )
    wells: list[str] = Field(default_factory=list, description="Well names in set")
    n_observations: int = Field(0, description="Total observation count")
    created_at: datetime = Field(..., description="When set was created")

    class Config:
        from_attributes = True


class ObservationSetSummary(BaseModel):
    """Minimal observation set info for listing."""

    id: str
    name: str
    source: Literal["upload", "zip_detected", "manual_marked"]
    format: Literal["long", "wide"]
    wells: list[str]
    n_observations: int
    created_at: datetime
    file_path: Optional[str] = Field(
        None, description="Path to source CSV file (for detected observations)"
    )


class MarkFileAsObservation(BaseModel):
    """Request to mark an existing file as observation data."""

    file_path: str = Field(..., description="Path to CSV file within project storage")
    name: str = Field(..., description="Display name for the observation set")
    column_mapping: ColumnMapping = Field(..., description="How to map CSV columns")


# ─── Observation Data ──────────────────────────────────────────────


class ObservationWellData(BaseModel):
    """Observation data for a single well."""

    layer: int
    row: Optional[int] = None
    col: Optional[int] = None
    node: Optional[int] = None
    times: list[float]
    heads: list[float]


class ObservationSetData(BaseModel):
    """Full parsed observation data for a set."""

    wells: dict[str, ObservationWellData]
    n_observations: int


# ─── File Categorization ───────────────────────────────────────────


class FileInfo(BaseModel):
    """Information about a single file."""

    path: str
    name: str
    extension: str
    description: str = ""
    size: int = 0


class CategoryInfo(BaseModel):
    """Info about a single category of files."""

    count: int
    size: int


class CategorizedFiles(BaseModel):
    """Response for categorized file listing."""

    categories: dict[str, list[FileInfo]] = Field(
        default_factory=dict,
        description="Files grouped by category: model_required, model_optional, pest, observation, other"
    )
    blocked_rejected: list[str] = Field(
        default_factory=list,
        description="Files that were rejected due to blocked extensions"
    )
    total_files: int = 0
    total_size_mb: float = 0.0

    # Auto-detected observation sets from ZIP
    detected_observations: list[ObservationSetSummary] = Field(
        default_factory=list,
        description="Observation CSVs detected in the uploaded files"
    )


# ─── PEST Multi-Set Support ────────────────────────────────────────


class ObservationSetSelection(BaseModel):
    """Selection of observation sets for PEST calibration."""

    set_id: str = Field(..., description="Observation set ID")
    weight_multiplier: float = Field(
        1.0, description="Multiplier applied to all weights in this set"
    )
    per_well_weights: Optional[dict[str, float]] = Field(
        None, description="Override weights for specific wells"
    )


class PestObservationConfig(BaseModel):
    """Configuration for observations in PEST calibration."""

    observation_sets: list[ObservationSetSelection] = Field(
        ..., description="Which observation sets to include"
    )
