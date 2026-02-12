"""Pydantic schemas for Project and Run."""

from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.project import ModelType, RunStatus, RunType


# -----------------------------------------------------------------------------
# Project Schemas
# -----------------------------------------------------------------------------


class ProjectCreate(BaseModel):
    """Schema for creating a new project."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    xoff: Optional[float] = None
    yoff: Optional[float] = None
    angrot: Optional[float] = None
    epsg: Optional[int] = None
    start_date: Optional[date] = None


class ProjectSummary(BaseModel):
    """Brief project summary for list views."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    model_type: Optional[ModelType]
    is_valid: bool
    nlay: Optional[int]
    nrow: Optional[int]
    ncol: Optional[int]
    grid_type: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ProjectDetail(BaseModel):
    """Full project details."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: Optional[str]
    model_type: Optional[ModelType]
    nlay: Optional[int]
    nrow: Optional[int]
    ncol: Optional[int]
    nper: Optional[int]
    grid_type: Optional[str] = None
    xoff: Optional[float] = None
    yoff: Optional[float] = None
    angrot: Optional[float] = None
    epsg: Optional[int] = None
    length_unit: Optional[str] = None
    stress_period_data: Optional[list[dict[str, Any]]] = None
    start_date: Optional[date] = None
    delr: Optional[list[float]] = None
    delc: Optional[list[float]] = None
    time_unit: Optional[str] = None
    storage_path: Optional[str]
    is_valid: bool
    validation_errors: Optional[dict[str, Any]]
    packages: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class ValidationReport(BaseModel):
    """Model validation report."""

    is_valid: bool
    model_type: Optional[ModelType]
    grid_info: Optional[dict[str, Any]]
    packages_found: list[str]
    packages_missing: list[str]
    errors: list[str]
    warnings: list[str]


# -----------------------------------------------------------------------------
# Run Schemas
# -----------------------------------------------------------------------------


class RunCreate(BaseModel):
    """Schema for creating a new run."""

    name: Optional[str] = None
    run_type: RunType = RunType.FORWARD
    config: Optional[dict[str, Any]] = None


class RunSummary(BaseModel):
    """Brief run summary for list views."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    name: Optional[str]
    run_type: RunType
    status: RunStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime


class RunDetail(BaseModel):
    """Full run details."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    name: Optional[str]
    run_type: RunType
    status: RunStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    celery_task_id: Optional[str]
    config: Optional[dict[str, Any]]
    exit_code: Optional[int]
    error_message: Optional[str]
    results_path: Optional[str]
    convergence_info: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


# -----------------------------------------------------------------------------
# Health Check Schemas
# -----------------------------------------------------------------------------


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    database: str = "connected"
    redis: str = "connected"
    minio: str = "connected"


class ServiceStatus(BaseModel):
    """Individual service status."""

    name: str
    status: str
    message: Optional[str] = None
