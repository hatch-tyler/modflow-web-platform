"""Project and Run database models."""

import enum
import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class ModelType(str, enum.Enum):
    """MODFLOW model type enumeration."""
    MODFLOW_2005 = "mf2005"
    MODFLOW_NWT = "mfnwt"
    MODFLOW_USG = "mfusg"
    MODFLOW_6 = "mf6"
    UNKNOWN = "unknown"


class RunStatus(str, enum.Enum):
    """Run status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunType(str, enum.Enum):
    """Run type enumeration."""
    FORWARD = "forward"
    PEST_GLM = "pest_glm"
    PEST_IES = "pest_ies"


class Project(Base, UUIDMixin, TimestampMixin):
    """Project model representing a MODFLOW model project."""

    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Model metadata (populated after parsing)
    model_type: Mapped[Optional[ModelType]] = mapped_column(
        Enum(ModelType),
        nullable=True,
        default=ModelType.UNKNOWN,
    )
    nlay: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    nrow: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ncol: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    nper: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    grid_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Grid coordinate reference (populated during validation)
    xoff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    yoff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    angrot: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    epsg: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    length_unit: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Temporal and grid spacing metadata (populated during validation)
    stress_period_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    start_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    delr: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    delc: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    time_unit: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Storage paths
    storage_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Validation status
    is_valid: Mapped[bool] = mapped_column(default=False)
    validation_errors: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Package information (which MODFLOW packages are present)
    packages: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    runs: Mapped[list["Run"]] = relationship(
        "Run",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Project {self.name} ({self.id})>"


class Run(Base, UUIDMixin, TimestampMixin):
    """Run model representing a simulation or calibration run."""

    __tablename__ = "runs"

    __table_args__ = (
        Index("idx_runs_project_status", "project_id", "status"),
        Index("idx_runs_project_created", "project_id", "created_at"),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    run_type: Mapped[RunType] = mapped_column(
        Enum(RunType),
        nullable=False,
        default=RunType.FORWARD,
        index=True,
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus),
        nullable=False,
        default=RunStatus.PENDING,
        index=True,
    )

    # Execution details
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Celery task tracking
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Run configuration (solver settings, PEST config, etc.)
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Results summary
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    results_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Performance metrics
    convergence_info: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="runs")

    def __repr__(self) -> str:
        return f"<Run {self.id} ({self.status.value})>"
