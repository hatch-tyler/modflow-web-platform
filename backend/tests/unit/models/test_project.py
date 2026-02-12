"""Tests for Project and Run database models.

Tests model instantiation, enums, relationships, and field validations.
"""

import uuid
from datetime import datetime

import pytest

from app.models.project import (
    ModelType,
    Project,
    Run,
    RunStatus,
    RunType,
)


class TestModelTypeEnum:
    """Tests for ModelType enumeration."""

    def test_model_types(self):
        """Test all model types are defined."""
        assert ModelType.MODFLOW_2005.value == "mf2005"
        assert ModelType.MODFLOW_NWT.value == "mfnwt"
        assert ModelType.MODFLOW_USG.value == "mfusg"
        assert ModelType.MODFLOW_6.value == "mf6"
        assert ModelType.UNKNOWN.value == "unknown"

    def test_model_type_is_string_enum(self):
        """Test that ModelType inherits from str."""
        assert isinstance(ModelType.MODFLOW_6, str)
        assert ModelType.MODFLOW_6 == "mf6"


class TestRunStatusEnum:
    """Tests for RunStatus enumeration."""

    def test_run_statuses(self):
        """Test all run statuses are defined."""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.QUEUED.value == "queued"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.CANCELLED.value == "cancelled"

    def test_run_status_is_string_enum(self):
        """Test that RunStatus inherits from str."""
        assert isinstance(RunStatus.RUNNING, str)
        assert RunStatus.RUNNING == "running"


class TestRunTypeEnum:
    """Tests for RunType enumeration."""

    def test_run_types(self):
        """Test all run types are defined."""
        assert RunType.FORWARD.value == "forward"
        assert RunType.PEST_GLM.value == "pest_glm"
        assert RunType.PEST_IES.value == "pest_ies"

    def test_run_type_is_string_enum(self):
        """Test that RunType inherits from str."""
        assert isinstance(RunType.FORWARD, str)


class TestProjectModel:
    """Tests for Project model."""

    def test_project_creation_minimal(self):
        """Test creating a project with minimal required fields."""
        project = Project(name="Test Project")

        assert project.name == "Test Project"
        assert project.description is None
        assert project.model_type is None
        # Default is set at DB level, not in Python - so it's None until persisted
        assert project.is_valid is None or project.is_valid is False

    def test_project_creation_full(self):
        """Test creating a project with all fields."""
        project_id = uuid.uuid4()
        now = datetime.utcnow()

        project = Project(
            id=project_id,
            name="Full Project",
            description="A complete test project",
            model_type=ModelType.MODFLOW_6,
            nlay=3,
            nrow=100,
            ncol=100,
            nper=12,
            xoff=1000.0,
            yoff=2000.0,
            angrot=0.0,
            epsg=26915,
            length_unit="meters",
            storage_path="projects/test",
            is_valid=True,
            validation_errors=None,
            packages={"npf": True, "wel": True},
            created_at=now,
            updated_at=now,
        )

        assert project.id == project_id
        assert project.name == "Full Project"
        assert project.model_type == ModelType.MODFLOW_6
        assert project.nlay == 3
        assert project.nrow == 100
        assert project.ncol == 100
        assert project.nper == 12
        assert project.xoff == 1000.0
        assert project.epsg == 26915
        assert project.is_valid is True
        assert project.packages == {"npf": True, "wel": True}

    def test_project_repr(self):
        """Test project string representation."""
        project_id = uuid.uuid4()
        project = Project(id=project_id, name="Test")

        repr_str = repr(project)

        assert "Test" in repr_str
        assert str(project_id) in repr_str

    def test_project_tablename(self):
        """Test project table name."""
        assert Project.__tablename__ == "projects"

    def test_project_grid_metadata_nullable(self):
        """Test that grid metadata fields are nullable."""
        project = Project(name="Test")

        assert project.nlay is None
        assert project.nrow is None
        assert project.ncol is None
        assert project.nper is None

    def test_project_spatial_reference_nullable(self):
        """Test that spatial reference fields are nullable."""
        project = Project(name="Test")

        assert project.xoff is None
        assert project.yoff is None
        assert project.angrot is None
        assert project.epsg is None

    def test_project_jsonb_fields(self):
        """Test JSONB field storage."""
        project = Project(
            name="Test",
            packages={"dis": True, "npf": True, "wel": True},
            validation_errors={"error1": "Some error"},
        )

        assert project.packages["dis"] is True
        assert project.validation_errors["error1"] == "Some error"


class TestRunModel:
    """Tests for Run model."""

    def test_run_creation_minimal(self):
        """Test creating a run with minimal required fields."""
        project_id = uuid.uuid4()
        # Default enum values are set at the column level, may be None before DB insert
        run = Run(project_id=project_id, run_type=RunType.FORWARD, status=RunStatus.PENDING)

        assert run.project_id == project_id
        assert run.run_type == RunType.FORWARD
        assert run.status == RunStatus.PENDING
        assert run.name is None

    def test_run_creation_full(self):
        """Test creating a run with all fields."""
        project_id = uuid.uuid4()
        run_id = uuid.uuid4()
        now = datetime.utcnow()

        run = Run(
            id=run_id,
            project_id=project_id,
            name="Calibration Run 1",
            run_type=RunType.PEST_GLM,
            status=RunStatus.RUNNING,
            started_at=now,
            completed_at=None,
            celery_task_id="task-abc-123",
            config={"n_workers": 8, "lambda_scale_fac": 0.5},
            exit_code=None,
            error_message=None,
            results_path="results/run1",
            convergence_info={"iterations": 10, "phi": 125.5},
            created_at=now,
            updated_at=now,
        )

        assert run.id == run_id
        assert run.name == "Calibration Run 1"
        assert run.run_type == RunType.PEST_GLM
        assert run.status == RunStatus.RUNNING
        assert run.celery_task_id == "task-abc-123"
        assert run.config["n_workers"] == 8
        assert run.convergence_info["phi"] == 125.5

    def test_run_repr(self):
        """Test run string representation."""
        run_id = uuid.uuid4()
        run = Run(id=run_id, project_id=uuid.uuid4(), status=RunStatus.COMPLETED)

        repr_str = repr(run)

        assert str(run_id) in repr_str
        assert "completed" in repr_str

    def test_run_tablename(self):
        """Test run table name."""
        assert Run.__tablename__ == "runs"

    def test_run_forward_type(self):
        """Test forward run type."""
        run = Run(project_id=uuid.uuid4(), run_type=RunType.FORWARD)

        assert run.run_type == RunType.FORWARD
        assert run.run_type.value == "forward"

    def test_run_pest_types(self):
        """Test PEST run types."""
        run_glm = Run(project_id=uuid.uuid4(), run_type=RunType.PEST_GLM)
        run_ies = Run(project_id=uuid.uuid4(), run_type=RunType.PEST_IES)

        assert run_glm.run_type == RunType.PEST_GLM
        assert run_ies.run_type == RunType.PEST_IES

    def test_run_status_transitions(self):
        """Test run status can be updated."""
        run = Run(project_id=uuid.uuid4(), status=RunStatus.PENDING)

        assert run.status == RunStatus.PENDING

        run.status = RunStatus.QUEUED
        assert run.status == RunStatus.QUEUED

        run.status = RunStatus.RUNNING
        assert run.status == RunStatus.RUNNING

        run.status = RunStatus.COMPLETED
        assert run.status == RunStatus.COMPLETED

    def test_run_failed_with_error_message(self):
        """Test failed run with error message."""
        run = Run(
            project_id=uuid.uuid4(),
            status=RunStatus.FAILED,
            exit_code=1,
            error_message="MODFLOW failed to converge",
        )

        assert run.status == RunStatus.FAILED
        assert run.exit_code == 1
        assert run.error_message == "MODFLOW failed to converge"

    def test_run_convergence_info_jsonb(self):
        """Test convergence info JSONB storage."""
        run = Run(
            project_id=uuid.uuid4(),
            convergence_info={
                "iterations": 15,
                "final_phi": 42.3,
                "parameter_changes": [0.1, 0.05, 0.02],
            },
        )

        assert run.convergence_info["iterations"] == 15
        assert run.convergence_info["final_phi"] == 42.3
        assert len(run.convergence_info["parameter_changes"]) == 3


class TestProjectRunRelationship:
    """Tests for Project-Run relationship."""

    def test_project_has_runs_relationship(self):
        """Test that Project has a runs relationship."""
        project = Project(name="Test")

        # The runs attribute should exist (will be populated by SQLAlchemy)
        assert hasattr(project, 'runs')

    def test_run_has_project_relationship(self):
        """Test that Run has a project relationship."""
        run = Run(project_id=uuid.uuid4())

        # The project attribute should exist
        assert hasattr(run, 'project')
        assert hasattr(run, 'project_id')
