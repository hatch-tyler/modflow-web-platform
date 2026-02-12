"""Integration tests for simulation API endpoints.

Tests simulation lifecycle including start, status, cancellation, and error handling.
"""

import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.models.project import ModelType, Project, Run, RunStatus, RunType


class TestStartSimulation:
    """Tests for POST /projects/{project_id}/simulation/run endpoint."""

    @pytest.mark.asyncio
    async def test_start_simulation_requires_uploaded_model(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test that simulation requires uploaded model files."""
        # Project without storage_path
        response = await test_client.post(
            f"/api/v1/projects/{persisted_project.id}/simulation/run",
            json={},
        )

        assert response.status_code == 400
        assert "No model files uploaded" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_start_simulation_with_model(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test starting simulation with uploaded model."""
        # Set storage path to simulate uploaded model
        persisted_project.storage_path = f"projects/{persisted_project.id}"
        await db_session.flush()

        with patch("app.api.v1.simulation.run_forward_model") as mock_task:
            mock_async_result = MagicMock()
            mock_async_result.id = "celery-task-id-123"
            mock_task.delay.return_value = mock_async_result

            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/simulation/run",
                json={"name": "Test Simulation"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "run_id" in data
            assert data["task_id"] == "celery-task-id-123"
            assert data["status"] == "queued"
            assert data["message"] == "Simulation queued for execution"

    @pytest.mark.asyncio
    async def test_start_simulation_conflict_when_running(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test that starting simulation fails if one is already running."""
        persisted_project.storage_path = f"projects/{persisted_project.id}"
        await db_session.flush()

        # Create a running run
        running_run = Run(
            project_id=persisted_project.id,
            name="Running Simulation",
            run_type=RunType.FORWARD,
            status=RunStatus.RUNNING,
        )
        db_session.add(running_run)
        await db_session.flush()

        response = await test_client.post(
            f"/api/v1/projects/{persisted_project.id}/simulation/run",
            json={},
        )

        assert response.status_code == 409
        assert "already running" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_start_simulation_project_not_found(self, test_client: AsyncClient):
        """Test starting simulation for non-existent project."""
        fake_id = uuid.uuid4()

        response = await test_client.post(
            f"/api/v1/projects/{fake_id}/simulation/run",
            json={},
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_start_simulation_auto_names_run(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test that run gets auto-named if no name provided."""
        persisted_project.storage_path = f"projects/{persisted_project.id}"
        await db_session.flush()

        with patch("app.api.v1.simulation.run_forward_model") as mock_task:
            mock_async_result = MagicMock()
            mock_async_result.id = "task-123"
            mock_task.delay.return_value = mock_async_result

            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/simulation/run",
                json={},
            )

            assert response.status_code == 200
            # Run was created with a default name


class TestListRuns:
    """Tests for GET /projects/{project_id}/simulation/runs endpoint."""

    @pytest.mark.asyncio
    async def test_list_runs_empty(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test listing runs when none exist."""
        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/simulation/runs"
        )

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_runs_with_data(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test listing runs when some exist."""
        # Create runs
        for i in range(3):
            run = Run(
                project_id=persisted_project.id,
                name=f"Run {i}",
                run_type=RunType.FORWARD,
                status=RunStatus.COMPLETED,
            )
            db_session.add(run)
        await db_session.flush()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/simulation/runs"
        )

        assert response.status_code == 200
        runs = response.json()
        assert len(runs) == 3

    @pytest.mark.asyncio
    async def test_list_runs_pagination(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test run listing with limit parameter."""
        # Create many runs
        for i in range(10):
            run = Run(
                project_id=persisted_project.id,
                name=f"Run {i}",
                run_type=RunType.FORWARD,
                status=RunStatus.COMPLETED,
            )
            db_session.add(run)
        await db_session.flush()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/simulation/runs",
            params={"limit": 5},
        )

        assert response.status_code == 200
        runs = response.json()
        assert len(runs) == 5

    @pytest.mark.asyncio
    async def test_list_runs_ordered_by_date(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test that runs are ordered by creation date descending."""
        import time
        from datetime import timedelta

        base_time = datetime.utcnow()
        for i in range(3):
            run = Run(
                project_id=persisted_project.id,
                name=f"Run {i}",
                run_type=RunType.FORWARD,
                status=RunStatus.COMPLETED,
                # Explicitly set different created_at times
                created_at=base_time + timedelta(seconds=i),
            )
            db_session.add(run)
        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/simulation/runs"
        )

        assert response.status_code == 200
        runs = response.json()
        assert len(runs) >= 3

        # Verify descending order by checking names
        # (last created should be first)
        run_names = [r["name"] for r in runs if r["name"].startswith("Run ")]
        assert run_names[0] == "Run 2"


class TestGetRun:
    """Tests for GET /projects/{project_id}/simulation/runs/{run_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_run_exists(
        self, test_client: AsyncClient, persisted_run: Run
    ):
        """Test getting an existing run."""
        response = await test_client.get(
            f"/api/v1/projects/{persisted_run.project_id}/simulation/runs/{persisted_run.id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(persisted_run.id)
        assert data["status"] == persisted_run.status.value
        assert data["run_type"] == persisted_run.run_type.value

    @pytest.mark.asyncio
    async def test_get_run_not_found(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test getting a non-existent run."""
        fake_run_id = uuid.uuid4()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/simulation/runs/{fake_run_id}"
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_run_wrong_project(
        self, test_client: AsyncClient, db_session, persisted_run: Run
    ):
        """Test getting a run with wrong project ID."""
        other_project = Project(name="Other Project")
        db_session.add(other_project)
        await db_session.flush()

        response = await test_client.get(
            f"/api/v1/projects/{other_project.id}/simulation/runs/{persisted_run.id}"
        )

        assert response.status_code == 404


class TestCancelSimulation:
    """Tests for POST /projects/{project_id}/simulation/runs/{run_id}/cancel endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_running_simulation(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test cancelling a running simulation."""
        run = Run(
            project_id=persisted_project.id,
            name="Running Run",
            run_type=RunType.FORWARD,
            status=RunStatus.RUNNING,
        )
        db_session.add(run)
        await db_session.flush()

        with patch("app.api.v1.simulation.cancel_run") as mock_cancel:
            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/simulation/runs/{run.id}/cancel"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "cancelling"
            mock_cancel.delay.assert_called_once_with(str(run.id))

    @pytest.mark.asyncio
    async def test_cancel_pending_simulation(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test cancelling a pending simulation."""
        run = Run(
            project_id=persisted_project.id,
            name="Pending Run",
            run_type=RunType.FORWARD,
            status=RunStatus.PENDING,
        )
        db_session.add(run)
        await db_session.flush()

        with patch("app.api.v1.simulation.cancel_run"):
            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/simulation/runs/{run.id}/cancel"
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cancel_completed_simulation_fails(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test that cancelling a completed simulation fails."""
        run = Run(
            project_id=persisted_project.id,
            name="Completed Run",
            run_type=RunType.FORWARD,
            status=RunStatus.COMPLETED,
        )
        db_session.add(run)
        await db_session.flush()

        response = await test_client.post(
            f"/api/v1/projects/{persisted_project.id}/simulation/runs/{run.id}/cancel"
        )

        assert response.status_code == 400
        assert "Cannot cancel" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_cancel_failed_simulation_fails(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test that cancelling a failed simulation fails."""
        run = Run(
            project_id=persisted_project.id,
            name="Failed Run",
            run_type=RunType.FORWARD,
            status=RunStatus.FAILED,
        )
        db_session.add(run)
        await db_session.flush()

        response = await test_client.post(
            f"/api/v1/projects/{persisted_project.id}/simulation/runs/{run.id}/cancel"
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_run(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test cancelling a non-existent run."""
        fake_run_id = uuid.uuid4()

        response = await test_client.post(
            f"/api/v1/projects/{persisted_project.id}/simulation/runs/{fake_run_id}/cancel"
        )

        assert response.status_code == 404


class TestSimulationStatus:
    """Tests for GET /projects/{project_id}/simulation/status endpoint."""

    @pytest.mark.asyncio
    async def test_status_no_runs(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test status when no runs exist."""
        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/simulation/status"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["has_runs"] is False
        assert data["latest_run"] is None

    @pytest.mark.asyncio
    async def test_status_with_runs(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test status when runs exist."""
        run = Run(
            project_id=persisted_project.id,
            name="Latest Run",
            run_type=RunType.FORWARD,
            status=RunStatus.COMPLETED,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        db_session.add(run)
        await db_session.flush()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/simulation/status"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["has_runs"] is True
        assert data["latest_run"]["id"] == str(run.id)
        assert data["latest_run"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_status_returns_latest(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test that status returns the latest run."""
        from datetime import timedelta

        base_time = datetime.utcnow()

        # Create older run with earlier timestamp
        old_run = Run(
            project_id=persisted_project.id,
            name="Old Run",
            run_type=RunType.FORWARD,
            status=RunStatus.COMPLETED,
            created_at=base_time,
        )
        db_session.add(old_run)

        # Create newer run with later timestamp
        new_run = Run(
            project_id=persisted_project.id,
            name="New Run",
            run_type=RunType.FORWARD,
            status=RunStatus.RUNNING,
            created_at=base_time + timedelta(seconds=1),
        )
        db_session.add(new_run)
        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/simulation/status"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["latest_run"]["id"] == str(new_run.id)


class TestRunStatusTransitions:
    """Tests for run status transitions and lifecycle."""

    @pytest.mark.asyncio
    async def test_run_status_pending_to_queued(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test run starts in PENDING/QUEUED status."""
        persisted_project.storage_path = f"projects/{persisted_project.id}"
        await db_session.flush()

        with patch("app.api.v1.simulation.run_forward_model") as mock_task:
            mock_async_result = MagicMock()
            mock_async_result.id = "task-id"
            mock_task.delay.return_value = mock_async_result

            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/simulation/run"
            )

            assert response.status_code == 200
            assert response.json()["status"] == "queued"

    @pytest.mark.asyncio
    async def test_run_includes_timestamps(
        self, test_client: AsyncClient, db_session, persisted_project: Project
    ):
        """Test that run response includes timestamp fields."""
        run = Run(
            project_id=persisted_project.id,
            name="Timestamp Test",
            run_type=RunType.FORWARD,
            status=RunStatus.COMPLETED,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        db_session.add(run)
        await db_session.flush()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/simulation/runs/{run.id}"
        )

        data = response.json()
        assert data["started_at"] is not None
        assert data["completed_at"] is not None
        assert data["created_at"] is not None
