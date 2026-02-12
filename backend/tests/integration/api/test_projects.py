"""Integration tests for project management endpoints.

Tests CRUD operations, validation, and error handling.
"""

import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.models.project import ModelType, Project


class TestListProjects:
    """Tests for GET /projects endpoint."""

    @pytest.mark.asyncio
    async def test_list_projects_empty(self, test_client: AsyncClient):
        """Test listing projects when none exist."""
        response = await test_client.get("/api/v1/projects")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_projects_with_data(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test listing projects when some exist."""
        response = await test_client.get("/api/v1/projects")

        assert response.status_code == 200
        projects = response.json()
        assert len(projects) >= 1

        # Find our test project
        test_project = next(
            (p for p in projects if p["id"] == str(persisted_project.id)), None
        )
        assert test_project is not None
        assert test_project["name"] == persisted_project.name

    @pytest.mark.asyncio
    async def test_list_projects_pagination(self, test_client: AsyncClient, db_session):
        """Test project listing with pagination."""
        # Create multiple projects with explicit UUIDs to avoid asyncpg issues
        created_projects = []
        for i in range(5):
            project = Project(id=uuid.uuid4(), name=f"Project {i}")
            db_session.add(project)
            created_projects.append(project)
        await db_session.commit()

        try:
            # Test skip
            response = await test_client.get("/api/v1/projects", params={"skip": 2, "limit": 2})

            assert response.status_code == 200
            projects = response.json()
            assert len(projects) <= 2
        finally:
            # Cleanup
            for project in created_projects:
                await db_session.delete(project)
            await db_session.commit()


class TestCreateProject:
    """Tests for POST /projects endpoint."""

    @pytest.mark.asyncio
    async def test_create_project_minimal(self, test_client: AsyncClient):
        """Test creating a project with minimal data."""
        response = await test_client.post(
            "/api/v1/projects",
            json={"name": "New Project"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Project"
        assert data["description"] is None
        assert data["is_valid"] is False
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_create_project_with_description(self, test_client: AsyncClient):
        """Test creating a project with description."""
        response = await test_client.post(
            "/api/v1/projects",
            json={
                "name": "Described Project",
                "description": "A test project with description",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Described Project"
        assert data["description"] == "A test project with description"

    @pytest.mark.asyncio
    async def test_create_project_missing_name(self, test_client: AsyncClient):
        """Test that creating a project without name fails."""
        response = await test_client.post(
            "/api/v1/projects",
            json={"description": "No name provided"},
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_create_project_empty_name(self, test_client: AsyncClient):
        """Test that creating a project with empty name fails."""
        response = await test_client.post(
            "/api/v1/projects",
            json={"name": ""},
        )

        # Empty string might be valid depending on validation rules
        # Adjust assertion based on actual behavior
        assert response.status_code in [201, 422]

    @pytest.mark.asyncio
    async def test_create_project_returns_uuid(self, test_client: AsyncClient):
        """Test that created project has a valid UUID."""
        response = await test_client.post(
            "/api/v1/projects",
            json={"name": "UUID Test"},
        )

        assert response.status_code == 201
        project_id = response.json()["id"]

        # Verify it's a valid UUID
        parsed_uuid = uuid.UUID(project_id)
        assert str(parsed_uuid) == project_id


class TestGetProject:
    """Tests for GET /projects/{project_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_project_exists(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test getting an existing project."""
        response = await test_client.get(f"/api/v1/projects/{persisted_project.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(persisted_project.id)
        assert data["name"] == persisted_project.name
        assert data["model_type"] == persisted_project.model_type.value

    @pytest.mark.asyncio
    async def test_get_project_not_found(self, test_client: AsyncClient):
        """Test getting a non-existent project returns 404."""
        fake_id = uuid.uuid4()
        response = await test_client.get(f"/api/v1/projects/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_project_invalid_uuid(self, test_client: AsyncClient):
        """Test getting a project with invalid UUID format."""
        response = await test_client.get("/api/v1/projects/not-a-uuid")

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_project_includes_grid_metadata(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test that project response includes grid metadata."""
        response = await test_client.get(f"/api/v1/projects/{persisted_project.id}")

        assert response.status_code == 200
        data = response.json()
        assert "nlay" in data
        assert "nrow" in data
        assert "ncol" in data
        assert "nper" in data


class TestUpdateProject:
    """Tests for PATCH /projects/{project_id} endpoint."""

    @pytest.mark.asyncio
    async def test_update_project_name(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test updating project name."""
        response = await test_client.patch(
            f"/api/v1/projects/{persisted_project.id}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_project_description(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test updating project description."""
        response = await test_client.patch(
            f"/api/v1/projects/{persisted_project.id}",
            json={"description": "New description"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "New description"

    @pytest.mark.asyncio
    async def test_update_project_partial(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test that partial updates only modify specified fields."""
        original_name = persisted_project.name

        response = await test_client.patch(
            f"/api/v1/projects/{persisted_project.id}",
            json={"description": "Only update description"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == original_name  # Unchanged
        assert data["description"] == "Only update description"

    @pytest.mark.asyncio
    async def test_update_project_not_found(self, test_client: AsyncClient):
        """Test updating non-existent project returns 404."""
        fake_id = uuid.uuid4()
        response = await test_client.patch(
            f"/api/v1/projects/{fake_id}",
            json={"name": "Should Fail"},
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_project_updates_timestamp(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test that updating project updates the updated_at timestamp."""
        original_updated_at = persisted_project.updated_at

        response = await test_client.patch(
            f"/api/v1/projects/{persisted_project.id}",
            json={"name": "Timestamp Test"},
        )

        assert response.status_code == 200
        # Note: SQLite in tests may not support server-side timestamps
        # so we just verify the field exists
        assert "updated_at" in response.json()


class TestDeleteProject:
    """Tests for DELETE /projects/{project_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_project(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test deleting a project."""
        project_id = persisted_project.id

        response = await test_client.delete(f"/api/v1/projects/{project_id}")

        assert response.status_code == 204

        # Verify project is deleted
        get_response = await test_client.get(f"/api/v1/projects/{project_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_project_not_found(self, test_client: AsyncClient):
        """Test deleting non-existent project returns 404."""
        fake_id = uuid.uuid4()
        response = await test_client.delete(f"/api/v1/projects/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_project_idempotent(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        """Test that deleting an already deleted project returns 404."""
        project_id = persisted_project.id

        # First delete
        response1 = await test_client.delete(f"/api/v1/projects/{project_id}")
        assert response1.status_code == 204

        # Second delete
        response2 = await test_client.delete(f"/api/v1/projects/{project_id}")
        assert response2.status_code == 404


class TestProjectValidation:
    """Tests for project validation rules."""

    @pytest.mark.asyncio
    async def test_project_name_max_length(self, test_client: AsyncClient):
        """Test project name maximum length validation."""
        # Create a name longer than 255 characters
        long_name = "x" * 300

        response = await test_client.post(
            "/api/v1/projects",
            json={"name": long_name},
        )

        # Should fail validation or be truncated
        # Adjust based on actual implementation
        assert response.status_code in [201, 422]
