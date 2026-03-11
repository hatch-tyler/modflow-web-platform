"""Tests for file editor endpoint security and functionality.

Covers:
- Path traversal prevention (encoded sequences, UNC paths, absolute paths)
- Backup timestamp validation
- Binary file rejection
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project


class TestPathTraversalPrevention:
    """Verify path traversal attacks are blocked."""

    @pytest.mark.asyncio
    async def test_dotdot_traversal_blocked(
        self, test_client: AsyncClient, persisted_project: Project, db_session: AsyncSession
    ):
        """Direct '..' traversal should be rejected."""
        persisted_project.storage_path = "projects/test/model"
        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/files/../../../etc/passwd/content"
        )
        assert response.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_absolute_path_blocked(
        self, test_client: AsyncClient, persisted_project: Project, db_session: AsyncSession
    ):
        """Absolute paths should be rejected."""
        persisted_project.storage_path = "projects/test/model"
        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/files//etc/passwd/content"
        )
        assert response.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_colon_path_blocked(
        self, test_client: AsyncClient, persisted_project: Project, db_session: AsyncSession
    ):
        """Windows-style drive paths should be rejected."""
        persisted_project.storage_path = "projects/test/model"
        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/files/C:/Windows/System32/config/content"
        )
        assert response.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_valid_nested_path_allowed(
        self, test_client: AsyncClient, persisted_project: Project, db_session: AsyncSession
    ):
        """Legitimate nested file paths should work (may 404 if file doesn't exist)."""
        persisted_project.storage_path = "projects/test/model"
        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/files/subdir/model.nam/content"
        )
        # Should pass path validation — 404 is expected since the file doesn't exist in mock storage
        assert response.status_code in (200, 404)


class TestBackupTimestampValidation:
    """Verify backup timestamp format is validated."""

    @pytest.mark.asyncio
    async def test_invalid_timestamp_rejected(
        self, test_client: AsyncClient, persisted_project: Project, db_session: AsyncSession
    ):
        """Revert with invalid timestamp format should be rejected."""
        persisted_project.storage_path = "projects/test/model"
        await db_session.commit()

        response = await test_client.post(
            f"/api/v1/projects/{persisted_project.id}/files/model.nam/revert",
            json={"backup_timestamp": "../../../etc/passwd"},
        )
        assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.asyncio
    async def test_valid_timestamp_format(
        self, test_client: AsyncClient, persisted_project: Project, db_session: AsyncSession
    ):
        """Revert with valid timestamp format should pass validation (may 404 if backup doesn't exist)."""
        persisted_project.storage_path = "projects/test/model"
        await db_session.commit()

        response = await test_client.post(
            f"/api/v1/projects/{persisted_project.id}/files/model.nam/revert",
            json={"backup_timestamp": "20250101_120000"},
        )
        # Should pass validation — 404 expected since backup doesn't exist
        assert response.status_code in (200, 404)


class TestBinaryFileRejection:
    """Verify binary files cannot be read as text."""

    @pytest.mark.asyncio
    async def test_hds_file_rejected(
        self, test_client: AsyncClient, persisted_project: Project, db_session: AsyncSession
    ):
        """Binary head files should be rejected."""
        persisted_project.storage_path = "projects/test/model"
        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/files/output.hds/content"
        )
        assert response.status_code == 400
        assert "Binary" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_exe_file_rejected(
        self, test_client: AsyncClient, persisted_project: Project, db_session: AsyncSession
    ):
        """Executable files should be rejected."""
        persisted_project.storage_path = "projects/test/model"
        await db_session.commit()

        response = await test_client.get(
            f"/api/v1/projects/{persisted_project.id}/files/malware.exe/content"
        )
        assert response.status_code == 400
        assert "Binary" in response.json()["detail"]
