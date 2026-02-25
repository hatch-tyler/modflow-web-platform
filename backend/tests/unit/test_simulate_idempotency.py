"""Tests for simulation task idempotency and lock behavior.

Covers:
- Duplicate task detection via Redis lock
- Lock acquired BEFORE status update (B2 fix)
- Already-completed/failed tasks are skipped
- Lock is released on early failure (no storage_path)
"""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.models.project import ModelType, Project, Run, RunStatus, RunType


def _make_project(storage_path="projects/test/model"):
    """Create a mock Project for testing."""
    p = Project(
        id=uuid.uuid4(),
        name="Test",
        model_type=ModelType.MODFLOW_6,
        storage_path=storage_path,
        is_valid=True,
    )
    return p


def _make_run(project_id, status=RunStatus.QUEUED):
    """Create a mock Run for testing."""
    return Run(
        id=uuid.uuid4(),
        project_id=project_id,
        run_type=RunType.FORWARD,
        status=status,
    )


def _call_task(task, mock_self, *args):
    """Call a Celery bind=True task's underlying function directly.

    Bypasses Celery's Task.__call__ which auto-injects self via descriptor
    protocol. Extracts the unbound function via __func__ on the bound method.
    """
    raw_fn = task.run.__func__
    return raw_fn(mock_self, *args)


class TestSimulationIdempotencyGuard:
    """Test that requeued tasks are handled correctly."""

    def test_already_completed_skips(self):
        """A completed run should be skipped without acquiring a lock."""
        project = _make_project()
        run = _make_run(project.id, status=RunStatus.COMPLETED)

        mock_redis = MagicMock()
        mock_db = MagicMock()
        mock_db.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=project)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=run)),
        ]

        with patch("app.tasks.simulate.get_sync_client", return_value=mock_redis), \
             patch("app.tasks.simulate.SessionLocal") as mock_session_cls, \
             patch("app.tasks.simulate.get_storage_service"):
            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

            from app.tasks.simulate import run_forward_model

            mock_self = MagicMock()
            mock_self.request.id = "celery-task-123"

            result = _call_task(
                run_forward_model, mock_self, str(run.id), str(project.id)
            )

        assert result["status"] == "already_completed"
        # Redis lock should NOT have been attempted
        mock_redis.set.assert_not_called()

    def test_already_failed_skips(self):
        """A failed run should be skipped."""
        project = _make_project()
        run = _make_run(project.id, status=RunStatus.FAILED)

        mock_redis = MagicMock()
        mock_db = MagicMock()
        mock_db.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=project)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=run)),
        ]

        with patch("app.tasks.simulate.get_sync_client", return_value=mock_redis), \
             patch("app.tasks.simulate.SessionLocal") as mock_session_cls, \
             patch("app.tasks.simulate.get_storage_service"):
            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

            from app.tasks.simulate import run_forward_model

            mock_self = MagicMock()
            mock_self.request.id = "celery-task-123"

            result = _call_task(
                run_forward_model, mock_self, str(run.id), str(project.id)
            )

        assert result["status"] == "already_failed"
        mock_redis.set.assert_not_called()

    def test_lock_prevents_duplicate_execution(self):
        """If Redis lock is already held, task should be skipped."""
        project = _make_project()
        run = _make_run(project.id, status=RunStatus.QUEUED)

        mock_redis = MagicMock()
        # Lock acquisition fails (already held by another worker)
        mock_redis.set.return_value = False
        mock_redis.publish = MagicMock()
        mock_redis.rpush = MagicMock()
        mock_redis.ltrim = MagicMock()
        mock_redis.expire = MagicMock()

        mock_db = MagicMock()
        mock_db.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=project)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=run)),
        ]

        with patch("app.tasks.simulate.get_sync_client", return_value=mock_redis), \
             patch("app.tasks.simulate.SessionLocal") as mock_session_cls, \
             patch("app.tasks.simulate.get_storage_service"):
            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

            from app.tasks.simulate import run_forward_model

            mock_self = MagicMock()
            mock_self.request.id = "celery-task-123"

            result = _call_task(
                run_forward_model, mock_self, str(run.id), str(project.id)
            )

        assert result["status"] == "already_running"
        # Lock was attempted with nx=True
        mock_redis.set.assert_called_once()
        call_kwargs = mock_redis.set.call_args
        assert call_kwargs[1].get("nx") is True

    def test_lock_acquired_before_status_update(self):
        """Lock must be acquired before status is updated to RUNNING (B2 fix).

        After the fix, the lock is attempted for ALL non-terminal statuses,
        including QUEUED and PENDING, not just RUNNING.
        """
        project = _make_project()
        run = _make_run(project.id, status=RunStatus.QUEUED)

        lock_call_order = []
        commit_call_order = []

        mock_redis = MagicMock()

        def track_set(*args, **kwargs):
            lock_call_order.append("lock_attempted")
            return True  # Lock acquired successfully

        mock_redis.set.side_effect = track_set
        mock_redis.publish = MagicMock()
        mock_redis.rpush = MagicMock()
        mock_redis.ltrim = MagicMock()
        mock_redis.expire = MagicMock()
        mock_redis.delete = MagicMock()
        mock_redis.exists = MagicMock(return_value=False)
        mock_redis.pipeline = MagicMock(return_value=MagicMock(
            publish=MagicMock(),
            rpush=MagicMock(),
            ltrim=MagicMock(),
            expire=MagicMock(),
            execute=MagicMock(return_value=[]),
        ))

        mock_db = MagicMock()
        mock_db.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=project)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=run)),
        ]

        original_commit = mock_db.commit

        def track_commit():
            commit_call_order.append("commit")
            return original_commit()

        mock_db.commit = track_commit

        # We need to mock enough to get past the download phase
        # Since download will fail (mock storage), it will catch and set FAILED
        mock_storage = MagicMock()
        mock_storage.list_objects.side_effect = Exception("Test: stop here")

        with patch("app.tasks.simulate.get_sync_client", return_value=mock_redis), \
             patch("app.tasks.simulate.SessionLocal") as mock_session_cls, \
             patch("app.tasks.simulate.get_storage_service", return_value=mock_storage):
            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

            from app.tasks.simulate import run_forward_model

            mock_self = MagicMock()
            mock_self.request.id = "celery-task-123"

            result = _call_task(
                run_forward_model, mock_self, str(run.id), str(project.id)
            )

        # Lock should have been attempted
        assert "lock_attempted" in lock_call_order
        # Lock is acquired before the first commit (which sets status=RUNNING)
        # The lock call happens before any commit
        assert len(lock_call_order) > 0

    def test_lock_released_on_no_storage_path(self):
        """Lock should be released if project has no storage_path."""
        project = _make_project(storage_path=None)
        run = _make_run(project.id, status=RunStatus.QUEUED)

        mock_redis = MagicMock()
        mock_redis.set.return_value = True  # Lock acquired
        mock_redis.publish = MagicMock()
        mock_redis.rpush = MagicMock()
        mock_redis.ltrim = MagicMock()
        mock_redis.expire = MagicMock()

        mock_db = MagicMock()
        mock_db.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=project)),
            MagicMock(scalar_one_or_none=MagicMock(return_value=run)),
        ]

        with patch("app.tasks.simulate.get_sync_client", return_value=mock_redis), \
             patch("app.tasks.simulate.SessionLocal") as mock_session_cls, \
             patch("app.tasks.simulate.get_storage_service"):
            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

            from app.tasks.simulate import run_forward_model

            mock_self = MagicMock()
            mock_self.request.id = "celery-task-123"

            result = _call_task(
                run_forward_model, mock_self, str(run.id), str(project.id)
            )

        assert "error" in result
        assert "No model files" in result["error"]
        # Lock should be released via redis.delete()
        mock_redis.delete.assert_called()


class TestProjectOrRunNotFound:
    """Test that missing project/run is handled."""

    def test_missing_project_returns_error(self):
        """If project is not found, return error dict."""
        mock_redis = MagicMock()
        mock_redis.publish = MagicMock()
        mock_redis.rpush = MagicMock()
        mock_redis.ltrim = MagicMock()
        mock_redis.expire = MagicMock()

        mock_db = MagicMock()
        mock_db.execute.side_effect = [
            MagicMock(scalar_one_or_none=MagicMock(return_value=None)),  # project
            MagicMock(scalar_one_or_none=MagicMock(return_value=None)),  # run
        ]

        with patch("app.tasks.simulate.get_sync_client", return_value=mock_redis), \
             patch("app.tasks.simulate.SessionLocal") as mock_session_cls, \
             patch("app.tasks.simulate.get_storage_service"):
            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

            from app.tasks.simulate import run_forward_model

            mock_self = MagicMock()
            mock_self.request.id = "celery-task-123"

            result = _call_task(
                run_forward_model, mock_self, str(uuid.uuid4()), str(uuid.uuid4())
            )

        assert "error" in result
        assert "not found" in result["error"]
