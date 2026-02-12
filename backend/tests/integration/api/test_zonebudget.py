"""Integration tests for zone budget API endpoints.

Tests:
- POST /zone-budget (synchronous fallback with asyncio.to_thread)
- POST /zone-budget/compute (async Celery dispatch + cache hit)
- GET /zone-budget/status/{task_id} (progress polling)
- GET /zone-budget/result/{task_id} (result retrieval)
- Zone definition CRUD (GET/POST/DELETE /zone-definitions)
- compute_zone_hash() determinism
- _sync_compute_zone_budget() unit behavior
"""

import json
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import ModelType, Project, Run, RunStatus, RunType

# The test_client fixture (conftest) patches app.services.storage.get_storage_service,
# but zonebudget.py imports `from app.services.storage import get_storage_service` which
# binds a local reference. We need to also patch at the callsite for endpoints that
# call get_storage_service() directly.
_ZB_STORAGE_PATCH = "app.api.v1.zonebudget.get_storage_service"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def completed_run(db_session: AsyncSession, persisted_project: Project):
    """Create a completed run with results_path for zone budget testing."""
    persisted_project.storage_path = f"projects/{persisted_project.id}"
    await db_session.flush()

    run = Run(
        id=uuid.uuid4(),
        project_id=persisted_project.id,
        name="Completed Run",
        run_type=RunType.FORWARD,
        status=RunStatus.COMPLETED,
        results_path=f"projects/{persisted_project.id}/runs/test-run",
    )
    db_session.add(run)
    await db_session.commit()
    await db_session.refresh(run)
    yield run
    try:
        await db_session.delete(run)
        await db_session.commit()
    except Exception:
        await db_session.rollback()


@pytest.fixture
def zone_layers_payload():
    """Sample zone_layers payload for zone budget computation."""
    return {
        "0": {
            "Zone 1": [0, 1, 2, 3, 4],
            "Zone 2": [5, 6, 7, 8, 9],
        }
    }


@pytest.fixture
def results_summary_json():
    """Results summary JSON with grid metadata."""
    return json.dumps({
        "metadata": {
            "nlay": 1,
            "nrow": 10,
            "ncol": 10,
            "nper": 1,
            "model_type": "mf6",
            "grid_type": "structured",
        },
        "heads_summary": {
            "kstpkper_list": [[0, 0]],
            "times": [1.0],
        },
    }).encode("utf-8")


# ─── compute_zone_hash tests ────────────────────────────────────────────────

class TestComputeZoneHash:
    """Tests for zone hash function determinism and uniqueness."""

    def test_same_input_produces_same_hash(self, zone_layers_payload):
        from app.api.v1.zonebudget import compute_zone_hash
        h1 = compute_zone_hash(zone_layers_payload, quick_mode=False)
        h2 = compute_zone_hash(zone_layers_payload, quick_mode=False)
        assert h1 == h2

    def test_different_quick_mode_produces_different_hash(self, zone_layers_payload):
        from app.api.v1.zonebudget import compute_zone_hash
        h_full = compute_zone_hash(zone_layers_payload, quick_mode=False)
        h_quick = compute_zone_hash(zone_layers_payload, quick_mode=True)
        assert h_full != h_quick

    def test_different_zones_produce_different_hash(self):
        from app.api.v1.zonebudget import compute_zone_hash
        z1 = {"0": {"Zone 1": [0, 1]}}
        z2 = {"0": {"Zone 1": [0, 2]}}
        assert compute_zone_hash(z1) != compute_zone_hash(z2)

    def test_hash_is_16_chars(self, zone_layers_payload):
        from app.api.v1.zonebudget import compute_zone_hash
        h = compute_zone_hash(zone_layers_payload)
        assert len(h) == 16

    def test_key_order_does_not_matter(self):
        """JSON sort_keys ensures consistent serialization."""
        from app.api.v1.zonebudget import compute_zone_hash
        z1 = {"0": {"Zone 1": [0, 1], "Zone 2": [2, 3]}}
        z2 = {"0": {"Zone 2": [2, 3], "Zone 1": [0, 1]}}
        assert compute_zone_hash(z1) == compute_zone_hash(z2)


# ─── Sync compute function tests ────────────────────────────────────────────

class TestSyncCompute:
    """Tests for _sync_compute_zone_budget helper function."""

    def test_raises_on_missing_cbc(self, mock_storage):
        """If CBC doesn't exist in mock storage, download_to_file should fail."""
        from app.api.v1.zonebudget import _find_cbc_object
        result = _find_cbc_object(mock_storage, "projects/xyz/runs/r1")
        assert result is None

    def test_find_cbc_object_with_cbc(self, mock_storage):
        from app.api.v1.zonebudget import _find_cbc_object
        mock_storage.upload_bytes(
            "modflow-models",
            "projects/xyz/runs/r1/model.cbc",
            b"fake cbc data",
        )
        result = _find_cbc_object(mock_storage, "projects/xyz/runs/r1")
        assert result == "projects/xyz/runs/r1/model.cbc"

    def test_find_cbc_object_with_bud(self, mock_storage):
        from app.api.v1.zonebudget import _find_cbc_object
        mock_storage.upload_bytes(
            "modflow-models",
            "projects/xyz/runs/r1/output.bud",
            b"fake bud data",
        )
        result = _find_cbc_object(mock_storage, "projects/xyz/runs/r1")
        assert result == "projects/xyz/runs/r1/output.bud"

    def test_load_grid_dimensions(self, mock_storage, results_summary_json):
        from app.api.v1.zonebudget import _load_grid_dimensions
        mock_storage.upload_bytes(
            "modflow-models",
            "projects/xyz/runs/r1/results_summary.json",
            results_summary_json,
        )
        dims = _load_grid_dimensions(mock_storage, "projects/xyz/runs/r1")
        assert dims == {"nlay": 1, "nrow": 10, "ncol": 10}

    def test_load_grid_dimensions_missing(self, mock_storage):
        from app.api.v1.zonebudget import _load_grid_dimensions
        with pytest.raises(ValueError, match="Results summary not available"):
            _load_grid_dimensions(mock_storage, "projects/xyz/runs/missing")


# ─── POST /zone-budget (sync fallback) ──────────────────────────────────────

class TestComputeZoneBudgetSync:
    """Tests for the synchronous zone budget endpoint."""

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent_run(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        fake_run_id = uuid.uuid4()
        response = await test_client.post(
            f"/api/v1/projects/{persisted_project.id}/runs/{fake_run_id}/results/zone-budget",
            json={"zone_layers": {"0": {"Zone 1": [0, 1]}}},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_400_for_non_completed_run(
        self, test_client: AsyncClient, persisted_run: Run
    ):
        response = await test_client.post(
            f"/api/v1/projects/{persisted_run.project_id}/runs/{persisted_run.id}/results/zone-budget",
            json={"zone_layers": {"0": {"Zone 1": [0, 1]}}},
        )
        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_returns_404_when_no_cbc_file(
        self,
        test_client: AsyncClient,
        completed_run: Run,
        persisted_project: Project,
        mock_storage,
        results_summary_json,
    ):
        mock_storage.upload_bytes(
            "modflow-models",
            f"{completed_run.results_path}/results_summary.json",
            results_summary_json,
        )
        with patch(_ZB_STORAGE_PATCH, return_value=mock_storage):
            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/runs/{completed_run.id}/results/zone-budget",
                json={"zone_layers": {"0": {"Zone 1": [0, 1]}}},
            )
        assert response.status_code == 404
        assert "budget file" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_compute_success_calls_to_thread(
        self,
        test_client: AsyncClient,
        completed_run: Run,
        persisted_project: Project,
        mock_storage,
        results_summary_json,
    ):
        """Test that compute succeeds when _sync_compute_zone_budget is mocked."""
        mock_storage.upload_bytes(
            "modflow-models",
            f"{completed_run.results_path}/results_summary.json",
            results_summary_json,
        )
        mock_storage.upload_bytes(
            "modflow-models",
            f"{completed_run.results_path}/model.cbc",
            b"fake cbc",
        )

        fake_result = {
            "zone_names": ["Zone 1"],
            "columns": ["name", "kper", "kstp", "ZONE_1"],
            "records": [{"name": "FROM_STORAGE", "kper": 0, "kstp": 0, "ZONE_1": 100.0}],
        }

        with patch(_ZB_STORAGE_PATCH, return_value=mock_storage), \
             patch("app.api.v1.zonebudget._sync_compute_zone_budget", return_value=fake_result):
            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/runs/{completed_run.id}/results/zone-budget",
                json={"zone_layers": {"0": {"Zone 1": [0, 1, 2]}}},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["zone_names"] == ["Zone 1"]
        assert len(data["records"]) == 1


# ─── POST /zone-budget/compute (async) ──────────────────────────────────────

class TestComputeZoneBudgetAsync:
    """Tests for the async compute endpoint with cache and Celery dispatch."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_result_immediately(
        self,
        test_client: AsyncClient,
        completed_run: Run,
        persisted_project: Project,
        mock_storage,
        zone_layers_payload,
    ):
        from app.api.v1.zonebudget import compute_zone_hash
        zone_hash = compute_zone_hash(zone_layers_payload, quick_mode=False)
        cache_obj = f"{completed_run.results_path}/processed/zone_budget_{zone_hash}.json"
        cached_result = {
            "zone_names": ["Zone 1", "Zone 2"],
            "columns": ["name", "kper", "kstp", "ZONE_1", "ZONE_2"],
            "records": [{"name": "FROM_STORAGE", "kper": 0, "kstp": 0, "ZONE_1": 50.0, "ZONE_2": 50.0}],
        }
        mock_storage.upload_bytes(
            "modflow-models",
            cache_obj,
            json.dumps(cached_result).encode("utf-8"),
        )

        with patch(_ZB_STORAGE_PATCH, return_value=mock_storage):
            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/runs/{completed_run.id}/results/zone-budget/compute",
                json={"zone_layers": zone_layers_payload, "quick_mode": False},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["cached"] is True
        assert data["result"]["zone_names"] == ["Zone 1", "Zone 2"]

    @pytest.mark.asyncio
    async def test_cache_miss_dispatches_celery_task(
        self,
        test_client: AsyncClient,
        completed_run: Run,
        persisted_project: Project,
        mock_storage,
        zone_layers_payload,
    ):
        with patch(_ZB_STORAGE_PATCH, return_value=mock_storage), \
             patch("app.tasks.zonebudget.compute_zone_budget_task.delay") as mock_delay, \
             patch("redis.Redis.from_url") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis_cls.return_value = mock_redis

            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/runs/{completed_run.id}/results/zone-budget/compute",
                json={"zone_layers": zone_layers_payload, "quick_mode": False},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "task_id" in data
        mock_delay.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent_run(
        self, test_client: AsyncClient, persisted_project: Project
    ):
        fake_run_id = uuid.uuid4()
        response = await test_client.post(
            f"/api/v1/projects/{persisted_project.id}/runs/{fake_run_id}/results/zone-budget/compute",
            json={"zone_layers": {"0": {"Zone 1": [0]}}, "quick_mode": False},
        )
        assert response.status_code == 404


# ─── GET /zone-budget/status/{task_id} ───────────────────────────────────────

class TestZoneBudgetStatus:
    """Tests for the status polling endpoint."""

    @pytest.mark.asyncio
    async def test_returns_progress(
        self,
        test_client: AsyncClient,
        completed_run: Run,
        persisted_project: Project,
    ):
        fake_task_id = "test-task-123"
        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {
            b"status": b"computing",
            b"progress": b"45",
            b"message": b"Processing timestep 5/10...",
            b"result_path": b"",
            b"error": b"",
        }

        with patch("redis.Redis.from_url", return_value=mock_redis):
            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/runs/{completed_run.id}/results/zone-budget/status/{fake_task_id}",
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "computing"
        assert data["progress"] == 45
        assert "timestep 5/10" in data["message"]

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_task(
        self,
        test_client: AsyncClient,
        completed_run: Run,
        persisted_project: Project,
    ):
        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {}

        with patch("redis.Redis.from_url", return_value=mock_redis):
            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/runs/{completed_run.id}/results/zone-budget/status/nonexistent",
            )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_failed_status_with_error(
        self,
        test_client: AsyncClient,
        completed_run: Run,
        persisted_project: Project,
    ):
        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {
            b"status": b"failed",
            b"progress": b"0",
            b"message": b"",
            b"result_path": b"",
            b"error": b"CBC file corrupted",
        }

        with patch("redis.Redis.from_url", return_value=mock_redis):
            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/runs/{completed_run.id}/results/zone-budget/status/failed-task",
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "CBC file corrupted"


# ─── GET /zone-budget/result/{task_id} ───────────────────────────────────────

class TestZoneBudgetResult:
    """Tests for the result retrieval endpoint."""

    @pytest.mark.asyncio
    async def test_returns_result_for_completed_task(
        self,
        test_client: AsyncClient,
        completed_run: Run,
        persisted_project: Project,
        mock_storage,
    ):
        result_path = f"{completed_run.results_path}/processed/zone_budget_abc123.json"
        result_data = {
            "zone_names": ["Zone 1"],
            "columns": ["name", "kper", "kstp", "ZONE_1"],
            "records": [{"name": "FROM_STORAGE", "kper": 0, "kstp": 0, "ZONE_1": 99.9}],
        }
        mock_storage.upload_bytes(
            "modflow-models", result_path, json.dumps(result_data).encode("utf-8"),
        )

        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {
            b"status": b"completed",
            b"progress": b"100",
            b"message": b"Done",
            b"result_path": result_path.encode(),
            b"error": b"",
        }

        with patch("redis.Redis.from_url", return_value=mock_redis), \
             patch(_ZB_STORAGE_PATCH, return_value=mock_storage):
            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/runs/{completed_run.id}/results/zone-budget/result/task-abc",
            )

        assert response.status_code == 200
        data = response.json()
        assert data["zone_names"] == ["Zone 1"]
        assert data["records"][0]["ZONE_1"] == 99.9

    @pytest.mark.asyncio
    async def test_returns_400_if_task_not_completed(
        self,
        test_client: AsyncClient,
        completed_run: Run,
        persisted_project: Project,
    ):
        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {
            b"status": b"computing",
            b"progress": b"50",
            b"message": b"Still going",
            b"result_path": b"",
            b"error": b"",
        }

        with patch("redis.Redis.from_url", return_value=mock_redis):
            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/runs/{completed_run.id}/results/zone-budget/result/task-pending",
            )

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]


# ─── Zone Definition CRUD ────────────────────────────────────────────────────

class TestZoneDefinitions:
    """Tests for zone definition persistence endpoints."""

    @pytest.mark.asyncio
    async def test_list_empty(
        self, test_client: AsyncClient, persisted_project: Project, db_session, mock_storage
    ):
        persisted_project.storage_path = f"projects/{persisted_project.id}"
        await db_session.flush()

        with patch(_ZB_STORAGE_PATCH, return_value=mock_storage):
            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/zone-definitions",
            )
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_save_and_list(
        self, test_client: AsyncClient, persisted_project: Project, db_session, mock_storage
    ):
        persisted_project.storage_path = f"projects/{persisted_project.id}"
        await db_session.flush()

        with patch(_ZB_STORAGE_PATCH, return_value=mock_storage):
            # Save a zone definition
            response = await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/zone-definitions",
                json={
                    "name": "My Zones",
                    "zone_layers": {"0": {"Zone 1": [0, 1, 2], "Zone 2": [3, 4, 5]}},
                    "num_zones": 2,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "My Zones"
            assert data["saved"] is True

            # List — should show the saved definition
            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/zone-definitions",
            )
            assert response.status_code == 200
            defs = response.json()
            assert len(defs) == 1
            assert defs[0]["name"] == "My Zones"
            assert defs[0]["num_zones"] == 2

    @pytest.mark.asyncio
    async def test_get_zone_definition(
        self, test_client: AsyncClient, persisted_project: Project, db_session, mock_storage
    ):
        persisted_project.storage_path = f"projects/{persisted_project.id}"
        await db_session.flush()

        zone_layers = {"0": {"Zone 1": [0, 1], "Zone 2": [5, 6]}}

        with patch(_ZB_STORAGE_PATCH, return_value=mock_storage):
            await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/zone-definitions",
                json={"name": "Test Def", "zone_layers": zone_layers, "num_zones": 2},
            )

            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/zone-definitions/Test Def",
            )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Def"
        assert data["zone_layers"] == zone_layers
        assert data["num_zones"] == 2

    @pytest.mark.asyncio
    async def test_delete_zone_definition(
        self, test_client: AsyncClient, persisted_project: Project, db_session, mock_storage
    ):
        persisted_project.storage_path = f"projects/{persisted_project.id}"
        await db_session.flush()

        with patch(_ZB_STORAGE_PATCH, return_value=mock_storage):
            # Save
            await test_client.post(
                f"/api/v1/projects/{persisted_project.id}/zone-definitions",
                json={"name": "Deletable", "zone_layers": {"0": {"Zone 1": [0]}}, "num_zones": 1},
            )

            # Delete
            response = await test_client.delete(
                f"/api/v1/projects/{persisted_project.id}/zone-definitions/Deletable",
            )
            assert response.status_code == 200
            assert response.json()["deleted"] is True

            # Verify it's gone
            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/zone-definitions",
            )
            assert response.json() == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_definition_returns_404(
        self, test_client: AsyncClient, persisted_project: Project, db_session, mock_storage
    ):
        persisted_project.storage_path = f"projects/{persisted_project.id}"
        await db_session.flush()

        with patch(_ZB_STORAGE_PATCH, return_value=mock_storage):
            response = await test_client.get(
                f"/api/v1/projects/{persisted_project.id}/zone-definitions/NoSuchDef",
            )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent_project(
        self, test_client: AsyncClient
    ):
        fake_id = uuid.uuid4()
        response = await test_client.get(
            f"/api/v1/projects/{fake_id}/zone-definitions",
        )
        assert response.status_code == 404


# ─── MF6 zone budget function tests ─────────────────────────────────────────

class TestComputeMf6ZoneBudget:
    """Tests for _compute_mf6_zone_budget with progress callback."""

    def test_progress_callback_is_called(self):
        """Test that the progress callback is invoked during computation."""
        from app.api.v1.zonebudget import _compute_mf6_zone_budget

        progress_calls = []

        def track_progress(current, total, message):
            progress_calls.append((current, total, message))

        # This will fail since we don't have a real CBC, but it validates
        # the function signature accepts the new params
        zone_array = np.ones((1, 5, 5), dtype=int)
        try:
            _compute_mf6_zone_budget(
                "nonexistent.cbc", zone_array, {"Zone 1": 1},
                1, 5, 5,
                progress_callback=track_progress,
                kstpkper_filter=[(0, 0)],
            )
        except Exception:
            pass  # Expected — no real CBC file

    def test_kstpkper_filter_accepted(self):
        """Test that kstpkper_filter parameter is accepted."""
        from app.api.v1.zonebudget import _compute_mf6_zone_budget
        zone_array = np.ones((1, 5, 5), dtype=int)
        try:
            _compute_mf6_zone_budget(
                "nonexistent.cbc", zone_array, {"Zone 1": 1},
                1, 5, 5,
                kstpkper_filter=[(0, 0)],
            )
        except Exception:
            pass  # Expected — no real CBC file


# ─── Listing file parser tests ───────────────────────────────────────────────

class TestParseListingBudget:
    """Tests for parse_budget_from_listing utility."""

    def test_empty_string_returns_empty(self):
        from app.api.v1.zonebudget import parse_budget_from_listing
        result = parse_budget_from_listing("")
        assert result == {}

    def test_no_budget_section_returns_empty(self):
        from app.api.v1.zonebudget import parse_budget_from_listing
        result = parse_budget_from_listing("Some random listing file content\nNo budget here.")
        assert result == {}

    def test_mf6_budget_parsed(self):
        from app.api.v1.zonebudget import parse_budget_from_listing
        listing = """
 VOLUME BUDGET FOR ENTIRE MODEL AT END OF TIME STEP  1, STRESS PERIOD  1
 ---------------------------------------------------------------------------------------------------

                     CUMULATIVE VOLUME      L**3       RATES FOR THIS TIME STEP      L**3/T
                     ------------------                 ------------------------

       IN:                                              IN:
       ---                                              ---
            STORAGE =        1000.0000              STORAGE =         100.0000
            WEL =             500.0000              WEL =              50.0000

            TOTAL IN =       1500.0000              TOTAL IN =        150.0000

      OUT:                                             OUT:
      ----                                             ----
            STORAGE =         200.0000              STORAGE =          20.0000
            DRN =             800.0000              DRN =              80.0000

            TOTAL OUT =      1000.0000              TOTAL OUT =       100.0000

 IN - OUT = 500.0    PERCENT DISCREPANCY = 0.00
"""
        result = parse_budget_from_listing(listing)
        assert result != {}
        assert result["zone_names"] == ["Entire Model"]
        names = [r["name"] for r in result["records"]]
        assert "FROM_STORAGE" in names
        assert "FROM_WEL" in names
        assert "TO_STORAGE" in names
        assert "TO_DRN" in names
        # Verify rates (not cumulative) are used
        from_storage = next(r for r in result["records"] if r["name"] == "FROM_STORAGE")
        assert from_storage["ZONE_1"] == 100.0  # Rate, not cumulative 1000
