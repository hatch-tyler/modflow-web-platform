# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MODFLOW Web Platform - A full-stack TypeScript/Python application for MODFLOW groundwater model visualization, simulation, parameter estimation, and uncertainty analysis. Docker Compose orchestration with 7 services.

## Tech Stack

- **Backend:** FastAPI (Python 3.11+), PostgreSQL + TimescaleDB, Redis, Celery, MinIO, SQLAlchemy 2.0 async
- **Frontend:** React 18 + TypeScript, Vite, Zustand, TanStack Query, Tailwind CSS
- **Visualization:** Plotly.js, Three.js/React Three Fiber, Maplibre GL, Deck.gl
- **Executables:** MODFLOW 6, MODFLOW 2005, MODFLOW NWT, MODFLOW-USG, PEST++

## Build & Development Commands

### Frontend
```bash
cd frontend
npm install
npm run dev          # Vite dev server (port 5173)
npm run build        # Production build (tsc -b && vite build)
npm run lint         # ESLint
npx vitest run       # Run all tests
npx vitest run tests/unit/someFile.test.ts  # Single test file
```

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload  # Dev server (port 8000)

# Tests
pytest                           # Run all tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests (needs services running)
USE_POSTGRES=true pytest tests/  # Integration tests against real Postgres

# Database migrations
alembic revision --autogenerate -m "message"
alembic upgrade head

# Celery worker
celery -A celery_app worker --loglevel=info --concurrency=1
```

### Docker Compose
```bash
docker-compose up -d                    # All services
docker-compose -f docker-compose.pest.yml up -d --scale pest-local-agent=8  # PEST++ agents

# Access: Web at localhost, API docs at localhost/api/v1/docs, MinIO at localhost:9001
```

## Architecture

### Data Flow for Simulations

1. **Upload:** ZIP uploaded → `upload.py` endpoint → extracts, validates with FloPy, stores in MinIO
2. **Start run:** `simulation.py` endpoint → creates Run DB record → queues Celery `simulate` task
3. **Simulate:** Celery downloads from MinIO → runs MODFLOW subprocess → uploads results back to MinIO → publishes status via Redis pub/sub
4. **Live results:** Parallel `live_results` task polls HDS/CBC during execution, caches timesteps in Redis for real-time frontend display
5. **Post-process:** `postprocess` task extracts heads/budget → creates JSON slices → stores in MinIO → updates `Run.convergence_info`
6. **View results:** Frontend polls `/results/` endpoints which check Redis cache first, then MinIO

### Backend Structure (backend/app/)
- `api/v1/router.py` - Aggregates all sub-routers; each endpoint module has its own `APIRouter`
- `api/v1/` - Endpoints: health, projects, upload, simulation, results, observations, visualization, zonebudget, pest
- `models/base.py` - Lazy engine initialization, async `get_db()` dependency, sync `SessionLocal()` for Celery
- `models/project.py` - Project and Run SQLAlchemy models
- `schemas/` - Pydantic request/response schemas
- `services/` - Business logic (see Services section below)
- `tasks/` - Celery tasks: simulate.py, postprocess.py, calibrate.py, live_results.py
- `config.py` - pydantic-settings with `@lru_cache` singleton

### Key Backend Services (backend/app/services/)
- `storage.py` - MinIO wrapper with retry logic (5 attempts, exponential backoff)
- `cache_service.py` - Two-tier caching: Redis (hot, 1-24h TTL) + MinIO (cold)
- `slice_cache.py` - Head slice cache, key pattern: `slice:{project_id}:{run_id}:L{layer}_SP{kper}_TS{kstp}`
- `modflow.py` - FloPy model parsing and validation; applies SFR2 patches before importing FloPy
- `head_extractor.py` - On-demand head slice extraction from HDS files
- `hds_streaming.py` - Streaming results for large files
- `pest_setup.py` - PEST++ PST control file generation via pyEMU
- `upload_tracker.py` - Upload stages in Redis: EXTRACTING → VALIDATING → GENERATING → COMPLETED/FAILED

### Frontend Structure (frontend/src/)
- `pages/` - Route pages: Projects, Upload, Viewer (3D), Console, Dashboard, Pest
- `components/` - Feature-organized: `dashboard/`, `upload/`, `viewer3d/`, `console/`, `layout/`
- `services/api.ts` - Axios client with namespaced API groups (projectsApi, simulationApi, resultsApi, pestApi, etc.)
- `store/projectStore.ts` - Zustand: currentProject, currentRun, sidebarOpen, uploadProgress
- `store/runManager.ts` - Zustand: active run lifecycle with SSE EventSource management, 10k-line output buffer
- `hooks/useWebSocket.ts` - WebSocket with auto-reconnect (3 attempts)
- `utils/binaryParser.ts` - Binary grid/array parsing, colormap lookup tables
- `types/index.ts` - TypeScript interfaces matching backend schemas
- Path alias: `@/*` → `src/`

### Routes (App.tsx)
```
/projects                          - Project listing
/projects/:projectId/upload        - Model upload + validation
/projects/:projectId/viewer        - 3D grid visualization
/projects/:projectId/console       - Live simulation console (SSE)
/projects/:projectId/dashboard     - Results charts and analysis
/projects/:projectId/pest          - PEST++ calibration
```

## Key Patterns

### Async/Sync Split
- **API endpoints:** Async SQLAlchemy 2.0 with asyncpg via `Depends(get_db)` (auto-commit/rollback)
- **Celery tasks:** Sync sessions via `with SessionLocal() as db:` (manual commit/rollback)
- Separate connection pools: async (10+20 overflow) and sync (5+10 overflow)

### Lazy Initialization
Database engines, Redis clients, and storage services are created on first access, not at import time. The app starts even if external services are temporarily unavailable. `wait_for_db()` retries with exponential backoff (max 30s delay). Debug mode auto-runs `init_db()` to create tables without Alembic.

### JSONB as Flexible Storage
- `Project.packages` - Which MODFLOW packages are present
- `Project.stress_period_data` - Temporal configuration
- `Run.config` - Solver/PEST settings
- `Run.convergence_info` - Post-processing status, warnings, solver metrics (not just convergence despite the name)
- `Project.validation_errors` - Model validation issues

### Celery Task Resilience
- `task_acks_late=True` + `task_reject_on_worker_lost=True` + `worker_prefetch_multiplier=1` for long-running job safety
- Worker startup (`@worker_ready.connect`): `cleanup_orphaned_runs()` marks stuck RUNNING tasks as FAILED, clears their Redis simulation locks
- Redis simulation lock prevents concurrent runs on same project
- PEST cancellation via Redis flag polling

### Frontend State Split
- **React Query:** Server state (API data with caching/refetching)
- **Zustand `projectStore`:** Cross-page UI context (current project/run, sidebar)
- **Zustand `runManager`:** Persistent SSE connections for live simulation output

### Live Updates
- Simulation console output: SSE via `EventSource` managed by `runManager`
- Upload progress: Redis-backed stage tracking polled by frontend
- Simulation status: Redis pub/sub channel `simulation:{run_id}:output`

## Database Models

**Project:** UUID PK, name, description (nullable), model_type enum (mf2005/mfnwt/mfusg/mf6/unknown), grid metadata (nlay/nrow/ncol/nper, delr/delc as JSONB), spatial reference (xoff/yoff/angrot/epsg), time_unit, length_unit, start_date, storage_path (MinIO), packages (JSONB), is_valid, validation_errors (JSONB)

**Run:** UUID PK, project_id FK (cascade delete), run_type enum (forward/pest_glm/pest_ies), status enum (pending/queued/running/completed/failed/cancelled), celery_task_id, exit_code (nullable), config (JSONB), results_path (MinIO), convergence_info (JSONB), started_at/completed_at timestamps

## MODFLOW-Specific Details

### File Extensions
- Budget files: `.cbc`, `.bud`, `.cbb`
- Head files: `.hds`, `.hed`
- MF6 entry point: `mfsim.nam` → model NAM → OC file chain

### MF6 OC File Structure
- OPTIONS block: `BUDGET FILEOUT {name}.cbc`
- PERIOD blocks: `SAVE BUDGET` or `SAVE BUDGET LAST`
- `simulate.py:enable_budget_output()` modifies OC to ensure CBC output exists

### Zone Budget
- FloPy's classic `ZoneBudget` does NOT work with MF6 CBC files (mixed 3D array + sparse node records causes IndexError)
- Custom `_compute_mf6_zone_budget()` in `zonebudget.py` handles MF6 manually
- MF6 CBC skip records: FLOW-JA-FACE, DATA-SPDIS, DATA-SAT, DATA-STOSS, DATA-STOSY
- MF6 sparse records use 1-based node indexing; convert to 0-based for zone array lookup

### FloPy Patches
- `backend/flopy_patches/apply_patches.py` monkeypatches FloPy's SFR2 module for parsing fixes
- Must be imported before FloPy in `modflow.py`
- Handles MODFLOW-USG node-based models and whitespace parsing issues

## Testing

### Backend Tests (backend/tests/)
- `conftest.py` provides fixtures: `test_client`, `db_session`, `mock_storage` (in-memory MinIO), `fake_redis`
- Default: In-memory SQLite with StaticPool; set `USE_POSTGRES=true` for real PostgreSQL
- Test DB uses custom UUID adapter for SQLite compatibility
- Sample data factories: `sample_project`, `persisted_project`, `sample_run`, `persisted_run`
- pytest markers: `slow`, `integration`, `e2e`; `asyncio_mode = auto` for async tests

### Frontend Tests (frontend/tests/)
- Vitest with jsdom environment, MSW for API mocking
- Coverage thresholds: 80% (statements, branches, functions, lines)
- Structure: `tests/unit/`, `tests/components/`, `tests/e2e/` (Playwright)
- Scripts: `npm test`, `npm run test:coverage`, `npm run test:e2e`, `npm run test:e2e:ui`

## Resource Limits

| Service | Memory Limit | Reservation | CPU Limit |
|---------|-------------|-------------|-----------|
| worker | 3GB | 1GB | 2.0 |
| api | 1GB | 256MB | - |
| frontend | 512MB | 128MB | - |
| postgres | 1GB | 256MB | - |
| redis | 256MB | 64MB | - |
| minio | 768MB | 256MB | - |
| nginx | 128MB | 32MB | - |

Worker runs concurrency=1 to prevent OOM on large MODFLOW models.

## Environment Configuration

Key variables in `.env` (see `.env.example`): POSTGRES_USER/PASSWORD/DB, REDIS_PASSWORD, MINIO_ACCESS_KEY/SECRET_KEY, MODFLOW_EXE_PATH, PESTPP_EXE_PATH, PEST_LOCAL_AGENTS (default 4), MAX_UPLOAD_SIZE_MB (500), MAX_MODEL_FILES (5000)

### Storage Paths
- MinIO buckets: `modflow-models` (uploaded ZIPs), `modflow-results` (post-processed output)
- Worker temp: `/tmp/modflow-runs` (downloaded models during simulation)
- PEST workspace: `/tmp/pest-workspace` (shared volume between worker and PEST agents)

## Infrastructure Notes

- Nginx reverse proxy routes: `/api/` → backend, `/` → frontend, WebSocket upgrade for `/api/v1/runs/`
- Nginx timeouts: `proxy_send_timeout 1800s` / `proxy_read_timeout 1800s` for API routes (large models with 4000+ files can take 15+ minutes for upload + validation)
- PEST++ supports three modes: local processes, local containers (`docker-compose.pest.yml`), distributed network (`docker-compose.agent.yml`); manager port 4004 exposed for remote agents
- Frontend Vite dev server proxies `/api` → `localhost:8000` and `/ws` → `ws://localhost:8000`; uses `watch.usePolling` for Docker file watching
- Frontend build requires `NODE_OPTIONS="--max-old-space-size=4096"` due to Plotly.js bundle size
- Docker service health checks: API/worker at 30s intervals; worker has 60s `start_period` for slow startup
- All services on single `modflow-network` bridge; PEST compose files reference it as external
