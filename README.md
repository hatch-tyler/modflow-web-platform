# MODFLOW Web Platform

A full-stack web application for uploading, validating, simulating, and analyzing MODFLOW groundwater models. Supports MODFLOW 6, MODFLOW-2005, MODFLOW-NWT, and MODFLOW-USG with PEST++ integration for parameter estimation and uncertainty analysis.

## Features

- **Model Upload & Validation** - Upload ZIP archives of MODFLOW models; automatic type detection, FloPy parsing, grid metadata extraction, and cross-platform path normalization
- **3D Visualization** - Interactive grid rendering with Three.js/React Three Fiber, boundary condition overlays, head contours, and cross-section views
- **Simulation Execution** - Run MODFLOW from the browser with live console output via SSE, automatic budget output configuration, and background Celery task management
- **Live Results** - Real-time head and budget data streamed during simulation via Redis pub/sub, cached for instant dashboard access
- **Results Dashboard** - Head maps, drawdown, water budget bar charts, time series, and obs-vs-sim scatter plots powered by Plotly.js
- **Zone Budget** - Draw zones on the grid and compute volumetric water budgets per zone using FloPy ZoneBudget (classic models) or a custom MF6 implementation
- **PEST++ Calibration** - Generate PEST control files via pyEMU, run PEST++ GLM/IES with local or distributed parallel agents, and monitor calibration progress in real time
- **Geospatial** - Automatic spatial reference extraction (EPSG, offsets, rotation) with Maplibre GL and Deck.gl map overlays

## Architecture

```
                        NGINX (reverse proxy)
                       /                      \
              React SPA                  FastAPI API
          (Vite + TypeScript)          (async, Python 3.11)
                                      /       |       \
                              PostgreSQL    Redis     MinIO
                             + TimescaleDB  broker   object store
                                              |
                                       Celery Workers
                                    (MODFLOW 6, PEST++)
```

**Services** (Docker Compose):
| Service | Role |
|---------|------|
| `nginx` | Reverse proxy, TLS termination, security headers |
| `frontend` | React 18 SPA (Vite dev server or static build) |
| `api` | FastAPI with async SQLAlchemy, rate limiting (slowapi) |
| `worker` | Celery (concurrency=1) running MODFLOW/PEST++ subprocesses |
| `postgres` | TimescaleDB (PostgreSQL 16) for projects, runs, metadata |
| `redis` | Task broker, pub/sub for live output, result caching |
| `minio` | S3-compatible object storage for model files and results |

## Quick Start

### Prerequisites

- Docker and Docker Compose v2
- (Optional) Node.js 20+ and Python 3.11+ for local development

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env - at minimum set strong passwords for production
```

### 2. Start all services

```bash
docker-compose up -d
```

### 3. Access the application

| URL | Description |
|-----|-------------|
| http://localhost | Web application |
| http://localhost/api/v1/docs | Interactive API documentation |
| http://localhost:9001 | MinIO console (dev only) |

### Production deployment

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

The production override disables debug mode, removes exposed dev ports, and requires strong passwords via environment variables.

## Local Development

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Run tests:**
```bash
# Backend
cd backend && pytest

# Frontend
cd frontend && npx vitest run
```

## Project Structure

```
model-app/
├── backend/
│   ├── app/
│   │   ├── api/v1/        # REST endpoints (health, upload, simulation, results, pest, etc.)
│   │   ├── models/        # SQLAlchemy ORM (Project, Run with UUID PKs, JSONB fields)
│   │   ├── schemas/       # Pydantic request/response models
│   │   ├── services/      # Business logic (storage, caching, FloPy parsing, PEST setup)
│   │   └── tasks/         # Celery tasks (simulate, postprocess, live_results, calibrate)
│   ├── alembic/           # Database migrations
│   ├── flopy_patches/     # SFR2 parser fixes applied before FloPy import
│   ├── scripts/           # Health checks, PEST agent deployment scripts
│   └── tests/             # pytest (unit, integration, e2e)
├── frontend/
│   ├── src/
│   │   ├── components/    # Feature-organized: dashboard/, viewer3d/, console/, upload/
│   │   ├── pages/         # Route pages (Projects, Upload, Viewer, Console, Dashboard, Pest)
│   │   ├── services/      # Axios API client with namespaced groups
│   │   ├── store/         # Zustand (projectStore, runManager with SSE auto-reconnect)
│   │   ├── hooks/         # WebSocket, custom React hooks
│   │   ├── types/         # TypeScript interfaces matching backend schemas
│   │   └── utils/         # Binary parsers, colormap LUTs
│   └── tests/             # Vitest unit + Playwright e2e
├── nginx/                 # Reverse proxy config with security headers
├── scripts/               # PEST++ distributed agent deployment
├── test-models/           # Small test MODFLOW models (MF6 + MF2005)
├── docker-compose.yml     # Development orchestration (7 services)
├── docker-compose.prod.yml      # Production overrides
├── docker-compose.pest.yml      # PEST++ local container agents
├── docker-compose.agent.yml     # PEST++ remote/distributed agents
└── CLAUDE.md              # AI assistant context and conventions
```

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | React 18, TypeScript, Vite, Zustand, TanStack Query, Tailwind CSS |
| Visualization | Plotly.js, Three.js / React Three Fiber, Maplibre GL, Deck.gl |
| Backend | FastAPI, SQLAlchemy 2.0 (async), Celery, slowapi (rate limiting) |
| Database | PostgreSQL 16 + TimescaleDB |
| Cache/Queue | Redis 7 (with authentication) |
| Storage | MinIO (S3-compatible) |
| Proxy | NGINX with CSP, HSTS, and security headers |
| MODFLOW | FloPy, MODFLOW 6, MODFLOW-2005, MODFLOW-NWT, MODFLOW-USG |
| Calibration | PEST++ (GLM, IES), pyEMU |
| CI/CD | GitHub Actions (unit, integration, e2e, API contract, coverage) |

## Environment Variables

See [`.env.example`](.env.example) for all configurable variables. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_PASSWORD` | - | PostgreSQL password (required in production) |
| `REDIS_PASSWORD` | `redis_secret` | Redis authentication password |
| `MINIO_SECRET_KEY` | - | MinIO secret key (required in production) |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum upload file size |
| `PEST_LOCAL_AGENTS` | `4` | Number of parallel PEST++ worker agents |

## License

MIT
