/**
 * MSW request handlers for API mocking
 *
 * These handlers intercept API requests during tests and return mock responses.
 * Organized by API domain (health, projects, simulation, results, pest).
 */

import { http, HttpResponse, delay } from 'msw';
import type {
  Project,
  Run,
  HealthCheck,
  ResultsSummary,
  HeadSlice,
  TimestepIndex,
} from '../../src/types';

const API_BASE = '/api/v1';

// ─── Mock Data ──────────────────────────────────────────────────────────────

export const mockProjects: Project[] = [
  {
    id: '550e8400-e29b-41d4-a716-446655440001',
    name: 'Test Model 1',
    description: 'A simple MODFLOW 6 test model',
    model_type: 'mf6',
    nlay: 3,
    nrow: 100,
    ncol: 100,
    nper: 12,
    is_valid: true,
    created_at: '2024-01-15T10:30:00Z',
    updated_at: '2024-01-15T14:45:00Z',
  },
  {
    id: '550e8400-e29b-41d4-a716-446655440002',
    name: 'Test Model 2',
    description: 'Another test model',
    model_type: 'mf2005',
    nlay: 1,
    nrow: 50,
    ncol: 50,
    nper: 1,
    is_valid: true,
    created_at: '2024-01-10T08:00:00Z',
    updated_at: '2024-01-10T08:00:00Z',
  },
];

export const mockRuns: Run[] = [
  {
    id: '660e8400-e29b-41d4-a716-446655440001',
    project_id: '550e8400-e29b-41d4-a716-446655440001',
    name: 'Forward Run 1',
    run_type: 'forward',
    status: 'completed',
    started_at: '2024-01-15T15:00:00Z',
    completed_at: '2024-01-15T15:05:00Z',
    exit_code: 0,
    created_at: '2024-01-15T15:00:00Z',
    updated_at: '2024-01-15T15:05:00Z',
  },
  {
    id: '660e8400-e29b-41d4-a716-446655440002',
    project_id: '550e8400-e29b-41d4-a716-446655440001',
    name: 'Forward Run 2',
    run_type: 'forward',
    status: 'running',
    started_at: '2024-01-15T16:00:00Z',
    created_at: '2024-01-15T16:00:00Z',
    updated_at: '2024-01-15T16:00:00Z',
  },
];

export const mockHealthCheck: HealthCheck = {
  status: 'healthy',
  version: '0.1.0',
  database: 'healthy',
  redis: 'healthy',
  minio: 'healthy',
};

export const mockTimestepIndex: TimestepIndex = {
  timesteps: [
    { kstpkper: [0, 0], totim: 1.0 },
    { kstpkper: [0, 1], totim: 30.0 },
    { kstpkper: [0, 2], totim: 60.0 },
  ],
  nlay: 3,
  nrow: 100,
  ncol: 100,
};

export const mockHeadSlice: HeadSlice = {
  layer: 0,
  kper: 0,
  kstp: 0,
  heads: Array(100).fill(null).map(() =>
    Array(100).fill(null).map(() => 45 + Math.random() * 10)
  ),
  nrow: 100,
  ncol: 100,
  min_head: 45.0,
  max_head: 55.0,
};

export const mockResultsSummary: ResultsSummary = {
  has_heads: true,
  has_budget: true,
  head_file: 'model.hds',
  budget_file: 'model.cbc',
  nlay: 3,
  nrow: 100,
  ncol: 100,
  nper: 12,
};

// ─── Request Handlers ───────────────────────────────────────────────────────

export const handlers = [
  // Health endpoints
  http.get(`${API_BASE}/health`, () => {
    return HttpResponse.json(mockHealthCheck);
  }),

  http.get(`${API_BASE}/health/live`, () => {
    return HttpResponse.json({ status: 'alive' });
  }),

  http.get(`${API_BASE}/health/ready`, () => {
    return HttpResponse.json({ status: 'ready' });
  }),

  // Projects endpoints
  http.get(`${API_BASE}/projects`, () => {
    return HttpResponse.json(mockProjects);
  }),

  http.get(`${API_BASE}/projects/:projectId`, ({ params }) => {
    const project = mockProjects.find((p) => p.id === params.projectId);
    if (!project) {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json(project);
  }),

  http.post(`${API_BASE}/projects`, async ({ request }) => {
    const body = (await request.json()) as { name: string; description?: string };
    const newProject: Project = {
      id: crypto.randomUUID(),
      name: body.name,
      description: body.description || null,
      model_type: null,
      nlay: null,
      nrow: null,
      ncol: null,
      nper: null,
      is_valid: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    return HttpResponse.json(newProject, { status: 201 });
  }),

  http.patch(`${API_BASE}/projects/:projectId`, async ({ params, request }) => {
    const body = (await request.json()) as Partial<Project>;
    const project = mockProjects.find((p) => p.id === params.projectId);
    if (!project) {
      return new HttpResponse(null, { status: 404 });
    }
    const updated = { ...project, ...body, updated_at: new Date().toISOString() };
    return HttpResponse.json(updated);
  }),

  http.delete(`${API_BASE}/projects/:projectId`, ({ params }) => {
    const projectIndex = mockProjects.findIndex((p) => p.id === params.projectId);
    if (projectIndex === -1) {
      return new HttpResponse(null, { status: 404 });
    }
    return new HttpResponse(null, { status: 204 });
  }),

  // Upload endpoints
  http.post(`${API_BASE}/projects/:projectId/upload`, async () => {
    await delay(100); // Simulate upload delay
    return HttpResponse.json({
      is_valid: true,
      model_type: 'mf6',
      errors: [],
      warnings: [],
      grid: { nlay: 3, nrow: 100, ncol: 100, nper: 12 },
    });
  }),

  http.get(`${API_BASE}/projects/:projectId/upload/status`, () => {
    return HttpResponse.json(null);
  }),

  // Simulation endpoints
  http.get(`${API_BASE}/projects/:projectId/simulation/runs`, ({ params }) => {
    const runs = mockRuns.filter((r) => r.project_id === params.projectId);
    return HttpResponse.json(runs);
  }),

  http.get(`${API_BASE}/projects/:projectId/simulation/runs/:runId`, ({ params }) => {
    const run = mockRuns.find((r) => r.id === params.runId);
    if (!run) {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json(run);
  }),

  http.post(`${API_BASE}/projects/:projectId/simulation/run`, async ({ params }) => {
    const runId = crypto.randomUUID();
    return HttpResponse.json({
      run_id: runId,
      task_id: crypto.randomUUID(),
      status: 'queued',
      message: 'Simulation queued for execution',
    });
  }),

  http.post(`${API_BASE}/projects/:projectId/simulation/runs/:runId/cancel`, ({ params }) => {
    return HttpResponse.json({
      run_id: params.runId,
      status: 'cancelled',
      message: 'Simulation cancelled',
    });
  }),

  http.get(`${API_BASE}/projects/:projectId/simulation/status`, ({ params }) => {
    const runs = mockRuns.filter((r) => r.project_id === params.projectId);
    return HttpResponse.json({
      has_runs: runs.length > 0,
      latest_run: runs.length > 0 ? runs[0] : null,
    });
  }),

  // Results endpoints
  http.get(`${API_BASE}/projects/:projectId/runs/:runId/results/summary`, () => {
    return HttpResponse.json(mockResultsSummary);
  }),

  http.get(`${API_BASE}/projects/:projectId/runs/:runId/results/heads/available`, () => {
    return HttpResponse.json(mockTimestepIndex);
  }),

  http.get(`${API_BASE}/projects/:projectId/runs/:runId/results/heads`, () => {
    return HttpResponse.json(mockHeadSlice);
  }),

  http.get(`${API_BASE}/projects/:projectId/runs/:runId/results/heads/statistics`, () => {
    return HttpResponse.json({
      min_head: 45.0,
      max_head: 55.0,
      layers: [
        { layer: 0, min: 45.0, max: 55.0 },
        { layer: 1, min: 44.0, max: 54.0 },
        { layer: 2, min: 43.0, max: 53.0 },
      ],
    });
  }),

  http.get(`${API_BASE}/projects/:projectId/runs/:runId/results/budget`, () => {
    return HttpResponse.json({
      components: ['STORAGE', 'WELLS', 'RECHARGE', 'DRAINS'],
      timesteps: [
        {
          kstpkper: [0, 0],
          totim: 1.0,
          in: { STORAGE: 100, RECHARGE: 500, WELLS: 0, DRAINS: 0 },
          out: { STORAGE: 0, WELLS: 200, RECHARGE: 0, DRAINS: 400 },
        },
      ],
    });
  }),

  http.get(`${API_BASE}/projects/:projectId/runs/:runId/results/timeseries`, () => {
    return HttpResponse.json({
      layer: 0,
      row: 50,
      col: 50,
      times: [1.0, 30.0, 60.0],
      heads: [50.0, 49.5, 49.0],
    });
  }),

  http.get(`${API_BASE}/projects/:projectId/runs/:runId/results/grid-geometry`, () => {
    return HttpResponse.json({
      nlay: 3,
      nrow: 100,
      ncol: 100,
      delr: Array(100).fill(100),
      delc: Array(100).fill(100),
      top: Array(100 * 100).fill(100),
      botm: [
        Array(100 * 100).fill(90),
        Array(100 * 100).fill(80),
        Array(100 * 100).fill(70),
      ],
    });
  }),

  // PEST endpoints
  http.get(`${API_BASE}/projects/:projectId/pest/parameters`, () => {
    return HttpResponse.json({
      parameters: [
        { name: 'hk_layer1', package: 'NPF', initial: 10.0, min: 0.1, max: 100.0, transform: 'log' },
        { name: 'hk_layer2', package: 'NPF', initial: 5.0, min: 0.1, max: 100.0, transform: 'log' },
      ],
    });
  }),

  http.get(`${API_BASE}/projects/:projectId/pest/config`, () => {
    return HttpResponse.json({ config: null });
  }),

  http.post(`${API_BASE}/projects/:projectId/pest/config`, async ({ request }) => {
    const config = await request.json();
    return HttpResponse.json({ success: true, config });
  }),

  http.post(`${API_BASE}/projects/:projectId/pest/run`, async () => {
    return HttpResponse.json({
      run_id: crypto.randomUUID(),
      task_id: crypto.randomUUID(),
      status: 'queued',
      message: 'PEST++ run queued for execution',
    });
  }),

  http.get(`${API_BASE}/projects/:projectId/pest/runs`, () => {
    return HttpResponse.json([]);
  }),

  http.get(`${API_BASE}/projects/:projectId/pest/runs/:runId/results`, () => {
    return HttpResponse.json({ results: null });
  }),

  // Observations endpoints
  http.get(`${API_BASE}/projects/:projectId/observations/sets`, () => {
    return HttpResponse.json([]);
  }),

  http.get(`${API_BASE}/projects/:projectId/observations`, () => {
    return HttpResponse.json({
      wells: [],
      observations: [],
      n_observations: 0,
    });
  }),

  // Files endpoints
  http.get(`${API_BASE}/projects/:projectId/files/categorized`, () => {
    return HttpResponse.json({
      model_core: [{ path: 'mfsim.nam', name: 'mfsim.nam', extension: '.nam' }],
      model_input: [],
      model_output: [],
      pest: [],
      observation: [],
      blocked: [],
      other: [],
    });
  }),

  http.get(`${API_BASE}/projects/:projectId/files`, () => {
    return HttpResponse.json({ files: ['mfsim.nam', 'model.nam', 'model.dis'] });
  }),
];

// ─── Error Simulation Handlers ──────────────────────────────────────────────

/**
 * Override handlers for simulating error conditions in tests.
 * Use with server.use() in individual tests.
 */
export const errorHandlers = {
  health503: http.get(`${API_BASE}/health`, () => {
    return HttpResponse.json(
      { status: 'unhealthy', database: 'unhealthy', redis: 'unhealthy', minio: 'unhealthy' },
      { status: 503 }
    );
  }),

  projectsNetworkError: http.get(`${API_BASE}/projects`, () => {
    return HttpResponse.error();
  }),

  projectNotFound: http.get(`${API_BASE}/projects/:projectId`, () => {
    return HttpResponse.json(
      { detail: 'Project not found' },
      { status: 404 }
    );
  }),

  simulationConflict: http.post(`${API_BASE}/projects/:projectId/simulation/run`, () => {
    return HttpResponse.json(
      { detail: 'Simulation already running' },
      { status: 409 }
    );
  }),

  uploadValidationError: http.post(`${API_BASE}/projects/:projectId/upload`, () => {
    return HttpResponse.json(
      {
        is_valid: false,
        model_type: null,
        errors: ['No MODFLOW name file found', 'Invalid grid dimensions'],
        warnings: [],
      },
      { status: 400 }
    );
  }),
};
