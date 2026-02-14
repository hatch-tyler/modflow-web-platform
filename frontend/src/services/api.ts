import axios from 'axios'
import type {
  Project, ProjectCreate, ProjectUpdate, Run, HealthCheck, ValidationReport,
  ResultsSummary, HeadSlice, BudgetData, TimeseriesData, GridGeometry,
  ObservationData, ObservationSet, ObservationSetUpdate, MarkFileAsObservation,
  CategorizedFiles, TimestepIndex, HeadStatistics,
  LiveTimestepIndex, LiveHeadSlice, LiveResultsSummary,
  PestConfig, PestResults, UploadStatus, AgentStatus,
  StructuredGridInfo,
  ZoneDefinitionSummary, ZoneDefinitionDetail, ZoneBudgetComputeResponse, ZoneBudgetProgress,
  ConvergenceDetail, StressSummary, Refinement, BackupInfo,
  ParameterScanResponse, ParameterScanProgress,
} from '../types'

const API_BASE = '/api/v1'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Health
export const healthApi = {
  check: () => api.get<HealthCheck>('/health').then(r => r.data),
}

// Projects
export const projectsApi = {
  list: (skip = 0, limit = 100) =>
    api.get<Project[]>('/projects', { params: { skip, limit } }).then(r => r.data),

  get: (id: string) =>
    api.get<Project>(`/projects/${id}`).then(r => r.data),

  create: (data: ProjectCreate) =>
    api.post<Project>('/projects', data).then(r => r.data),

  update: (id: string, data: ProjectUpdate) =>
    api.patch<Project>(`/projects/${id}`, data).then(r => r.data),

  delete: (id: string) =>
    api.delete(`/projects/${id}`),

  uploadModel: (id: string, file: File, onProgress?: (progress: number) => void) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post<ValidationReport>(`/projects/${id}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      // Large models can take 15+ minutes for upload + validation + caching
      timeout: 30 * 60 * 1000, // 30 minutes
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded * 100) / e.total))
        }
      },
    }).then(r => r.data)
  },

  getValidation: (id: string) =>
    api.get<ValidationReport>(`/projects/${id}/validation`).then(r => r.data),

  revalidate: (id: string) =>
    api.post<ValidationReport>(`/projects/${id}/revalidate`).then(r => r.data),

  getUploadStatus: (projectId: string) =>
    api.get<UploadStatus | null>(`/projects/${projectId}/upload/status`).then(r => r.data),
}

// Upload status
export const uploadApi = {
  getStatus: (jobId: string) =>
    api.get<UploadStatus>(`/upload/status/${jobId}`).then(r => r.data),

  getProjectStatus: (projectId: string) =>
    api.get<UploadStatus | null>(`/projects/${projectId}/upload/status`).then(r => r.data),
}

// Simulation
export const simulationApi = {
  start: (projectId: string, options?: { name?: string; save_budget?: boolean }) =>
    api.post<{ run_id: string; task_id: string; status: string; message: string }>(
      `/projects/${projectId}/simulation/run`,
      options || {}
    ).then(r => r.data),

  listRuns: (projectId: string, limit = 10) =>
    api.get<Run[]>(`/projects/${projectId}/simulation/runs`, { params: { limit } }).then(r => r.data),

  getRun: (projectId: string, runId: string) =>
    api.get<Run>(`/projects/${projectId}/simulation/runs/${runId}`).then(r => r.data),

  cancel: (projectId: string, runId: string) =>
    api.post<{ run_id: string; status: string; message: string }>(
      `/projects/${projectId}/simulation/runs/${runId}/cancel`
    ).then(r => r.data),

  getStatus: (projectId: string) =>
    api.get<{ has_runs: boolean; latest_run: Run | null }>(`/projects/${projectId}/simulation/status`).then(r => r.data),
}

// SSE for live console output
export function createOutputEventSource(projectId: string, runId: string): EventSource {
  return new EventSource(`${API_BASE}/projects/${projectId}/simulation/runs/${runId}/output`)
}

// Results
export const resultsApi = {
  getSummary: (projectId: string, runId: string) =>
    api.get<ResultsSummary>(`/projects/${projectId}/runs/${runId}/results/summary`).then(r => r.data),

  // On-demand head slice - fetches from cache or extracts on-demand from HDS file
  getHeads: (projectId: string, runId: string, layer: number, kper: number, kstp: number) =>
    api.get<HeadSlice>(`/projects/${projectId}/runs/${runId}/results/heads`, {
      params: { layer, kper, kstp },
    }).then(r => r.data),

  // Get available timesteps (cached index of HDS file contents)
  getAvailableTimesteps: (projectId: string, runId: string) =>
    api.get<TimestepIndex>(`/projects/${projectId}/runs/${runId}/results/heads/available`).then(r => r.data),

  // Get head statistics (min/max for color scaling)
  getHeadStatistics: (projectId: string, runId: string) =>
    api.get<HeadStatistics>(`/projects/${projectId}/runs/${runId}/results/heads/statistics`).then(r => r.data),

  getBudget: (projectId: string, runId: string) =>
    api.get<BudgetData>(`/projects/${projectId}/runs/${runId}/results/budget`).then(r => r.data),

  getTimeseries: (projectId: string, runId: string, params: { row?: number; col?: number; layer: number; node?: number }) =>
    api.get<TimeseriesData>(`/projects/${projectId}/runs/${runId}/results/timeseries`, {
      params,
    }).then(r => r.data),

  getGridGeometry: (projectId: string, runId: string) =>
    api.get<GridGeometry>(`/projects/${projectId}/runs/${runId}/results/grid-geometry`)
      .then(r => r.data),

  getStructuredGridInfo: (projectId: string, runId: string) =>
    api.get<StructuredGridInfo>(`/projects/${projectId}/runs/${runId}/results/grid-info`)
      .then(r => r.data),

  // Live results (available during simulation)
  getLiveSummary: (projectId: string, runId: string) =>
    api.get<LiveResultsSummary>(`/projects/${projectId}/runs/${runId}/results/live/summary`).then(r => r.data),

  getLiveAvailableTimesteps: (projectId: string, runId: string) =>
    api.get<LiveTimestepIndex>(`/projects/${projectId}/runs/${runId}/results/live/available`).then(r => r.data),

  getLiveHeads: (projectId: string, runId: string, layer: number, kper: number, kstp: number) =>
    api.get<LiveHeadSlice>(`/projects/${projectId}/runs/${runId}/results/live/heads`, {
      params: { layer, kper, kstp },
    }).then(r => r.data),

  getModelZoneBudget: (projectId: string, runId: string, refresh?: boolean) =>
    api.get(`/projects/${projectId}/runs/${runId}/results/zone-budget/model`, {
      params: refresh ? { refresh: true } : undefined,
    }).then(r => r.data),

  computeZoneBudget: (projectId: string, runId: string, zoneLayers: Record<string, Record<string, number[]>>) =>
    api.post(`/projects/${projectId}/runs/${runId}/results/zone-budget`, { zone_layers: zoneLayers })
      .then(r => r.data),

  exportHeadsCsvUrl: (projectId: string, runId: string, layer: number, kper: number, kstp: number) =>
    `${API_BASE}/projects/${projectId}/runs/${runId}/results/export/heads?layer=${layer}&kper=${kper}&kstp=${kstp}`,

  exportDrawdownCsvUrl: (projectId: string, runId: string, layer: number, kper: number, kstp: number) =>
    `${API_BASE}/projects/${projectId}/runs/${runId}/results/export/drawdown?layer=${layer}&kper=${kper}&kstp=${kstp}`,

  exportBudgetCsvUrl: (projectId: string, runId: string) =>
    `${API_BASE}/projects/${projectId}/runs/${runId}/results/export/budget`,

  exportTimeseriesCsvUrl: (projectId: string, runId: string, params: { layer: number; row?: number; col?: number; node?: number }) => {
    const qs = new URLSearchParams({ layer: String(params.layer) })
    if (params.row !== undefined) qs.set('row', String(params.row))
    if (params.col !== undefined) qs.set('col', String(params.col))
    if (params.node !== undefined) qs.set('node', String(params.node))
    return `${API_BASE}/projects/${projectId}/runs/${runId}/results/export/timeseries?${qs.toString()}`
  },
}

// Files API (categorization and management)
export const filesApi = {
  getCategorized: (projectId: string) =>
    api.get<CategorizedFiles>(`/projects/${projectId}/files/categorized`).then(r => r.data),

  deleteFile: (projectId: string, path: string) =>
    api.delete(`/projects/${projectId}/files/${encodeURIComponent(path)}`).then(r => r.data),

  listFiles: (projectId: string) =>
    api.get<{ files: string[] }>(`/projects/${projectId}/files`).then(r => r.data),

  previewCsv: (projectId: string, path: string, rows = 10) =>
    api.get<{ headers: string[]; rows: string[][]; total_rows: number; file_path: string }>(
      `/projects/${projectId}/files/${encodeURIComponent(path)}/preview`,
      { params: { rows } }
    ).then(r => r.data),
}

// Observations API (multi-set support)
export const observationsApi = {
  // Multi-set endpoints
  listSets: (projectId: string) =>
    api.get<ObservationSet[]>(`/projects/${projectId}/observations/sets`).then(r => r.data),

  createSet: (projectId: string, file: File, name?: string) => {
    const formData = new FormData()
    formData.append('file', file)
    if (name) {
      formData.append('name', name)
    }
    return api.post<ObservationSet>(`/projects/${projectId}/observations/sets`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data)
  },

  getSet: (projectId: string, setId: string) =>
    api.get<ObservationData>(`/projects/${projectId}/observations/sets/${setId}`).then(r => r.data),

  updateSet: (projectId: string, setId: string, data: ObservationSetUpdate) =>
    api.patch<ObservationSet>(`/projects/${projectId}/observations/sets/${setId}`, data).then(r => r.data),

  deleteSet: (projectId: string, setId: string) =>
    api.delete(`/projects/${projectId}/observations/sets/${setId}`).then(r => r.data),

  markFileAsObs: (projectId: string, data: MarkFileAsObservation) =>
    api.post<ObservationSet>(`/projects/${projectId}/observations/from-file`, data).then(r => r.data),

  // Legacy endpoints (backwards compatibility)
  upload: (projectId: string, file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/projects/${projectId}/observations/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data)
  },

  get: (projectId: string) =>
    api.get<ObservationData>(`/projects/${projectId}/observations`).then(r => r.data),

  delete: (projectId: string) =>
    api.delete(`/projects/${projectId}/observations`).then(r => r.data),
}

// PEST++
export const pestApi = {
  // Fast read-only: returns cached params or scan status (never loads model)
  getParameters: (projectId: string) =>
    api.get<ParameterScanResponse>(`/projects/${projectId}/pest/parameters`).then(r => r.data),

  // Trigger parameter discovery on Celery worker
  startParameterScan: (projectId: string, forceRefresh = false) =>
    api.post<{ status: string; task_id: string }>(
      `/projects/${projectId}/pest/parameters/scan`,
      { force_refresh: forceRefresh },
    ).then(r => r.data),

  // Poll scan progress
  getParameterScanStatus: (projectId: string, taskId: string) =>
    api.get<ParameterScanProgress>(
      `/projects/${projectId}/pest/parameters/scan/${taskId}`,
    ).then(r => r.data),

  clearParameterCache: (projectId: string) =>
    api.delete(`/projects/${projectId}/pest/parameters/cache`).then(r => r.data),

  getConfig: (projectId: string) =>
    api.get<{ config: PestConfig | null }>(`/projects/${projectId}/pest/config`).then(r => r.data),

  saveConfig: (projectId: string, config: PestConfig) =>
    api.post(`/projects/${projectId}/pest/config`, config).then(r => r.data),

  startRun: (projectId: string, method: string, name?: string, config?: PestConfig) =>
    api.post<{ run_id: string; task_id: string; status: string; message: string }>(
      `/projects/${projectId}/pest/run`,
      { method, name, config }
    ).then(r => r.data),

  listRuns: (projectId: string, limit = 20) =>
    api.get<Run[]>(`/projects/${projectId}/pest/runs`, { params: { limit } }).then(r => r.data),

  getResults: (projectId: string, runId: string) =>
    api.get<{ results: PestResults | null }>(`/projects/${projectId}/pest/runs/${runId}/results`).then(r => r.data),

  exportPhiCsvUrl: (projectId: string, runId: string) =>
    `${API_BASE}/projects/${projectId}/pest/runs/${runId}/export/phi`,

  exportParametersCsvUrl: (projectId: string, runId: string) =>
    `${API_BASE}/projects/${projectId}/pest/runs/${runId}/export/parameters`,

  exportResidualsCsvUrl: (projectId: string, runId: string) =>
    `${API_BASE}/projects/${projectId}/pest/runs/${runId}/export/residuals`,

  // Network mode / agent status
  getAgentStatus: (projectId: string, runId: string) =>
    api.get<AgentStatus>(`/projects/${projectId}/pest/runs/${runId}/agents`).then(r => r.data),

  getNetworkInfo: (projectId: string) =>
    api.get<{
      manager_port: number
      minio_port: number
      network_mode_enabled: boolean
      instructions: Record<string, string>
    }>(`/projects/${projectId}/pest/network-info`).then(r => r.data),

  getCurrentRunInfo: (projectId: string) =>
    api.get<{
      has_active_run: boolean
      run_id?: string
      workspace_prefix?: string
      pst_file?: string
      manager_port?: number
      message?: string
    }>(`/projects/${projectId}/pest/current-run`).then(r => r.data),
}

export function createPestOutputEventSource(projectId: string, runId: string): EventSource {
  return new EventSource(`${API_BASE}/projects/${projectId}/pest/runs/${runId}/output`)
}

// Visualization (placeholder endpoints)
export const visualizationApi = {
  getGrid: (projectId: string) =>
    api.get(`/projects/${projectId}/grid`, { responseType: 'arraybuffer' }).then(r => r.data),

  getArray: (projectId: string, arrayName: string) =>
    api.get(`/projects/${projectId}/arrays/${arrayName}`, { responseType: 'arraybuffer' }).then(r => r.data),

  getBoundaries: (projectId: string) =>
    api.get(`/projects/${projectId}/boundaries`).then(r => r.data),
}


// Zone Definitions (per project, persisted in MinIO)
export const zoneDefinitionsApi = {
  list: (projectId: string) =>
    api.get<ZoneDefinitionSummary[]>(`/projects/${projectId}/zone-definitions`).then(r => r.data),

  get: (projectId: string, name: string) =>
    api.get<ZoneDefinitionDetail>(`/projects/${projectId}/zone-definitions/${encodeURIComponent(name)}`).then(r => r.data),

  save: (projectId: string, defn: { name: string; zone_layers: Record<string, Record<string, number[]>>; num_zones: number }) =>
    api.post<{ name: string; saved: boolean }>(`/projects/${projectId}/zone-definitions`, defn).then(r => r.data),

  delete: (projectId: string, name: string) =>
    api.delete<{ name: string; deleted: boolean }>(`/projects/${projectId}/zone-definitions/${encodeURIComponent(name)}`).then(r => r.data),
}

// Zone Budget (async compute via Celery)
export const zoneBudgetApi = {
  compute: (projectId: string, runId: string, zoneLayers: Record<string, Record<string, number[]>>, quickMode = false) =>
    api.post<ZoneBudgetComputeResponse>(
      `/projects/${projectId}/runs/${runId}/results/zone-budget/compute`,
      { zone_layers: zoneLayers, quick_mode: quickMode },
    ).then(r => r.data),

  getStatus: (projectId: string, runId: string, taskId: string) =>
    api.get<ZoneBudgetProgress>(
      `/projects/${projectId}/runs/${runId}/results/zone-budget/status/${taskId}`,
    ).then(r => r.data),

  getResult: (projectId: string, runId: string, taskId: string) =>
    api.get(`/projects/${projectId}/runs/${runId}/results/zone-budget/result/${taskId}`).then(r => r.data),
}

// Convergence Analysis
export const convergenceApi = {
  getDetail: (projectId: string, runId: string) =>
    api.get<ConvergenceDetail>(`/projects/${projectId}/runs/${runId}/results/convergence/detail`).then(r => r.data),

  getStressData: (projectId: string, runId: string) =>
    api.get<StressSummary>(`/projects/${projectId}/runs/${runId}/results/convergence/stress-data`).then(r => r.data),

  getRecommendations: (projectId: string, runId: string) =>
    api.get<{ recommendations: Refinement[] }>(`/projects/${projectId}/runs/${runId}/results/convergence/recommendations`).then(r => r.data),

  applyRefinements: (projectId: string, runId: string, refinementIds: string[]) =>
    api.post<{ backup_timestamp: string; modified_files: { file: string; refinements: string[]; backup: string }[] }>(
      `/projects/${projectId}/runs/${runId}/results/convergence/apply-refinements`,
      { refinement_ids: refinementIds },
    ).then(r => r.data),

  revertRefinements: (projectId: string, runId: string, backupTimestamp: string) =>
    api.post<{ restored_files: string[]; backup_timestamp: string }>(
      `/projects/${projectId}/runs/${runId}/results/convergence/revert-refinements`,
      { backup_timestamp: backupTimestamp },
    ).then(r => r.data),
}

// File Editor
export const fileEditorApi = {
  getContent: (projectId: string, filePath: string) =>
    api.get<{ content: string; size: number; encoding: string }>(
      `/projects/${projectId}/files/${encodeURIComponent(filePath)}/content`,
    ).then(r => r.data),

  saveContent: (projectId: string, filePath: string, content: string, createBackup = true) =>
    api.put<{ saved: boolean; size: number; backup_timestamp: string | null }>(
      `/projects/${projectId}/files/${encodeURIComponent(filePath)}/content`,
      { content, create_backup: createBackup },
    ).then(r => r.data),

  getBackups: (projectId: string, filePath: string) =>
    api.get<{ backups: BackupInfo[] }>(
      `/projects/${projectId}/files/${encodeURIComponent(filePath)}/backups`,
    ).then(r => r.data),

  revert: (projectId: string, filePath: string, backupTimestamp: string) =>
    api.post<{ reverted: boolean; content: string; size: number }>(
      `/projects/${projectId}/files/${encodeURIComponent(filePath)}/revert`,
      { backup_timestamp: backupTimestamp },
    ).then(r => r.data),
}

// MODFLOW Docs / Definitions
export const modflowDocsApi = {
  listPackages: (modelType: string) =>
    api.get<{ model_type: string; packages: { name: string; description: string; file_extensions: string[] }[] }>(
      `/modflow/definitions/${modelType}`,
    ).then(r => r.data),

  getDefinition: (modelType: string, packageName: string) =>
    api.get(`/modflow/definitions/${modelType}/${packageName.toLowerCase()}`).then(r => r.data),
}

export default api
