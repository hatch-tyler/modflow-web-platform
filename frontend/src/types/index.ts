// API Types - matching backend schemas

export type ModelType = 'mf2005' | 'mfnwt' | 'mfusg' | 'mf6' | 'unknown'
export type RunStatus = 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
export type RunType = 'forward' | 'pest_glm' | 'pest_ies'

export interface StressPeriodData {
  perlen: number
  nstp: number
  tsmult: number
}

export interface Project {
  id: string
  name: string
  description?: string
  model_type?: ModelType
  nlay?: number
  nrow?: number
  ncol?: number
  nper?: number
  grid_type?: string
  xoff?: number
  yoff?: number
  angrot?: number
  epsg?: number
  length_unit?: string
  stress_period_data?: StressPeriodData[]
  start_date?: string
  delr?: number[]
  delc?: number[]
  time_unit?: string
  storage_path?: string
  is_valid: boolean
  validation_errors?: Record<string, unknown>
  packages?: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface ProjectCreate {
  name: string
  description?: string
}

export interface ProjectUpdate {
  name?: string
  description?: string
  xoff?: number
  yoff?: number
  angrot?: number
  epsg?: number
  start_date?: string | null
}

export interface Run {
  id: string
  project_id: string
  name?: string
  run_type: RunType
  status: RunStatus
  started_at?: string
  completed_at?: string
  celery_task_id?: string
  config?: Record<string, unknown>
  exit_code?: number
  error_message?: string
  results_path?: string
  convergence_info?: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface RunCreate {
  name?: string
  run_type: RunType
  config?: Record<string, unknown>
}

export interface ValidationReport {
  is_valid: boolean
  model_type?: ModelType
  grid_info?: {
    nlay: number
    nrow: number
    ncol: number
    nper: number
    grid_type?: string
  }
  packages_found: string[]
  packages_missing: string[]
  errors: string[]
  warnings: string[]
}

export interface HealthCheck {
  status: string
  version: string
  database: string
  redis: string
  minio: string
}

// Results types
export interface ResultsSummary {
  heads_summary: {
    nstp_total: number
    kstpkper_list: [number, number][]
    times: number[]
    max_head: number | null
    min_head: number | null
  }
  budget: BudgetData
  convergence: ConvergenceData
  metadata: {
    model_type: string
    grid_type: string
    nlay: number
    nrow: number
    ncol: number
    nper: number
    xoff?: number
    yoff?: number
    angrot?: number
    epsg?: number
    stress_period_data?: StressPeriodData[]
    start_date?: string | null
    time_unit?: string
    length_unit?: string
    delr?: number[]
    delc?: number[]
  }
}

export interface HeadSlice {
  layer: number
  kper: number
  kstp: number
  shape: [number, number]
  data: (number | null)[][]
}

// On-demand head slice types
export interface TimestepIndex {
  kstpkper_list: [number, number][]  // [kstp, kper] pairs
  times: number[]
  nlay: number
  grid_shape: number[]  // [nlay, nrow, ncol] or [nlay, ncpl]
  is_unstructured: boolean
  hds_path?: string
  error?: string
}

export interface HeadStatistics {
  min_head: number | null
  max_head: number | null
  timesteps_sampled: number
  total_timesteps: number
  error?: string
}

// Live results types (during simulation)
export interface LiveTimestepIndex extends TimestepIndex {
  simulation_status: RunStatus
  live: boolean
  message?: string
}

export interface LiveHeadSlice extends HeadSlice {
  simulation_status: RunStatus
  live: boolean
}

export interface LiveResultsSummary extends ResultsSummary {
  simulation_status: RunStatus
  live: boolean
  message?: string
}

export interface BudgetData {
  record_names: string[]
  periods: Record<string, BudgetPeriod>
  quick_mode?: boolean
  total_timesteps?: number
  timesteps_processed?: number
}

export interface BudgetPeriod {
  kstp: number
  kper: number
  in: Record<string, number>
  out: Record<string, number>
  total_in: number
  total_out: number
  discrepancy: number
  percent_discrepancy: number
}

export interface ConvergenceData {
  converged: boolean
  mass_balance_error_pct: number | null
  max_head_changes: number[]
  percent_discrepancies?: number[]
  warnings?: string[]
}

export interface TimeseriesData {
  layer: number
  row?: number
  col?: number
  node?: number
  times: number[]
  heads: (number | null)[]
}

export interface GridGeometry {
  grid_type: string
  extent: [number, number, number, number]
  nlay: number
  ncpl: number
  layers: Record<string, {
    polygons: number[][][]
  }>
}

export interface ObservationWell {
  layer: number
  row?: number
  col?: number
  node?: number
  times: number[]
  heads: number[]
}

export interface ObservationData {
  wells: Record<string, ObservationWell>
  n_observations: number
}

// Observation Set types (multi-set support)
export type ObservationSetSource = 'upload' | 'zip_detected' | 'manual_marked'
export type ObservationFormat = 'long' | 'wide'

export interface ColumnMapping {
  well_name: string
  layer: string | number
  row?: string | number | null
  col?: string | number | null
  node?: string | number | null
  time: string
  value: string
}

export interface ObservationSet {
  id: string
  name: string
  source: ObservationSetSource
  format: ObservationFormat
  column_mapping?: ColumnMapping | null
  wells: string[]
  n_observations: number
  created_at: string
  file_path?: string | null  // Path to source CSV (for detected observations)
}

export interface ObservationSetCreate {
  name: string
  column_mapping?: ColumnMapping
}

export interface ObservationSetUpdate {
  name?: string
  column_mapping?: ColumnMapping
}

export interface MarkFileAsObservation {
  file_path: string
  name: string
  column_mapping: ColumnMapping
}

export interface ObservationSetSelection {
  set_id: string
  weight_multiplier: number
  per_well_weights?: Record<string, number>
}

// File categorization types
export type FileCategory = 'model_core' | 'model_input' | 'model_output' | 'pest' | 'observation' | 'blocked' | 'other'

export interface FileInfo {
  path: string
  name: string
  extension: string
  description: string
  size: number
}

export interface CategorizedFiles {
  categories: Record<FileCategory, FileInfo[]>
  blocked_rejected: string[]
  total_files: number
  total_size_mb: number
  detected_observations: ObservationSet[]
}

// PEST types
export type ParameterApproach = 'multiplier' | 'pilotpoints'
export type PackageType = 'array' | 'list'

export interface VariogramConfig {
  type: 'exponential' | 'spherical' | 'gaussian'
  correlation_length: number
  anisotropy: number
  bearing: number
}

export interface PestParameter {
  property: string
  property_name: string
  short_name: string
  layer: number | null  // null for list-based parameters
  package_type: PackageType
  count?: number  // For list packages (e.g., number of HFB barriers)
  stats: {
    mean: number | null
    min: number | null
    max: number | null
  }
  suggested_transform: string
  suggested_lower: number
  suggested_upper: number
}

export interface PestParameterConfig {
  property: string
  layer: number | null
  approach: ParameterApproach
  pp_space?: number  // Pilot point spacing
  variogram?: VariogramConfig
  initial_value: number
  lower_bound: number
  upper_bound: number
  transform: string
  group: string
  package_type?: PackageType
}

export interface PestSettings {
  noptmax: number
  phiredstp: number
  nphinored: number
  maxsing?: number
  eigthresh: number
  // IES-specific settings
  ies_num_reals?: number
  ies_initial_lambda?: number
  ies_bad_phi_sigma?: number
  ies_subset_size?: number
  // Parallel execution settings
  num_workers?: number
  // Network mode - for distributed agents across machines
  network_mode?: boolean
  // Hybrid agent configuration (Phase 2)
  local_containers?: boolean  // Run containerized agents on this machine
  local_agents?: number       // Number of local container agents
  remote_agents?: number      // Expected remote agents from other machines
}

// Agent status for network mode runs
export interface AgentStatus {
  network_mode: boolean
  expected: number
  connected: number
  status: 'waiting' | 'running' | 'completed' | 'failed' | 'not_network_mode'
  run_status?: string
  message?: string
  // Hybrid agent breakdown (Phase 2)
  local_containers?: boolean
  local_agents_expected?: number
  local_agents_connected?: number
  remote_agents_expected?: number
  remote_agents_connected?: number
}

export interface PestConfig {
  parameters: PestParameterConfig[]
  observation_weights: Record<string, number>
  settings: PestSettings
}

export interface PestParamSummary {
  mean: number
  std: number
  min: number
  max: number
  p5: number
  p25: number
  p50: number
  p75: number
  p95: number
  values: number[]
}

export interface PestEnsembleData {
  phi_per_real: Record<string, { iteration: number; phi: number }[]>
  par_summary: Record<string, PestParamSummary>
  prior_summary?: Record<string, PestParamSummary>
  obs_summary: Record<string, { mean: number; std: number; p5: number; p95: number }>
  n_reals: number
  n_failed: number
}

export interface PestResults {
  phi_history: { iteration: number; phi: number; phi_min?: number; phi_max?: number; phi_std?: number }[]
  parameters: Record<string, number>
  residuals: { name: string; observed: number; simulated: number; residual: number }[]
  converged: boolean
  ensemble?: PestEnsembleData
}

// Post-processing progress types
export type PostProcessStage = 'heads_budget' | 'heads' | 'budget' | 'listing' | 'geometry' | 'uploading' | 'finalizing'

export interface PostProcessProgress {
  postprocess_status: 'running' | 'completed' | 'failed'
  postprocess_progress: number
  postprocess_message?: string
  postprocess_stage?: PostProcessStage
  postprocess_completed?: PostProcessStage[]
  budget_warning?: string
}

// Upload status types
export type UploadStage = 'receiving' | 'extracting' | 'validating' | 'storing' | 'caching' | 'complete' | 'failed'

export interface UploadStatus {
  job_id: string
  project_id: string
  stage: UploadStage
  progress: number
  message: string
  file_count?: number
  files_processed?: number
  is_valid?: boolean
  error?: string
}

export interface StructuredGridInfo {
  delr: number[] | null
  delc: number[] | null
  nrow: number
  ncol: number
  nlay: number
  xoff: number
  yoff: number
  angrot: number
  epsg: number | null
  length_unit: string | null
}

// Convergence analysis types
export interface TimestepConvergence {
  kper: number
  kstp: number
  outer_iterations: number
  converged: boolean
  max_dvmax: number
  max_dvmax_cell: string
  max_rclose: number
  max_rclose_cell: string
  backtracking_events: number
}

export interface StressPeriodSummary {
  kper: number
  total_outer_iters: number
  max_iterations: number
  failed_timesteps: number
  avg_iterations: number
  percent_discrepancy: number | null
  difficulty: 'low' | 'moderate' | 'high' | 'failed'
}

export interface ProblemCell {
  cell_id: string
  occurrences: number
  type: string
  affected_sps: number[]
}

export interface SolverSettings {
  solver_type: string
  complexity?: string
  outer_dvclose?: number
  outer_maximum?: number
  inner_rclose?: number
  inner_dvclose?: number
  under_relaxation?: string
  backtracking_number?: number
  linear_acceleration?: string
  hclose?: number
  rclose?: number
  inner_maximum?: number
}

export interface ConvergenceDetail {
  model_type: string
  total_timesteps: number
  failed_timesteps: number
  timesteps: TimestepConvergence[]
  stress_period_summary: StressPeriodSummary[]
  problem_cells: ProblemCell[]
  solver_settings: SolverSettings
}

export interface StressPeriodPackageData {
  total_rate: number
  mean_rate?: number
  n_active: number
}

export interface StressPeriod {
  kper: number
  perlen?: number
  nstp?: number
  tsmult?: number
  [packageName: string]: StressPeriodPackageData | number | undefined
}

export interface StressSummary {
  packages: string[]
  periods: StressPeriod[]
}

export interface RefinementFileModification {
  package: string
  block: string | null
  variable: string
  old_value: string
  new_value: string
  stress_period?: number
}

export interface Refinement {
  id: string
  category: 'solver' | 'temporal' | 'package'
  priority: 'high' | 'medium' | 'low'
  title: string
  description: string
  current_value: string
  suggested_value: string
  file_modification: RefinementFileModification | null
}

export interface BackupInfo {
  timestamp: string
  size: number
  path: string
}

// Zone budget types
export interface ZoneDefinitionSummary {
  name: string
  num_zones: number
  zone_count: number
}

export interface ZoneDefinitionDetail {
  name: string
  zone_layers: Record<string, Record<string, number[]>>
  num_zones: number
}

export interface ZoneBudgetComputeResponse {
  status: 'completed' | 'queued'
  result?: Record<string, unknown>
  cached?: boolean
  task_id?: string
}

export interface ZoneBudgetProgress {
  status: 'queued' | 'downloading' | 'computing' | 'completed' | 'failed'
  progress: number
  message: string
  error?: string | null
}

// Parameter scan types (async discovery via Celery worker)
export type ParameterScanStatus = 'not_started' | 'queued' | 'downloading' | 'loading' | 'caching' | 'scanning' | 'completed' | 'failed'

export interface ParameterScanResponse {
  parameters: PestParameter[]
  status: ParameterScanStatus
  task_id?: string
  progress?: number
  message?: string
}

export interface ParameterScanProgress {
  task_id: string
  status: ParameterScanStatus
  progress: number
  message: string
  error?: string | null
  parameters?: PestParameter[]
}
