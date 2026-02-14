import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Settings,
  Play,
  Sliders,
  BarChart,
  Loader2,
  Square,
  CheckSquare,
  AlertCircle,
  Upload,
  FileSpreadsheet,
  RefreshCw,
} from 'lucide-react'
import Plot from 'react-plotly.js'
import clsx from 'clsx'
import { useVirtualizer, VirtualItem } from '@tanstack/react-virtual'
import { projectsApi, pestApi, observationsApi } from '../services/api'
import { useRunManager } from '../store/runManager'
import ExportButton from '../components/dashboard/ExportButton'
import type {
  PestParameter,
  PestParameterConfig,
  PestConfig,
  PestSettings,
  PestResults,
  PestEnsembleData,
  ObservationSet,
  ParameterApproach,
} from '../types'

type TabId = 'parameters' | 'observations' | 'settings' | 'results'
type PestMethod = 'glm' | 'ies'

const DEFAULT_SETTINGS: PestSettings = {
  noptmax: 20,
  phiredstp: 0.005,
  nphinored: 4,
  eigthresh: 1e-6,
  ies_num_reals: 50,
  ies_initial_lambda: 100.0,
  ies_bad_phi_sigma: 2.0,
  num_workers: 4,
  network_mode: false,
  // Hybrid agent configuration (Phase 2)
  local_containers: false,
  local_agents: 0,
  remote_agents: 0,
}

export default function PestPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<TabId>('parameters')
  const [method, setMethod] = useState<PestMethod>('glm')

  // Parameter configuration state
  const [selectedParams, setSelectedParams] = useState<Record<string, PestParameterConfig>>({})
  const [obsWeights, setObsWeights] = useState<Record<string, number>>({})
  const [pestSettings, setPestSettings] = useState<PestSettings>(DEFAULT_SETTINGS)

  // Run state - integrated with global run manager
  const [currentRunId, setCurrentRunId] = useState<string | null>(null)
  const [localOutput, setLocalOutput] = useState<string[]>([])
  const outputRef = useRef<HTMLDivElement | null>(null)

  // Global run manager for persistent run tracking
  const {
    activeRuns,
    startRun: globalStartRun,
    selectRun,
    getActiveRunsForProject,
  } = useRunManager()

  // Get active runs for this project and current run state
  const projectActiveRuns = projectId ? getActiveRunsForProject(projectId) : []
  const activeRun = currentRunId ? activeRuns[currentRunId] : null
  const isRunning = activeRun?.status === 'running'
  const output = activeRun ? activeRun.output : localOutput

  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  })

  const [rescanning, setRescanning] = useState(false)

  const { data: paramsData, isLoading: paramsLoading, isFetching: paramsFetching } = useQuery({
    queryKey: ['pest-parameters', projectId],
    queryFn: () => pestApi.getParameters(projectId!),
    enabled: !!projectId && !!project?.is_valid,
    staleTime: 10 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    retry: 1,
    retryDelay: 5000,
  })

  // Reset rescanning flag when fetch completes
  useEffect(() => {
    if (rescanning && !paramsFetching) {
      setRescanning(false)
    }
  }, [rescanning, paramsFetching])

  const handleRescanParameters = useCallback(async () => {
    if (!projectId) return
    setRescanning(true)
    try {
      await pestApi.clearParameterCache(projectId)
    } catch {
      // Cache may already be gone
    }
    queryClient.invalidateQueries({ queryKey: ['pest-parameters', projectId] })
  }, [projectId, queryClient])

  const { data: obsData } = useQuery({
    queryKey: ['observations', projectId],
    queryFn: () => observationsApi.get(projectId!),
    enabled: !!projectId,
    retry: false,
  })

  // Query for observation sets (multi-set support)
  const { data: observationSets = [] } = useQuery({
    queryKey: ['observation-sets', projectId],
    queryFn: () => observationsApi.listSets(projectId!),
    enabled: !!projectId,
    retry: false,
  })

  // Selected observation sets for calibration
  const [selectedSetIds, setSelectedSetIds] = useState<Set<string>>(new Set())
  const [setWeightMultipliers, setSetWeightMultipliers] = useState<Record<string, number>>({})

  // Auto-select all observation sets initially (only once when data loads)
  const [setsInitialized, setSetsInitialized] = useState(false)
  useEffect(() => {
    if (observationSets.length > 0 && !setsInitialized) {
      setSelectedSetIds(new Set(observationSets.map(s => s.id)))
      setSetsInitialized(true)
    }
  }, [observationSets, setsInitialized])

  const { data: pestRuns } = useQuery({
    queryKey: ['pest-runs', projectId],
    queryFn: () => pestApi.listRuns(projectId!),
    enabled: !!projectId,
    refetchInterval: isRunning ? 3000 : false,
  })

  const { data: savedConfig } = useQuery({
    queryKey: ['pest-config', projectId],
    queryFn: () => pestApi.getConfig(projectId!),
    enabled: !!projectId,
  })

  // Load saved config (only once when data first loads)
  const [configLoaded, setConfigLoaded] = useState(false)
  useEffect(() => {
    if (savedConfig?.config && !configLoaded) {
      const cfg = savedConfig.config
      const params: Record<string, PestParameterConfig> = {}
      for (const p of cfg.parameters || []) {
        // Key format depends on package type
        const key = p.package_type === 'list'
          ? p.property
          : `${p.property}_l${p.layer}`
        params[key] = p
      }
      setSelectedParams(params)
      setObsWeights(cfg.observation_weights || {})
      // Merge saved settings with defaults to ensure all fields exist
      setPestSettings({ ...DEFAULT_SETTINGS, ...(cfg.settings || {}) })
      setConfigLoaded(true)
    }
  }, [savedConfig, configLoaded])

  // Auto-register running PEST runs from API that aren't tracked in run manager
  // This handles runs that were started before the page loaded or before code deployment
  useEffect(() => {
    if (!projectId || !project || !pestRuns) return

    // Find runs from API that are running/queued but not in run manager
    const untrackedRunningRuns = pestRuns.filter(
      (run) =>
        (run.status === 'running' || run.status === 'queued') &&
        !activeRuns[run.id]
    )

    // Register each untracked running run to establish SSE connection
    for (const run of untrackedRunningRuns) {
      // Determine run type from run metadata or default to glm
      const runType = run.run_type === 'pest_ies' ? 'pest_ies' : 'pest_glm'
      globalStartRun({
        runId: run.id,
        projectId: projectId,
        projectName: project.name || 'Unknown Project',
        runType: runType,
      })
    }
  }, [projectId, project, pestRuns, activeRuns, globalStartRun])

  // Auto-select latest run or active run from manager
  useEffect(() => {
    if (projectId && projectActiveRuns.length > 0) {
      // If there's an active run in the manager, use that
      const runningRun = projectActiveRuns.find((r) => r.status === 'running')
      if (runningRun && currentRunId !== runningRun.runId) {
        setCurrentRunId(runningRun.runId)
        selectRun(runningRun.runId)
        return
      }
    }

    // Otherwise, select from the runs list
    if (pestRuns && pestRuns.length > 0 && !currentRunId) {
      const latest = pestRuns[0]
      setCurrentRunId(latest.id)
      // If the run is active in the manager, select it there too
      if (activeRuns[latest.id]) {
        selectRun(latest.id)
      }
    }
  }, [pestRuns, currentRunId, projectId, projectActiveRuns, activeRuns, selectRun])

  // Invalidate queries when run completes (tracked by manager)
  useEffect(() => {
    if (activeRun && activeRun.status !== 'running') {
      queryClient.invalidateQueries({ queryKey: ['pest-runs', projectId] })
    }
  }, [activeRun?.status, projectId, queryClient])

  // Auto-scroll output
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  const saveConfigMutation = useMutation({
    mutationFn: () => {
      const config: PestConfig = {
        parameters: Object.values(selectedParams),
        observation_weights: obsWeights,
        settings: pestSettings,
      }
      return pestApi.saveConfig(projectId!, config)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pest-config', projectId] })
    },
  })

  const startRunMutation = useMutation({
    mutationFn: () => {
      const config: PestConfig = {
        parameters: Object.values(selectedParams),
        observation_weights: obsWeights,
        settings: pestSettings,
      }
      return pestApi.startRun(projectId!, method, undefined, config)
    },
    onSuccess: (data) => {
      setCurrentRunId(data.run_id)

      // Register with global run manager
      globalStartRun({
        runId: data.run_id,
        projectId: projectId!,
        projectName: project?.name || 'Unknown Project',
        runType: method === 'ies' ? 'pest_ies' : 'pest_glm',
      })

      setLocalOutput([])
      setActiveTab('results')
      queryClient.invalidateQueries({ queryKey: ['pest-runs', projectId] })
    },
  })

  const toggleParam = useCallback(
    (param: PestParameter) => {
      // Key format depends on package type
      const key = param.package_type === 'list'
        ? param.property
        : `${param.property}_l${param.layer}`

      setSelectedParams((prev) => {
        if (prev[key]) {
          const next = { ...prev }
          delete next[key]
          return next
        }
        return {
          ...prev,
          [key]: {
            property: param.property,
            layer: param.layer,
            approach: 'multiplier' as ParameterApproach,
            initial_value: 1.0,
            lower_bound: param.suggested_lower,
            upper_bound: param.suggested_upper,
            transform: param.suggested_transform,
            group: param.property,
            package_type: param.package_type,
          },
        }
      })
    },
    []
  )

  const updateParam = useCallback(
    (key: string, field: string, value: number | string) => {
      setSelectedParams((prev) => ({
        ...prev,
        [key]: { ...prev[key], [field]: value },
      }))
    },
    []
  )

  if (projectLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (!project?.is_valid) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
        <Settings className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-yellow-800 mb-2">No Valid Model</h3>
        <p className="text-yellow-600">
          Please upload a valid MODFLOW model first to configure parameter estimation.
        </p>
      </div>
    )
  }

  const nSelected = Object.keys(selectedParams).length
  const hasObs = obsData && Object.keys(obsData.wells || {}).length > 0
  const hasSelectedSets = selectedSetIds.size > 0
  const canRun = nSelected > 0 && (hasObs || hasSelectedSets) && !isRunning

  // Toggle observation set selection
  const toggleSetSelection = (setId: string) => {
    setSelectedSetIds(prev => {
      const next = new Set(prev)
      if (next.has(setId)) {
        next.delete(setId)
      } else {
        next.add(setId)
      }
      return next
    })
  }

  const tabs: { id: TabId; label: string; icon: React.ElementType }[] = [
    { id: 'parameters', label: 'Parameters', icon: Sliders },
    { id: 'observations', label: 'Observations', icon: BarChart },
    { id: 'settings', label: 'PEST++ Settings', icon: Settings },
    { id: 'results', label: 'Results', icon: BarChart },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-800">
          Parameter Estimation & Uncertainty
        </h2>
        <div className="flex items-center gap-3">
          {/* Method selector */}
          <div className="flex rounded-lg border border-slate-300 overflow-hidden">
            <button
              onClick={() => setMethod('glm')}
              className={clsx(
                'px-3 py-2 text-sm font-medium transition-colors',
                method === 'glm'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-slate-600 hover:bg-slate-50'
              )}
            >
              GLM
            </button>
            <button
              onClick={() => setMethod('ies')}
              className={clsx(
                'px-3 py-2 text-sm font-medium transition-colors border-l border-slate-300',
                method === 'ies'
                  ? 'bg-orange-500 text-white'
                  : 'bg-white text-slate-600 hover:bg-slate-50'
              )}
            >
              IES
            </button>
          </div>

          <button
            onClick={() => saveConfigMutation.mutate()}
            disabled={nSelected === 0}
            className="flex items-center gap-2 px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Settings className="h-4 w-4" />
            Save Config
          </button>
          <button
            onClick={() => startRunMutation.mutate()}
            disabled={!canRun}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed',
              method === 'ies'
                ? 'bg-orange-500 hover:bg-orange-600'
                : 'bg-blue-600 hover:bg-blue-700'
            )}
          >
            {isRunning ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            {isRunning
              ? 'Running...'
              : method === 'ies'
                ? 'Run PEST++ IES'
                : 'Run PEST++ GLM'}
          </button>
        </div>
      </div>

      {startRunMutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex items-center gap-2 text-red-700 text-sm">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          {(startRunMutation.error as Error)?.message || 'Failed to start calibration'}
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-200">
        <nav className="flex gap-4">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              )}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
              {tab.id === 'parameters' && nSelected > 0 && (
                <span className="ml-1 px-1.5 py-0.5 bg-blue-100 text-blue-600 text-xs rounded-full">
                  {nSelected}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab content */}
      <div className="bg-white rounded-lg border border-slate-200 p-6">
        {activeTab === 'parameters' && (
          <ParametersTab
            parameters={paramsData?.parameters || []}
            loading={paramsLoading || paramsFetching}
            selectedParams={selectedParams}
            onToggle={toggleParam}
            onUpdate={updateParam}
            onRescan={handleRescanParameters}
            rescanning={rescanning}
          />
        )}
        {activeTab === 'observations' && (
          <ObservationsTab
            projectId={projectId!}
            obsData={obsData}
            obsWeights={obsWeights}
            onWeightsChange={setObsWeights}
            observationSets={observationSets}
            selectedSetIds={selectedSetIds}
            onToggleSet={toggleSetSelection}
            setWeightMultipliers={setWeightMultipliers}
            onSetWeightChange={(setId, weight) => setSetWeightMultipliers(prev => ({ ...prev, [setId]: weight }))}
          />
        )}
        {activeTab === 'settings' && (
          <SettingsTab
            settings={pestSettings}
            onChange={setPestSettings}
            method={method}
          />
        )}
        {activeTab === 'results' && (
          <ResultsTab
            projectId={projectId!}
            runs={pestRuns || []}
            currentRunId={currentRunId}
            isRunning={isRunning}
            output={output}
            outputRef={outputRef}
            onSelectRun={setCurrentRunId}
          />
        )}
      </div>
    </div>
  )
}

// ─── Parameters Tab ──────────────────────────────────────────────

interface ParametersTabProps {
  parameters: PestParameter[]
  loading: boolean
  selectedParams: Record<string, PestParameterConfig>
  onToggle: (param: PestParameter) => void
  onUpdate: (key: string, field: string, value: number | string) => void
  onRescan?: () => void
  rescanning?: boolean
}

function ParametersTab({
  parameters,
  loading,
  selectedParams,
  onToggle,
  onUpdate,
  onRescan,
  rescanning,
}: ParametersTabProps) {
  if (loading) {
    return (
      <div className="h-64 flex flex-col items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500 mb-4" />
        <span className="text-slate-700 font-medium">Scanning model for adjustable parameters...</span>
        <div className="mt-3 text-sm text-slate-500 max-w-md text-center">
          <p>This scans all model packages (NPF, STO, HFB, SFR, GHB, RIV, DRN, RCH, EVT, etc.) to extract adjustable properties.</p>
          <p className="mt-2 text-xs text-slate-400">
            Large models may take several minutes. Please wait...
          </p>
        </div>
      </div>
    )
  }

  if (parameters.length === 0) {
    return (
      <div className="h-48 flex flex-col items-center justify-center gap-4">
        <span className="text-slate-400">No adjustable parameters found in this model.</span>
        {onRescan && (
          <button
            onClick={onRescan}
            disabled={rescanning}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-50 text-blue-700 rounded-md hover:bg-blue-100 disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', rescanning && 'animate-spin')} />
            Rescan Model
          </button>
        )}
      </div>
    )
  }

  // Separate array-based (layer) and list-based (package) parameters
  const arrayParams = parameters.filter(p => p.package_type !== 'list')
  const listParams = parameters.filter(p => p.package_type === 'list')

  // Group array params by property
  const grouped: Record<string, PestParameter[]> = {}
  for (const p of arrayParams) {
    if (!grouped[p.property]) grouped[p.property] = []
    grouped[p.property].push(p)
  }

  return (
    <div className="space-y-6">
      {/* Rescan button header */}
      {onRescan && (
        <div className="flex justify-end">
          <button
            onClick={onRescan}
            disabled={rescanning}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-100 text-slate-600 rounded-md hover:bg-slate-200 disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-3.5 w-3.5', rescanning && 'animate-spin')} />
            Rescan Model
          </button>
        </div>
      )}

      {/* Layer-Based Parameters Section */}
      {Object.keys(grouped).length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-slate-800 mb-1">
            Layer-Based Parameters
          </h3>
          <p className="text-sm text-slate-500 mb-4">
            Apply multipliers or pilot points to adjust property values per layer.
          </p>

          <div className="space-y-4">
            {Object.entries(grouped).map(([prop, params]) => (
              <div key={prop} className="border border-slate-200 rounded-lg overflow-hidden">
                <div className="bg-slate-50 px-4 py-2 font-medium text-slate-700 text-sm">
                  {params[0].property_name} ({params[0].short_name})
                </div>
                <div className="divide-y divide-slate-100">
                  {params.map((param) => {
                    const key = `${param.property}_l${param.layer}`
                    const selected = !!selectedParams[key]
                    const config = selectedParams[key]

                    return (
                      <div
                        key={key}
                        className={clsx(
                          'px-4 py-3 transition-colors',
                          selected ? 'bg-blue-50' : 'hover:bg-slate-50'
                        )}
                      >
                        <div className="flex items-center gap-4">
                          <button
                            onClick={() => onToggle(param)}
                            className="flex-shrink-0"
                          >
                            {selected ? (
                              <CheckSquare className="h-5 w-5 text-blue-600" />
                            ) : (
                              <Square className="h-5 w-5 text-slate-400" />
                            )}
                          </button>

                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-sm">
                              Layer {(param.layer ?? 0) + 1}
                            </div>
                            {param.stats.mean !== null && (
                              <div className="text-xs text-slate-400">
                                Base: mean={param.stats.mean}, range=[
                                {param.stats.min}, {param.stats.max}]
                              </div>
                            )}
                          </div>

                          {selected && config && (
                            <div className="flex items-center gap-3 text-xs">
                              {/* Approach Selector */}
                              <label className="flex items-center gap-1">
                                <span className="text-slate-500">Approach:</span>
                                <select
                                  value={config.approach || 'multiplier'}
                                  onChange={(e) =>
                                    onUpdate(key, 'approach', e.target.value as ParameterApproach)
                                  }
                                  className="px-1 py-0.5 border border-slate-300 rounded text-xs"
                                >
                                  <option value="multiplier">Multiplier</option>
                                  <option value="pilotpoints">Pilot Points</option>
                                </select>
                              </label>
                            </div>
                          )}
                        </div>

                        {/* Expanded options when selected */}
                        {selected && config && (
                          <div className="mt-3 ml-9 flex flex-wrap items-center gap-3 text-xs">
                            {config.approach === 'pilotpoints' && (
                              <label className="flex items-center gap-1">
                                <span className="text-slate-500">Spacing:</span>
                                <input
                                  type="number"
                                  value={config.pp_space ?? 5}
                                  onChange={(e) =>
                                    onUpdate(key, 'pp_space', Number(e.target.value))
                                  }
                                  className="w-14 px-1 py-0.5 border border-slate-300 rounded text-xs"
                                  min="2"
                                  max="20"
                                />
                                <span className="text-slate-400">cells</span>
                              </label>
                            )}
                            <label className="flex items-center gap-1">
                              <span className="text-slate-500">Bounds:</span>
                              <input
                                type="number"
                                value={config.lower_bound}
                                onChange={(e) =>
                                  onUpdate(key, 'lower_bound', Number(e.target.value))
                                }
                                className="w-16 px-1 py-0.5 border border-slate-300 rounded text-xs"
                                step="0.01"
                              />
                              <span className="text-slate-400">-</span>
                              <input
                                type="number"
                                value={config.upper_bound}
                                onChange={(e) =>
                                  onUpdate(key, 'upper_bound', Number(e.target.value))
                                }
                                className="w-16 px-1 py-0.5 border border-slate-300 rounded text-xs"
                                step="0.01"
                              />
                            </label>
                            <label className="flex items-center gap-1">
                              <span className="text-slate-500">Transform:</span>
                              <select
                                value={config.transform}
                                onChange={(e) =>
                                  onUpdate(key, 'transform', e.target.value)
                                }
                                className="px-1 py-0.5 border border-slate-300 rounded text-xs"
                              >
                                <option value="log">Log</option>
                                <option value="none">None</option>
                              </select>
                            </label>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Package-Based Parameters Section (HFB, SFR) */}
      {listParams.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-slate-800 mb-1">
            Package-Based Parameters
          </h3>
          <p className="text-sm text-slate-500 mb-4">
            Apply a single multiplier to all elements in a package.
          </p>

          <div className="border border-slate-200 rounded-lg overflow-hidden">
            <div className="divide-y divide-slate-100">
              {listParams.map((param) => {
                const key = param.property
                const selected = !!selectedParams[key]
                const config = selectedParams[key]

                return (
                  <div
                    key={key}
                    className={clsx(
                      'px-4 py-3 transition-colors',
                      selected ? 'bg-blue-50' : 'hover:bg-slate-50'
                    )}
                  >
                    <div className="flex items-center gap-4">
                      <button
                        onClick={() => onToggle(param)}
                        className="flex-shrink-0"
                      >
                        {selected ? (
                          <CheckSquare className="h-5 w-5 text-blue-600" />
                        ) : (
                          <Square className="h-5 w-5 text-slate-400" />
                        )}
                      </button>

                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">
                          {param.property_name}
                        </div>
                        <div className="text-xs text-slate-400">
                          {param.count?.toLocaleString()} elements
                          {param.stats.mean !== null && (
                            <> | mean={param.stats.mean}, range=[{param.stats.min}, {param.stats.max}]</>
                          )}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">
                          Single multiplier applied to all {param.short_name} values
                        </div>
                      </div>

                      {selected && config && (
                        <div className="flex items-center gap-3 text-xs">
                          <label className="flex items-center gap-1">
                            <span className="text-slate-500">Bounds:</span>
                            <input
                              type="number"
                              value={config.lower_bound}
                              onChange={(e) =>
                                onUpdate(key, 'lower_bound', Number(e.target.value))
                              }
                              className="w-16 px-1 py-0.5 border border-slate-300 rounded text-xs"
                              step="0.01"
                            />
                            <span className="text-slate-400">-</span>
                            <input
                              type="number"
                              value={config.upper_bound}
                              onChange={(e) =>
                                onUpdate(key, 'upper_bound', Number(e.target.value))
                              }
                              className="w-16 px-1 py-0.5 border border-slate-300 rounded text-xs"
                              step="0.01"
                            />
                          </label>
                          <label className="flex items-center gap-1">
                            <span className="text-slate-500">Transform:</span>
                            <select
                              value={config.transform}
                              onChange={(e) =>
                                onUpdate(key, 'transform', e.target.value)
                              }
                              className="px-1 py-0.5 border border-slate-300 rounded text-xs"
                            >
                              <option value="log">Log</option>
                              <option value="none">None</option>
                            </select>
                          </label>
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Observations Tab ────────────────────────────────────────────

interface ObservationsTabProps {
  projectId: string
  obsData: any
  obsWeights: Record<string, number>
  onWeightsChange: (weights: Record<string, number>) => void
  // Multi-set support
  observationSets: ObservationSet[]
  selectedSetIds: Set<string>
  onToggleSet: (setId: string) => void
  setWeightMultipliers: Record<string, number>
  onSetWeightChange: (setId: string, weight: number) => void
}

function ObservationsTab({
  projectId,
  obsData,
  obsWeights,
  onWeightsChange,
  observationSets,
  selectedSetIds,
  onToggleSet,
  setWeightMultipliers,
  onSetWeightChange,
}: ObservationsTabProps) {
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const uploadMutation = useMutation({
    mutationFn: (file: File) => observationsApi.createSet(projectId, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['observations', projectId] })
      queryClient.invalidateQueries({ queryKey: ['observation-sets', projectId] })
    },
  })

  const wells = obsData?.wells || {}
  const wellNames = Object.keys(wells)

  // Calculate totals from selected sets
  const selectedSetsObs = observationSets
    .filter(s => selectedSetIds.has(s.id))
    .reduce((sum, s) => sum + s.n_observations, 0)

  const selectedSetsWells = new Set(
    observationSets
      .filter(s => selectedSetIds.has(s.id))
      .flatMap(s => s.wells)
  ).size

  return (
    <div className="space-y-6">
      {/* Observation Sets Section */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
              <FileSpreadsheet className="h-5 w-5 text-orange-500" />
              Observation Sets
            </h3>
            <p className="text-sm text-slate-500">
              Select which observation sets to include in calibration.
            </p>
          </div>
          <div className="flex gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) uploadMutation.mutate(file)
                if (fileInputRef.current) fileInputRef.current.value = ''
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadMutation.isPending}
              className="flex items-center gap-1 px-3 py-1.5 text-sm border border-slate-300 rounded-lg hover:bg-slate-50 disabled:opacity-50"
            >
              {uploadMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Upload className="h-4 w-4" />
              )}
              Upload CSV
            </button>
          </div>
        </div>

        {observationSets.length === 0 ? (
          <div className="bg-slate-50 rounded-lg p-6 text-center text-slate-400">
            <BarChart className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No observation sets available</p>
            <p className="text-xs mt-1">
              Upload observation CSVs on the Upload page or here
            </p>
          </div>
        ) : (
          <>
            {/* Summary of selected */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
              <div className="text-sm text-blue-700">
                <span className="font-medium">{selectedSetIds.size}</span> of {observationSets.length} sets selected:
                <span className="ml-2 font-medium">{selectedSetsWells}</span> wells,
                <span className="ml-1 font-medium">{selectedSetsObs}</span> observations
              </div>
            </div>

            {/* Observation sets table */}
            <div className="border border-slate-200 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-slate-600 w-10">Include</th>
                    <th className="px-4 py-2 text-left text-slate-600">Set Name</th>
                    <th className="px-4 py-2 text-left text-slate-600">Source</th>
                    <th className="px-4 py-2 text-left text-slate-600">Wells</th>
                    <th className="px-4 py-2 text-left text-slate-600">Obs</th>
                    <th className="px-4 py-2 text-left text-slate-600">Weight</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {observationSets.map((set) => {
                    const isSelected = selectedSetIds.has(set.id)
                    return (
                      <tr
                        key={set.id}
                        className={clsx(
                          'transition-colors',
                          isSelected ? 'bg-blue-50/50' : 'hover:bg-slate-50'
                        )}
                      >
                        <td className="px-4 py-2">
                          <button onClick={() => onToggleSet(set.id)}>
                            {isSelected ? (
                              <CheckSquare className="h-5 w-5 text-blue-600" />
                            ) : (
                              <Square className="h-5 w-5 text-slate-400" />
                            )}
                          </button>
                        </td>
                        <td className="px-4 py-2 font-medium">{set.name}</td>
                        <td className="px-4 py-2 text-slate-500 capitalize">{set.source.replace('_', ' ')}</td>
                        <td className="px-4 py-2 text-slate-500">{set.wells.length}</td>
                        <td className="px-4 py-2 text-slate-500">{set.n_observations}</td>
                        <td className="px-4 py-2">
                          <input
                            type="number"
                            value={setWeightMultipliers[set.id] ?? 1.0}
                            onChange={(e) => onSetWeightChange(set.id, Number(e.target.value))}
                            disabled={!isSelected}
                            className="w-20 px-2 py-1 border border-slate-300 rounded text-xs disabled:opacity-50 disabled:bg-slate-100"
                            step="0.1"
                            min="0"
                          />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      {/* Legacy per-well weights section (for backwards compatibility) */}
      {wellNames.length > 0 && (
        <div className="border-t pt-6">
          <h4 className="text-sm font-medium text-slate-700 mb-3">
            Per-Well Weight Overrides
            <span className="ml-2 text-xs text-slate-400 font-normal">
              (from merged observation data)
            </span>
          </h4>
          <div className="border border-slate-200 rounded-lg overflow-hidden max-h-64 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 sticky top-0">
                <tr>
                  <th className="px-4 py-2 text-left text-slate-600">Well</th>
                  <th className="px-4 py-2 text-left text-slate-600">Location</th>
                  <th className="px-4 py-2 text-left text-slate-600">Obs Count</th>
                  <th className="px-4 py-2 text-left text-slate-600">Weight</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {wellNames.map((name) => {
                  const well = wells[name]
                  const loc =
                    well.node !== undefined
                      ? `L${well.layer + 1} N${well.node + 1}`
                      : `L${well.layer + 1} R${(well.row ?? 0) + 1} C${(well.col ?? 0) + 1}`

                  return (
                    <tr key={name}>
                      <td className="px-4 py-2 font-medium">{name}</td>
                      <td className="px-4 py-2 text-slate-500">{loc}</td>
                      <td className="px-4 py-2 text-slate-500">{well.times?.length || 0}</td>
                      <td className="px-4 py-2">
                        <input
                          type="number"
                          value={obsWeights[name] ?? 1.0}
                          onChange={(e) =>
                            onWeightsChange({
                              ...obsWeights,
                              [name]: Number(e.target.value),
                            })
                          }
                          className="w-20 px-2 py-1 border border-slate-300 rounded text-xs"
                          step="0.1"
                          min="0"
                        />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Settings Tab ────────────────────────────────────────────────

interface SettingsTabProps {
  settings: PestSettings
  onChange: (settings: PestSettings) => void
  method: PestMethod
}

function SettingsTab({ settings, onChange, method }: SettingsTabProps) {
  const update = (field: keyof PestSettings, value: number) => {
    onChange({ ...settings, [field]: value })
  }

  return (
    <div>
      <h3 className="text-lg font-semibold text-slate-800 mb-4">
        PEST++ {method === 'ies' ? 'IES' : 'GLM'} Settings
      </h3>

      {/* Parallel Execution Section */}
      <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="text-sm font-semibold text-blue-800 mb-3">
          Parallel Execution
        </h4>

        {/* Execution Mode Selection */}
        <div className="mb-4 space-y-3">
          {/* Local Process Workers (default) */}
          <label className="flex items-start gap-3 cursor-pointer p-3 rounded-lg border border-transparent hover:bg-blue-100/50 transition-colors">
            <input
              type="radio"
              name="execution_mode"
              checked={settings.network_mode === false && settings.local_containers === false}
              onChange={() => onChange({ ...settings, network_mode: false, local_containers: false, local_agents: 0, remote_agents: 0 })}
              className="mt-1 w-4 h-4 text-blue-600 border-slate-300 focus:ring-blue-500"
            />
            <div className="flex-1">
              <span className="text-sm font-medium text-slate-700">Local Process Workers</span>
              <span className="ml-2 text-xs px-2 py-0.5 bg-slate-100 text-slate-600 rounded">Default</span>
              <p className="text-xs text-slate-500 mt-1">
                Run parallel workers as processes on this machine. Good for single-machine setups.
              </p>
            </div>
          </label>

          {/* Local Container Agents */}
          <label className="flex items-start gap-3 cursor-pointer p-3 rounded-lg border border-transparent hover:bg-blue-100/50 transition-colors">
            <input
              type="radio"
              name="execution_mode"
              checked={settings.local_containers === true && settings.network_mode === false}
              onChange={() => onChange({ ...settings, network_mode: false, local_containers: true, local_agents: 8, remote_agents: 0 })}
              className="mt-1 w-4 h-4 text-blue-600 border-slate-300 focus:ring-blue-500"
            />
            <div className="flex-1">
              <span className="text-sm font-medium text-slate-700">Local Container Agents</span>
              <span className="ml-2 text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded">Docker</span>
              <p className="text-xs text-slate-500 mt-1">
                Run isolated Docker containers on this machine. Better resource isolation and monitoring.
              </p>
            </div>
          </label>

          {/* Hybrid Network Mode */}
          <label className="flex items-start gap-3 cursor-pointer p-3 rounded-lg border border-transparent hover:bg-blue-100/50 transition-colors">
            <input
              type="radio"
              name="execution_mode"
              checked={settings.network_mode === true}
              onChange={() => onChange({ ...settings, network_mode: true, local_containers: true, local_agents: 8, remote_agents: 12 })}
              className="mt-1 w-4 h-4 text-blue-600 border-slate-300 focus:ring-blue-500"
            />
            <div className="flex-1">
              <span className="text-sm font-medium text-slate-700">Hybrid Network Mode</span>
              <span className="ml-2 text-xs px-2 py-0.5 bg-orange-100 text-orange-700 rounded">Distributed</span>
              <p className="text-xs text-slate-500 mt-1">
                Combine local container agents with remote agents from other machines in your network.
              </p>
            </div>
          </label>
        </div>

        {/* Local Process Workers Configuration */}
        {settings.network_mode === false && settings.local_containers === false && (
          <div className="max-w-sm mt-4 pt-4 border-t border-blue-200">
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Number of Workers
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min={1}
                max={16}
                value={settings.num_workers ?? 4}
                onChange={(e) => update('num_workers', Number(e.target.value))}
                className="flex-1 h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
              />
              <input
                type="number"
                min={1}
                max={32}
                value={settings.num_workers ?? 4}
                onChange={(e) => update('num_workers', Number(e.target.value))}
                className="w-16 px-2 py-1 text-center border border-slate-300 rounded-lg"
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              More workers = faster calibration on multi-core systems.
              {method === 'ies' && (
                <span className="block mt-1">
                  With {settings.ies_num_reals ?? 50} realizations and {settings.num_workers ?? 4} workers:
                  ~{Math.ceil((settings.ies_num_reals ?? 50) / (settings.num_workers ?? 4))} model runs per worker per iteration.
                </span>
              )}
            </p>
          </div>
        )}

        {/* Local Container Configuration */}
        {settings.local_containers === true && settings.network_mode === false && (
          <div className="mt-4 pt-4 border-t border-blue-200 space-y-4">
            <div className="max-w-sm">
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Local Container Agents
              </label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min={1}
                  max={16}
                  value={settings.local_agents ?? 8}
                  onChange={(e) => onChange({ ...settings, local_agents: Number(e.target.value) })}
                  className="flex-1 h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
                />
                <input
                  type="number"
                  min={1}
                  max={32}
                  value={settings.local_agents ?? 8}
                  onChange={(e) => onChange({ ...settings, local_agents: Number(e.target.value) })}
                  className="w-16 px-2 py-1 text-center border border-slate-300 rounded-lg"
                />
              </div>
            </div>

            {/* Resource estimate */}
            <div className="bg-white/60 rounded-lg p-3 text-xs">
              <div className="font-medium text-slate-700 mb-1">Resource Estimate (i7-14700)</div>
              <div className="grid grid-cols-2 gap-2 text-slate-600">
                <div>Memory: ~{(settings.local_agents ?? 8) * 2}GB of 32GB</div>
                <div>CPUs: {settings.local_agents ?? 8} of 20 cores</div>
              </div>
              {(settings.local_agents ?? 8) > 14 && (
                <p className="mt-2 text-orange-600">
                  ⚠️ High agent count may cause memory pressure. Recommended: 14-16 agents max.
                </p>
              )}
            </div>
          </div>
        )}

        {/* Hybrid Network Configuration */}
        {settings.network_mode === true && (
          <div className="mt-4 pt-4 border-t border-blue-200 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              {/* Local Agents */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Local Container Agents
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={0}
                    max={16}
                    value={settings.local_agents ?? 8}
                    onChange={(e) => onChange({ ...settings, local_containers: Number(e.target.value) > 0, local_agents: Number(e.target.value) })}
                    className="flex-1 h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <input
                    type="number"
                    min={0}
                    max={32}
                    value={settings.local_agents ?? 8}
                    onChange={(e) => onChange({ ...settings, local_containers: Number(e.target.value) > 0, local_agents: Number(e.target.value) })}
                    className="w-16 px-2 py-1 text-center border border-slate-300 rounded-lg"
                  />
                </div>
                <p className="text-xs text-slate-500 mt-1">On this machine</p>
              </div>

              {/* Remote Agents */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Remote Network Agents
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={0}
                    max={32}
                    value={settings.remote_agents ?? 12}
                    onChange={(e) => onChange({ ...settings, remote_agents: Number(e.target.value) })}
                    className="flex-1 h-2 bg-orange-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={settings.remote_agents ?? 12}
                    onChange={(e) => onChange({ ...settings, remote_agents: Number(e.target.value) })}
                    className="w-16 px-2 py-1 text-center border border-slate-300 rounded-lg"
                  />
                </div>
                <p className="text-xs text-slate-500 mt-1">From other machines</p>
              </div>
            </div>

            {/* Total agents summary */}
            <div className="bg-white/60 rounded-lg p-3">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-slate-700">Total Agents:</span>
                <span className="text-lg font-bold text-blue-600">
                  {(settings.local_agents ?? 0) + (settings.remote_agents ?? 0)}
                </span>
              </div>
              {method === 'ies' && (
                <p className="text-xs text-slate-500 mt-2">
                  With {settings.ies_num_reals ?? 50} realizations:
                  ~{Math.ceil((settings.ies_num_reals ?? 50) / ((settings.local_agents ?? 0) + (settings.remote_agents ?? 0) || 1))} model runs per agent per iteration.
                </p>
              )}
            </div>

            {/* Network setup instructions */}
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 text-xs text-orange-800">
              <div className="font-medium mb-1">Setup Required:</div>
              <ol className="list-decimal list-inside space-y-1 text-orange-700">
                <li>Ensure port 4004 is open on this machine's firewall</li>
                <li>Run <code className="bg-orange-100 px-1 rounded">./scripts/deploy-agents.sh</code> to deploy to remote machines</li>
                <li>Or manually start agents: <code className="bg-orange-100 px-1 rounded">docker compose -f docker-compose.pest.yml up</code></li>
              </ol>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-6 max-w-xl">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            Maximum Iterations (NOPTMAX)
          </label>
          <input
            type="number"
            value={settings.noptmax}
            onChange={(e) => update('noptmax', Number(e.target.value))}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg"
            min={1}
            max={200}
          />
          <p className="text-xs text-slate-400 mt-1">
            {method === 'ies'
              ? 'Maximum number of IES iterations'
              : 'Maximum number of optimization iterations'}
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            Phi Reduction Stop (PHIREDSTP)
          </label>
          <input
            type="number"
            value={settings.phiredstp}
            onChange={(e) => update('phiredstp', Number(e.target.value))}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg"
            step={0.001}
            min={0}
          />
          <p className="text-xs text-slate-400 mt-1">
            Relative phi reduction to trigger early stop
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            No-Phi-Reduction Count (NPHINORED)
          </label>
          <input
            type="number"
            value={settings.nphinored}
            onChange={(e) => update('nphinored', Number(e.target.value))}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg"
            min={1}
            max={20}
          />
          <p className="text-xs text-slate-400 mt-1">
            Iterations without phi reduction before stopping
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            SVD Eigenvalue Threshold
          </label>
          <input
            type="number"
            value={settings.eigthresh}
            onChange={(e) => update('eigthresh', Number(e.target.value))}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg"
            step={1e-7}
            min={0}
          />
          <p className="text-xs text-slate-400 mt-1">
            Threshold for truncating singular values
          </p>
        </div>

        {/* IES-specific settings */}
        {method === 'ies' && (
          <>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Ensemble Size (Realizations)
              </label>
              <input
                type="number"
                value={settings.ies_num_reals ?? 50}
                onChange={(e) => update('ies_num_reals', Number(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                min={10}
                max={1000}
              />
              <p className="text-xs text-slate-400 mt-1">
                Number of parameter realizations in the ensemble
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Initial Lambda
              </label>
              <input
                type="number"
                value={settings.ies_initial_lambda ?? 100.0}
                onChange={(e) =>
                  update('ies_initial_lambda', Number(e.target.value))
                }
                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                step={10}
                min={0.01}
              />
              <p className="text-xs text-slate-400 mt-1">
                Initial Marquardt lambda for regularization
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Bad Phi Sigma
              </label>
              <input
                type="number"
                value={settings.ies_bad_phi_sigma ?? 2.0}
                onChange={(e) =>
                  update('ies_bad_phi_sigma', Number(e.target.value))
                }
                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                step={0.5}
                min={0.5}
              />
              <p className="text-xs text-slate-400 mt-1">
                Standard deviations from mean phi to flag failed realizations
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Subset Size
              </label>
              <input
                type="number"
                value={settings.ies_subset_size ?? Math.max(4, Math.floor((settings.ies_num_reals ?? 50) / 10))}
                onChange={(e) =>
                  update('ies_subset_size', Number(e.target.value))
                }
                className="w-full px-3 py-2 border border-slate-300 rounded-lg"
                min={2}
              />
              <p className="text-xs text-slate-400 mt-1">
                Subset of realizations for lambda testing
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ─── Results Tab ─────────────────────────────────────────────────

interface ResultsTabProps {
  projectId: string
  runs: any[]
  currentRunId: string | null
  isRunning: boolean
  output: string[]
  outputRef: React.RefObject<HTMLDivElement | null>
  onSelectRun: (runId: string) => void
}

function ResultsTab({
  projectId,
  runs,
  currentRunId,
  isRunning,
  output,
  outputRef,
  onSelectRun,
}: ResultsTabProps) {
  const selectedRun = runs.find((r) => r.id === currentRunId)
  const isCompleted = selectedRun?.status === 'completed'
  const isIES = selectedRun?.run_type === 'pest_ies'

  const { data: resultsData } = useQuery({
    queryKey: ['pest-results', projectId, currentRunId],
    queryFn: () => pestApi.getResults(projectId, currentRunId!),
    // Enable query when run is no longer running (completed or failed) to avoid race condition
    // where SSE "completed" event arrives before runs list refetches
    enabled: !!projectId && !!currentRunId && (isCompleted || !isRunning),
    retry: 3,
    retryDelay: 1000,
  })

  // Query for agent status during network mode runs
  const { data: agentStatus } = useQuery({
    queryKey: ['pest-agent-status', projectId, currentRunId],
    queryFn: () => pestApi.getAgentStatus(projectId, currentRunId!),
    enabled: !!projectId && !!currentRunId && isRunning,
    refetchInterval: isRunning ? 3000 : false, // Poll every 3 seconds while running
  })

  const results = resultsData?.results as PestResults | undefined

  return (
    <div>
      <div className="flex items-center gap-4 mb-4">
        <h3 className="text-lg font-semibold text-slate-800">
          Calibration Results
        </h3>
        {runs.length > 0 && (
          <select
            value={currentRunId || ''}
            onChange={(e) => onSelectRun(e.target.value)}
            className="px-2 py-1 text-sm border border-slate-300 rounded"
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name} ({r.status})
              </option>
            ))}
          </select>
        )}
        {selectedRun && (
          <span
            className={clsx(
              'text-xs px-2 py-0.5 rounded font-medium',
              isIES
                ? 'bg-orange-100 text-orange-700'
                : 'bg-blue-100 text-blue-700'
            )}
          >
            {isIES ? 'IES' : 'GLM'}
          </span>
        )}
      </div>

      {/* Agent status for network mode or local container runs */}
      {isRunning && (agentStatus?.network_mode || agentStatus?.local_containers) && (
        <div className="mb-4 p-4 bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-purple-500 animate-pulse" />
                <span className="text-sm font-semibold text-purple-800">
                  {agentStatus?.network_mode ? 'Hybrid Network Mode' : 'Container Agents'}
                </span>
              </div>
              <span className="text-xs text-slate-500">
                Status: {agentStatus.status}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {agentStatus.connected}
                </div>
                <div className="text-xs text-slate-500">Connected</div>
              </div>
              <div className="text-slate-300">/</div>
              <div className="text-center">
                <div className="text-2xl font-bold text-slate-400">
                  {agentStatus.expected}
                </div>
                <div className="text-xs text-slate-500">Expected</div>
              </div>
            </div>
          </div>

          {/* Local/Remote breakdown for hybrid mode */}
          {agentStatus?.network_mode && (agentStatus.local_agents_expected ?? 0) > 0 && (
            <div className="mt-3 pt-3 border-t border-purple-200 grid grid-cols-2 gap-4 text-xs">
              <div className="flex items-center justify-between bg-blue-100/50 rounded-lg px-3 py-2">
                <span className="text-blue-700 font-medium">Local Containers</span>
                <span className="text-blue-800 font-bold">
                  {agentStatus.local_agents_connected ?? 0} / {agentStatus.local_agents_expected ?? 0}
                </span>
              </div>
              <div className="flex items-center justify-between bg-orange-100/50 rounded-lg px-3 py-2">
                <span className="text-orange-700 font-medium">Remote Agents</span>
                <span className="text-orange-800 font-bold">
                  {agentStatus.remote_agents_connected ?? 0} / {agentStatus.remote_agents_expected ?? 0}
                </span>
              </div>
            </div>
          )}

          {agentStatus.connected < agentStatus.expected && (
            <p className="mt-2 text-xs text-purple-700">
              Waiting for {agentStatus.expected - agentStatus.connected} more agent(s) to connect...
              {agentStatus?.network_mode && (
                <span className="block mt-1">
                  {(agentStatus.remote_agents_expected ?? 0) > (agentStatus.remote_agents_connected ?? 0) &&
                    'Start remote agents with: ./scripts/deploy-agents.sh'}
                </span>
              )}
            </p>
          )}
          {agentStatus.connected >= agentStatus.expected && (
            <p className="mt-2 text-xs text-green-700">
              All expected agents connected. Calibration in progress.
            </p>
          )}
        </div>
      )}

      {/* Console output for active runs */}
      {(isRunning || output.length > 0) && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-2">
            <h4 className="text-sm font-medium text-slate-700">Console Output</h4>
            {isRunning && (
              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
            )}
          </div>
          <div
            ref={outputRef as React.RefObject<HTMLDivElement>}
            className="bg-slate-900 text-green-400 font-mono text-xs p-4 rounded-lg h-64 overflow-y-auto"
          >
            {output.map((line, i) => (
              <div key={i}>{line || '\u00A0'}</div>
            ))}
          </div>
        </div>
      )}

      {/* No runs */}
      {runs.length === 0 && !isRunning && (
        <div className="bg-slate-50 rounded-lg p-8 text-center text-slate-400">
          <BarChart className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p>No calibration runs yet</p>
          <p className="text-xs mt-1">
            Configure parameters and observations, then run PEST++
          </p>
        </div>
      )}

      {/* Export buttons */}
      {results && currentRunId && (
        <div className="flex items-center gap-2 mb-4">
          {results.phi_history.length > 0 && (
            <ExportButton url={pestApi.exportPhiCsvUrl(projectId, currentRunId)} label="Phi CSV" />
          )}
          {Object.keys(results.parameters).length > 0 && (
            <ExportButton url={pestApi.exportParametersCsvUrl(projectId, currentRunId)} label="Params CSV" />
          )}
          {results.residuals.length > 0 && (
            <ExportButton url={pestApi.exportResidualsCsvUrl(projectId, currentRunId)} label="Residuals CSV" />
          )}
        </div>
      )}

      {/* Results charts */}
      {results && (
        <div className="space-y-6">
          {/* Phi progress — different for GLM vs IES */}
          {results.phi_history.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-slate-700 mb-2">
                Objective Function (Phi) Progress
                {results.converged && (
                  <span className="ml-2 text-xs text-green-600 bg-green-50 px-2 py-0.5 rounded">
                    Converged
                  </span>
                )}
                {isIES && results.ensemble && (
                  <span className="ml-2 text-xs text-slate-500">
                    {results.ensemble.n_reals} realizations
                    {results.ensemble.n_failed > 0 &&
                      `, ${results.ensemble.n_failed} failed`}
                  </span>
                )}
              </h4>
              <Plot
                data={
                  isIES && results.phi_history[0]?.phi_min !== undefined
                    ? [
                        // IES: shaded envelope (min to max)
                        {
                          x: [
                            ...results.phi_history.map((h) => h.iteration),
                            ...results.phi_history
                              .map((h) => h.iteration)
                              .reverse(),
                          ],
                          y: [
                            ...results.phi_history.map((h) => h.phi_min!),
                            ...results.phi_history
                              .map((h) => h.phi_max!)
                              .reverse(),
                          ],
                          fill: 'toself',
                          fillcolor: 'rgba(249,115,22,0.15)',
                          line: { color: 'transparent' },
                          type: 'scatter' as const,
                          showlegend: true,
                          name: 'Min-Max Range',
                          hoverinfo: 'skip' as const,
                        },
                        // IES: mean line
                        {
                          x: results.phi_history.map((h) => h.iteration),
                          y: results.phi_history.map((h) => h.phi),
                          type: 'scatter' as const,
                          mode: 'lines+markers' as const,
                          marker: { color: '#f97316', size: 6 },
                          line: { color: '#f97316', width: 2 },
                          name: 'Mean Phi',
                        },
                        // IES: min line
                        {
                          x: results.phi_history.map((h) => h.iteration),
                          y: results.phi_history.map((h) => h.phi_min!),
                          type: 'scatter' as const,
                          mode: 'lines' as const,
                          line: {
                            color: '#fb923c',
                            width: 1,
                            dash: 'dot' as const,
                          },
                          name: 'Best Realization',
                        },
                      ]
                    : [
                        // GLM: simple line
                        {
                          x: results.phi_history.map((h) => h.iteration),
                          y: results.phi_history.map((h) => h.phi),
                          type: 'scatter' as const,
                          mode: 'lines+markers' as const,
                          marker: { color: '#3b82f6', size: 6 },
                          line: { color: '#3b82f6', width: 2 },
                          name: 'Phi',
                        },
                      ]
                }
                layout={{
                  autosize: true,
                  height: 300,
                  margin: { l: 60, r: 20, t: 10, b: 40 },
                  xaxis: { title: { text: 'Iteration' } },
                  yaxis: { title: { text: 'Phi' }, type: 'log' },
                  legend: { orientation: 'h', y: -0.2 },
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* IES: Parameter box plots (prior vs posterior) */}
          {isIES && results.ensemble?.par_summary && (
            <EnsembleParamChart ensemble={results.ensemble} />
          )}

          {/* Observed vs Simulated */}
          {results.residuals.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-slate-700 mb-2">
                Observed vs Simulated
                {isIES && (
                  <span className="ml-1 text-xs text-slate-400">
                    (ensemble mean)
                  </span>
                )}
              </h4>
              <Plot
                data={[
                  {
                    x: results.residuals.map((r) => r.observed),
                    y: results.residuals.map((r) => r.simulated),
                    type: 'scatter',
                    mode: 'markers',
                    marker: {
                      color: isIES ? '#f97316' : '#3b82f6',
                      size: 6,
                    },
                    text: results.residuals.map((r) => r.name),
                    name: 'Observations',
                  },
                  {
                    x: (() => {
                      const obs = results.residuals.map((r) => r.observed)
                      const mn = Math.min(...obs)
                      const mx = Math.max(...obs)
                      return [mn, mx]
                    })(),
                    y: (() => {
                      const obs = results.residuals.map((r) => r.observed)
                      const mn = Math.min(...obs)
                      const mx = Math.max(...obs)
                      return [mn, mx]
                    })(),
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#94a3b8', dash: 'dash', width: 1 },
                    showlegend: false,
                  },
                ]}
                layout={{
                  autosize: true,
                  height: 350,
                  margin: { l: 60, r: 20, t: 10, b: 50 },
                  xaxis: { title: { text: 'Observed' } },
                  yaxis: { title: { text: 'Simulated' } },
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Parameter values */}
          {Object.keys(results.parameters).length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-slate-700 mb-2">
                {isIES ? 'Ensemble Mean Parameter Multipliers' : 'Final Parameter Multipliers'}
              </h4>
              <div className="border border-slate-200 rounded-lg overflow-hidden">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-slate-600">
                        Parameter
                      </th>
                      <th className="px-4 py-2 text-right text-slate-600">
                        {isIES ? 'Mean' : 'Multiplier'}
                      </th>
                      {isIES && results.ensemble?.par_summary && (
                        <>
                          <th className="px-4 py-2 text-right text-slate-600">
                            Std
                          </th>
                          <th className="px-4 py-2 text-right text-slate-600">
                            P5-P95
                          </th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {Object.entries(results.parameters).map(([name, value]) => {
                      const summary = results.ensemble?.par_summary?.[name]
                      return (
                        <tr key={name}>
                          <td className="px-4 py-2 font-mono text-xs">
                            {name}
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-xs">
                            {typeof value === 'number'
                              ? value.toExponential(4)
                              : value}
                          </td>
                          {isIES && results.ensemble?.par_summary && (
                            <>
                              <td className="px-4 py-2 text-right font-mono text-xs text-slate-500">
                                {summary
                                  ? summary.std.toExponential(3)
                                  : '-'}
                              </td>
                              <td className="px-4 py-2 text-right font-mono text-xs text-slate-500">
                                {summary
                                  ? `${summary.p5.toExponential(3)} - ${summary.p95.toExponential(3)}`
                                  : '-'}
                              </td>
                            </>
                          )}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Residuals table - virtualized for performance with large datasets */}
          {results.residuals.length > 0 && (
            <VirtualizedResidualsTable residuals={results.residuals} />
          )}
        </div>
      )}
    </div>
  )
}

// ─── Virtualized Residuals Table ───────────────────────────────────

interface Residual {
  name: string
  observed: number
  simulated: number
  residual: number
}

function VirtualizedResidualsTable({ residuals }: { residuals: Residual[] }) {
  const parentRef = useRef<HTMLDivElement>(null)

  const virtualizer = useVirtualizer({
    count: residuals.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 32, // Estimated row height
    overscan: 10, // Render 10 extra items for smooth scrolling
  })

  return (
    <div>
      <h4 className="text-sm font-medium text-slate-700 mb-2">
        Residuals ({residuals.length.toLocaleString()} observations)
      </h4>
      <div className="border border-slate-200 rounded-lg overflow-hidden">
        {/* Fixed header */}
        <div className="bg-slate-50 grid grid-cols-4 text-sm sticky top-0 z-10">
          <div className="px-4 py-2 text-left text-slate-600 font-medium">Observation</div>
          <div className="px-4 py-2 text-right text-slate-600 font-medium">Observed</div>
          <div className="px-4 py-2 text-right text-slate-600 font-medium">Simulated</div>
          <div className="px-4 py-2 text-right text-slate-600 font-medium">Residual</div>
        </div>
        {/* Virtualized rows */}
        <div
          ref={parentRef}
          className="max-h-64 overflow-y-auto"
        >
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              width: '100%',
              position: 'relative',
            }}
          >
            {virtualizer.getVirtualItems().map((virtualRow: VirtualItem) => {
              const r = residuals[virtualRow.index]
              return (
                <div
                  key={r.name}
                  className="grid grid-cols-4 text-sm border-t border-slate-100 absolute w-full"
                  style={{
                    height: `${virtualRow.size}px`,
                    transform: `translateY(${virtualRow.start}px)`,
                  }}
                >
                  <div className="px-4 py-2 font-mono text-xs truncate">{r.name}</div>
                  <div className="px-4 py-2 text-right font-mono text-xs">{r.observed.toFixed(3)}</div>
                  <div className="px-4 py-2 text-right font-mono text-xs">{r.simulated.toFixed(3)}</div>
                  <div
                    className={clsx(
                      'px-4 py-2 text-right font-mono text-xs',
                      Math.abs(r.residual) > 1 ? 'text-red-600' : 'text-green-600'
                    )}
                  >
                    {r.residual.toFixed(3)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Ensemble Parameter Chart ────────────────────────────────────

function EnsembleParamChart({ ensemble }: { ensemble: PestEnsembleData }) {
  const paramNames = Object.keys(ensemble.par_summary)
  if (paramNames.length === 0) return null

  const hasPrior = !!ensemble.prior_summary && Object.keys(ensemble.prior_summary).length > 0

  const posteriorTraces: Plotly.Data[] = paramNames.map((name, i) => {
    const s = ensemble.par_summary[name]
    return {
      type: 'box' as const,
      name,
      y: s.values,
      boxpoints: false,
      marker: { color: '#f97316' },
      line: { color: '#ea580c' },
      offsetgroup: 'posterior',
      legendgroup: 'Posterior',
      showlegend: i === 0,
      legendgrouptitle: i === 0 ? { text: 'Posterior' } : undefined,
    }
  })

  const priorTraces: Plotly.Data[] = hasPrior
    ? paramNames.map((name, i) => {
        const s = ensemble.prior_summary![name]
        if (!s) return null
        return {
          type: 'box' as const,
          name,
          y: s.values,
          boxpoints: false,
          marker: { color: '#94a3b8' },
          line: { color: '#64748b' },
          offsetgroup: 'prior',
          legendgroup: 'Prior',
          showlegend: i === 0,
          legendgrouptitle: i === 0 ? { text: 'Prior' } : undefined,
        }
      }).filter(Boolean) as Plotly.Data[]
    : []

  return (
    <div>
      <h4 className="text-sm font-medium text-slate-700 mb-2">
        Parameter Distributions
        {hasPrior && (
          <span className="ml-1 text-xs text-slate-400">(prior vs posterior)</span>
        )}
      </h4>
      <Plot
        data={[...priorTraces, ...posteriorTraces]}
        layout={{
          autosize: true,
          height: 350,
          margin: { l: 60, r: 20, t: 10, b: 80 },
          boxmode: 'group',
          xaxis: { title: { text: 'Parameter' } },
          yaxis: { title: { text: 'Value' } },
          legend: { orientation: 'h', y: -0.3 },
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
      />
    </div>
  )
}
