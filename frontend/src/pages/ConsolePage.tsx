import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Terminal, Play, Square, Loader2, History, CheckCircle, XCircle, Clock, Radio } from 'lucide-react'
import clsx from 'clsx'
import { projectsApi, simulationApi } from '../services/api'
import { useRunManager } from '../store/runManager'
import type { Run } from '../types'

type RunStatus = 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

function getStatusIcon(status: RunStatus) {
  switch (status) {
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'failed':
      return <XCircle className="h-4 w-4 text-red-500" />
    case 'cancelled':
      return <XCircle className="h-4 w-4 text-yellow-500" />
    case 'running':
    case 'queued':
    case 'pending':
      return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
    default:
      return <Clock className="h-4 w-4 text-slate-400" />
  }
}

function getStatusColor(status: RunStatus) {
  switch (status) {
    case 'completed':
      return 'bg-green-100 text-green-800'
    case 'failed':
      return 'bg-red-100 text-red-800'
    case 'cancelled':
      return 'bg-yellow-100 text-yellow-800'
    case 'running':
      return 'bg-blue-100 text-blue-800'
    case 'queued':
    case 'pending':
      return 'bg-slate-100 text-slate-800'
    default:
      return 'bg-slate-100 text-slate-600'
  }
}

export default function ConsolePage() {
  const { projectId } = useParams<{ projectId: string }>()
  const queryClient = useQueryClient()
  const terminalRef = useRef<HTMLDivElement>(null)

  // Global run manager
  const {
    activeRuns,
    selectedRunId,
    startRun,
    selectRun,
    getActiveRunsForProject,
  } = useRunManager()

  // Local state for initial output before run starts
  const [localOutput, setLocalOutput] = useState<string[]>([
    '$ MODFLOW Web Platform Console',
    '$ Ready to run simulations...',
    '',
  ])

  // Simulation options
  const [saveBudget, setSaveBudget] = useState(true)

  // Get active runs for this project
  const projectRuns = projectId ? getActiveRunsForProject(projectId) : []
  const currentRun = selectedRunId ? activeRuns[selectedRunId] : null
  const isRunning = currentRun?.status === 'running'

  // Determine which output to display
  const displayOutput = currentRun ? currentRun.output : localOutput

  // Fetch project
  const { data: project, isLoading: isLoadingProject } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  })

  // Fetch run history
  const { data: runs, isLoading: isLoadingRuns } = useQuery({
    queryKey: ['simulation-runs', projectId],
    queryFn: () => simulationApi.listRuns(projectId!),
    enabled: !!projectId,
    refetchInterval: isRunning ? 3000 : false,
  })

  // Auto-register running runs from API that aren't tracked in run manager
  // This handles runs that were started before the page loaded or before code deployment
  useEffect(() => {
    if (!projectId || !project || !runs) return

    // Find runs from API that are running/queued but not in run manager
    const untrackedRunningRuns = runs.filter(
      (run: Run) =>
        (run.status === 'running' || run.status === 'queued') &&
        !activeRuns[run.id]
    )

    // Register each untracked running run to establish SSE connection
    for (const run of untrackedRunningRuns) {
      startRun({
        runId: run.id,
        projectId: projectId,
        projectName: project.name || 'Unknown Project',
        runType: 'simulation',
        startedAt: run.started_at ? new Date(run.started_at) : undefined,
        isReconnect: true,
      })
    }
  }, [projectId, project, runs, activeRuns, startRun])

  // Auto-select the first active run for this project when page loads
  useEffect(() => {
    if (projectId && projectRuns.length > 0 && !currentRun) {
      // Find a running run first, otherwise use the most recent
      const runningRun = projectRuns.find((r) => r.status === 'running')
      if (runningRun) {
        selectRun(runningRun.runId)
      }
    }
  }, [projectId, projectRuns, currentRun, selectRun])

  // Start simulation mutation
  const startMutation = useMutation({
    mutationFn: () => simulationApi.start(projectId!, { save_budget: saveBudget }),
    onSuccess: (data) => {
      // Register with global run manager
      startRun({
        runId: data.run_id,
        projectId: projectId!,
        projectName: project?.name || 'Unknown Project',
        runType: 'simulation',
      })

      setLocalOutput((prev) => [
        ...prev,
        `$ Starting simulation (run: ${data.run_id.slice(0, 8)})...`,
        '',
      ])

      queryClient.invalidateQueries({ queryKey: ['simulation-runs', projectId] })
    },
    onError: (error: Error & { response?: { data?: { detail?: string } } }) => {
      const message = error.response?.data?.detail || error.message
      setLocalOutput((prev) => [...prev, `$ ERROR: ${message}`, ''])
    },
  })

  // Cancel simulation mutation
  const cancelMutation = useMutation({
    mutationFn: () => simulationApi.cancel(projectId!, currentRun?.runId || ''),
    onSuccess: () => {
      // Output will be updated via SSE
    },
    onError: (error: Error & { response?: { data?: { detail?: string } } }) => {
      const message = error.response?.data?.detail || error.message
      setLocalOutput((prev) => [...prev, `$ Cancel failed: ${message}`, ''])
    },
  })

  // Invalidate queries when run completes
  useEffect(() => {
    if (currentRun && currentRun.status !== 'running') {
      queryClient.invalidateQueries({ queryKey: ['simulation-runs', projectId] })
    }
  }, [currentRun?.status, projectId, queryClient])

  // Auto-scroll to bottom
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [displayOutput])

  const handleStartRun = useCallback(() => {
    startMutation.mutate()
  }, [startMutation])

  const handleStopRun = useCallback(() => {
    if (currentRun) {
      cancelMutation.mutate()
    }
  }, [cancelMutation, currentRun])

  const handleClearConsole = useCallback(() => {
    setLocalOutput([
      '$ MODFLOW Web Platform Console',
      '$ Ready to run simulations...',
      '',
    ])
    selectRun(null)
  }, [selectRun])

  const handleSelectRun = useCallback((runId: string) => {
    selectRun(runId)
  }, [selectRun])

  if (isLoadingProject) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (!project?.is_valid) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
        <Terminal className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-yellow-800 mb-2">No Valid Model</h3>
        <p className="text-yellow-600">
          Please upload a valid MODFLOW model first to run simulations.
        </p>
      </div>
    )
  }

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4">
      {/* Main Console */}
      <div className="flex-1 flex flex-col">
        {/* Controls */}
        <div className="flex items-center gap-4 mb-4">
          <h2 className="text-xl font-bold text-slate-800">Simulation Console</h2>
          <span className="px-2 py-1 bg-slate-100 rounded text-sm text-slate-600">
            {project.model_type?.toUpperCase() || 'MF6'}
          </span>

          {/* Active runs selector */}
          {projectRuns.length > 0 && (
            <div className="flex items-center gap-2">
              <Radio className="h-4 w-4 text-blue-500" />
              <select
                value={selectedRunId || ''}
                onChange={(e) => handleSelectRun(e.target.value)}
                className="text-sm border border-slate-300 rounded px-2 py-1"
              >
                <option value="">Select run...</option>
                {projectRuns.map((run) => (
                  <option key={run.runId} value={run.runId}>
                    {run.runId.slice(0, 8)} - {run.status}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="flex-1" />

          {/* Simulation options */}
          {!isRunning && (
            <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
              <input
                type="checkbox"
                checked={saveBudget}
                onChange={(e) => setSaveBudget(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
              />
              <span>Save Water Budget</span>
              <span className="text-xs text-slate-400" title="Generates CBC file for water budget analysis on the Dashboard. Disable to reduce output file size if budget data is not needed.">(?)</span>
            </label>
          )}

          <button
            onClick={handleClearConsole}
            className="px-3 py-1 text-sm text-slate-600 hover:text-slate-800"
          >
            Clear
          </button>
          {isRunning ? (
            <button
              onClick={handleStopRun}
              disabled={cancelMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors disabled:opacity-50"
            >
              <Square className="h-4 w-4" />
              {cancelMutation.isPending ? 'Cancelling...' : 'Stop'}
            </button>
          ) : (
            <button
              onClick={handleStartRun}
              disabled={startMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              {startMutation.isPending ? 'Starting...' : 'Run Simulation'}
            </button>
          )}
        </div>

        {/* Terminal */}
        <div
          ref={terminalRef}
          className="flex-1 bg-slate-900 rounded-lg p-4 font-mono text-sm text-green-400 overflow-auto"
        >
          {displayOutput.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap">
              {line || '\u00A0'}
            </div>
          ))}
          {isRunning && (
            <div className="flex items-center gap-2 mt-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="animate-pulse">Running...</span>
            </div>
          )}
        </div>
      </div>

      {/* Run History Sidebar */}
      <div className="w-80 flex flex-col bg-slate-50 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-4">
          <History className="h-5 w-5 text-slate-600" />
          <h3 className="font-semibold text-slate-800">Run History</h3>
          {projectRuns.filter((r) => r.status === 'running').length > 0 && (
            <span className="ml-auto px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full">
              {projectRuns.filter((r) => r.status === 'running').length} active
            </span>
          )}
        </div>

        {isLoadingRuns ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
          </div>
        ) : runs?.length === 0 ? (
          <p className="text-sm text-slate-500 text-center py-8">
            No runs yet. Click "Run Simulation" to start.
          </p>
        ) : (
          <div className="flex-1 overflow-auto space-y-2">
            {runs?.map((run: Run) => {
              const isActive = activeRuns[run.id]
              const isSelected = selectedRunId === run.id
              const isRunningOrQueued = run.status === 'running' || run.status === 'queued'
              const canConnect = isActive || isRunningOrQueued

              const handleRunClick = () => {
                if (isActive) {
                  // Already tracked, just select it
                  handleSelectRun(run.id)
                } else if (isRunningOrQueued && project) {
                  // Not tracked but running - register and select
                  startRun({
                    runId: run.id,
                    projectId: projectId!,
                    projectName: project.name || 'Unknown Project',
                    runType: 'simulation',
                    startedAt: run.started_at ? new Date(run.started_at) : undefined,
                    isReconnect: true,
                  })
                }
              }

              return (
                <div
                  key={run.id}
                  onClick={handleRunClick}
                  className={clsx(
                    'bg-white rounded-lg p-3 border transition-colors',
                    isSelected
                      ? 'border-blue-500 ring-2 ring-blue-200'
                      : 'border-slate-200 hover:border-slate-300',
                    canConnect && 'cursor-pointer'
                  )}
                >
                  <div className="flex items-center gap-2 mb-1">
                    {getStatusIcon(run.status as RunStatus)}
                    <span className="font-medium text-sm text-slate-800 truncate flex-1">
                      {run.name || `Run ${run.id.slice(0, 8)}`}
                    </span>
                    {isActive ? (
                      <span className="px-1.5 py-0.5 bg-blue-50 text-blue-600 text-xs rounded">
                        Live
                      </span>
                    ) : isRunningOrQueued ? (
                      <span className="px-1.5 py-0.5 bg-yellow-50 text-yellow-600 text-xs rounded">
                        Connect
                      </span>
                    ) : null}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span className={`px-2 py-0.5 rounded ${getStatusColor(run.status as RunStatus)}`}>
                      {run.status}
                    </span>
                    {run.exit_code !== null && (
                      <span>Exit: {run.exit_code}</span>
                    )}
                  </div>
                  {run.started_at && (
                    <div className="text-xs text-slate-400 mt-1">
                      {new Date(run.started_at).toLocaleString()}
                    </div>
                  )}
                  {run.completed_at && run.started_at && (
                    <div className="text-xs text-slate-400">
                      Duration: {Math.round((new Date(run.completed_at).getTime() - new Date(run.started_at).getTime()) / 1000)}s
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
