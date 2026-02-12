import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { BarChart3, Loader2, GitCompareArrows, PlayCircle, Clock, RefreshCw, Radio, CheckCircle2 } from 'lucide-react'
import { projectsApi, simulationApi, resultsApi } from '../services/api'
import type { Run, PostProcessProgress, PostProcessStage } from '../types'
import RunSelector from '../components/dashboard/RunSelector'
import StatCards from '../components/dashboard/StatCards'
import HeadContourChart from '../components/dashboard/HeadContourChart'
import LiveHeadContourChart from '../components/dashboard/LiveHeadContourChart'
import DrawdownChart from '../components/dashboard/DrawdownChart'
import WaterBudgetChart from '../components/dashboard/WaterBudgetChart'
import HeadTimeSeriesChart from '../components/dashboard/HeadTimeSeriesChart'
import ConvergencePlot from '../components/dashboard/ConvergencePlot'
import ExpandableChart from '../components/dashboard/ExpandableChart'
import ChartTabs from '../components/dashboard/ChartTabs'
import AwaitingData from '../components/dashboard/AwaitingData'

const TABS = [
  { id: 'contour', label: 'Head Contour' },
  { id: 'drawdown', label: 'Drawdown' },
  { id: 'budget', label: 'Water Budget' },
  { id: 'timeseries', label: 'Head Time Series' },
  { id: 'convergence', label: 'Mass Balance' },
]

function formatDuration(startTime: string): string {
  const start = new Date(startTime)
  const now = new Date()
  const diffMs = now.getTime() - start.getTime()
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMins / 60)
  const mins = diffMins % 60

  if (diffHours > 0) {
    return `${diffHours}h ${mins}m`
  }
  return `${diffMins}m`
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'running': return 'text-blue-600 bg-blue-50 border-blue-200'
    case 'pending':
    case 'queued': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    case 'completed': return 'text-green-600 bg-green-50 border-green-200'
    case 'failed': return 'text-red-600 bg-red-50 border-red-200'
    default: return 'text-slate-600 bg-slate-50 border-slate-200'
  }
}

const POLLING_OPTIONS = [
  { value: 10000, label: '10 seconds' },
  { value: 20000, label: '20 seconds' },
  { value: 30000, label: '30 seconds' },
  { value: 60000, label: '1 minute' },
  { value: 300000, label: '5 minutes' },
]

const STAGE_LABELS: Record<PostProcessStage, string> = {
  heads_budget: 'Processing',
  heads: 'Heads',
  budget: 'Budget',
  listing: 'Listing',
  geometry: 'Geometry',
  uploading: 'Uploading',
  finalizing: 'Finalizing',
}

interface PostProcessingStatusProps {
  run?: Run
  projectId: string
  pollingInterval: number
  onPollingIntervalChange: (interval: number) => void
  onRefresh: () => void
  activeTab: string
  onTabChange: (tab: string) => void
}

function PostProcessingStatus({
  run,
  projectId,
  pollingInterval,
  onPollingIntervalChange,
  onRefresh,
  activeTab,
  onTabChange,
}: PostProcessingStatusProps) {
  // Get post-processing status from run's convergence_info
  const convergenceInfo = run?.convergence_info as PostProcessProgress | undefined
  const postprocessProgress = convergenceInfo?.postprocess_progress ?? 0
  const postprocessMessage = convergenceInfo?.postprocess_message
  const postprocessStage = convergenceInfo?.postprocess_stage
  const completedStages = convergenceInfo?.postprocess_completed || []

  // Map tab to data type for AwaitingData
  const tabToDataType: Record<string, 'heads' | 'drawdown' | 'budget' | 'timeseries' | 'listing'> = {
    contour: 'heads',
    drawdown: 'drawdown',
    budget: 'budget',
    timeseries: 'timeseries',
    convergence: 'listing',
  }

  return (
    <>
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
          <div className="flex-1">
            <div className="font-medium text-blue-800">Post-processing Results</div>
            <div className="text-sm text-blue-600">
              {postprocessMessage || 'Reading output files and generating visualizations...'}
            </div>
          </div>
          <Link
            to={`/projects/${projectId}/console`}
            className="px-3 py-1.5 text-sm font-medium text-blue-700 bg-blue-100 hover:bg-blue-200 rounded-md transition-colors"
          >
            View Console
          </Link>
        </div>

        {/* Enhanced progress bar with stage indicator */}
        <div className="mt-3">
          <div className="flex justify-between text-xs text-blue-600 mb-1">
            <span>
              {postprocessStage ? STAGE_LABELS[postprocessStage] : (postprocessMessage || 'Starting...')}
            </span>
            <span>{postprocessProgress}%</span>
          </div>
          <div className="w-full h-2 bg-blue-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500"
              style={{ width: `${postprocessProgress}%` }}
            />
          </div>
        </div>

        {/* Completed stages as chips */}
        {completedStages.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-3">
            {completedStages.map(stage => (
              <span
                key={stage}
                className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-green-100 text-green-700 rounded-full"
              >
                <CheckCircle2 className="h-3 w-3" />
                {STAGE_LABELS[stage]}
              </span>
            ))}
          </div>
        )}

        <div className="mt-3 flex items-center gap-2 text-xs text-blue-500">
          <RefreshCw className="h-3 w-3 animate-spin" />
          <span>Auto-refreshing every</span>
          <select
            value={pollingInterval}
            onChange={(e) => onPollingIntervalChange(Number(e.target.value))}
            className="px-1.5 py-0.5 text-xs bg-blue-100 border border-blue-300 rounded text-blue-700 focus:outline-none focus:ring-1 focus:ring-blue-400"
          >
            {POLLING_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <button
            onClick={onRefresh}
            className="ml-2 underline hover:no-underline"
          >
            Refresh now
          </button>
        </div>
      </div>

      {/* Skeleton stat cards */}
      <StatCards awaitingProgress={convergenceInfo} />

      {/* Chart tabs with awaiting data states */}
      <div>
        <ChartTabs tabs={TABS} activeTab={activeTab} onTabChange={onTabChange} />
        <div className="mt-4">
          <ExpandableChart title={TABS.find(t => t.id === activeTab)?.label || ''}>
            {(expanded) => (
              <AwaitingData
                dataType={tabToDataType[activeTab]}
                progress={convergenceInfo}
                height={expanded ? window.innerHeight - 300 : 300}
              />
            )}
          </ExpandableChart>
        </div>
      </div>
    </>
  )
}

interface ActiveRunStatusProps {
  run: Run
  projectId: string
  isPostProcessing?: boolean
  pollingInterval: number
  onPollingIntervalChange: (interval: number) => void
}

function ActiveRunStatus({ run, projectId, isPostProcessing, pollingInterval, onPollingIntervalChange }: ActiveRunStatusProps) {
  const isRunning = run.status === 'running'
  const isPending = ['pending', 'queued'].includes(run.status)

  return (
    <div className={`border rounded-lg p-4 ${getStatusColor(run.status)}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {isRunning || isPostProcessing ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : isPending ? (
            <Clock className="h-5 w-5" />
          ) : (
            <PlayCircle className="h-5 w-5" />
          )}
          <div>
            <div className="font-medium">
              {isPostProcessing ? 'Post-processing Results' : (
                isRunning ? 'Simulation Running' :
                isPending ? 'Simulation Queued' :
                `Simulation ${run.status}`
              )}
            </div>
            <div className="text-sm opacity-75">
              {run.name || 'Unnamed run'}
              {run.started_at && (
                <span className="ml-2">• Running for {formatDuration(run.started_at)}</span>
              )}
            </div>
          </div>
        </div>
        <Link
          to={`/projects/${projectId}/console`}
          className="px-3 py-1.5 text-sm font-medium rounded-md bg-white/50 hover:bg-white/80 transition-colors"
        >
          View Console
        </Link>
      </div>
      <div className="mt-3 flex items-center gap-2 text-xs opacity-75">
        <RefreshCw className="h-3 w-3 animate-spin" />
        <span>Auto-refreshing every</span>
        <select
          value={pollingInterval}
          onChange={(e) => onPollingIntervalChange(Number(e.target.value))}
          className="px-1.5 py-0.5 text-xs bg-white/50 border border-current/20 rounded focus:outline-none"
        >
          {POLLING_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
      {isPostProcessing && (
        <div className="mt-2 text-sm">
          <div className="flex items-center gap-2">
            Reading output files and generating visualizations...
          </div>
          <div className="text-xs mt-1 opacity-75">
            This may take a few minutes for large models.
          </div>
        </div>
      )}
    </div>
  )
}

export default function DashboardPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('contour')
  const [compareMode, setCompareMode] = useState(false)
  const [compareRunId, setCompareRunId] = useState<string | null>(null)

  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  })

  // Track if we should be polling for updates
  const [isPolling, setIsPolling] = useState(false)
  const [pollingInterval, setPollingInterval] = useState(30000) // Default 30 seconds

  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ['runs', projectId],
    queryFn: () => simulationApi.listRuns(projectId!, 50),
    enabled: !!projectId,
    refetchInterval: isPolling ? pollingInterval : false,
  })

  // Determine if there are active runs
  const activeRun = runs?.find(r => ['pending', 'queued', 'running'].includes(r.status))

  // Auto-select latest completed run
  useEffect(() => {
    if (runs && !selectedRunId) {
      const completed = runs.filter(r => r.status === 'completed')
      if (completed.length > 0) {
        setSelectedRunId(completed[0].id)
      }
    }
  }, [runs, selectedRunId])

  // Auto-select compare run when compare mode activates
  useEffect(() => {
    if (compareMode && !compareRunId && runs && selectedRunId) {
      const completed = runs.filter(r => r.status === 'completed' && r.id !== selectedRunId)
      if (completed.length > 0) {
        setCompareRunId(completed[0].id)
      }
    }
    if (!compareMode) {
      setCompareRunId(null)
    }
  }, [compareMode, runs, selectedRunId, compareRunId])

  const { data: summary, isLoading: summaryLoading, isError: summaryError, refetch: refetchSummary } = useQuery({
    queryKey: ['results-summary', projectId, selectedRunId],
    queryFn: () => resultsApi.getSummary(projectId!, selectedRunId!),
    enabled: !!projectId && !!selectedRunId,
    retry: false,
    refetchInterval: isPolling && selectedRunId ? pollingInterval : false,
  })

  // Enable polling when there's an active run or we're waiting for post-processing
  useEffect(() => {
    const shouldPoll = !!activeRun || (summaryError && !!selectedRunId)
    setIsPolling(shouldPoll)
  }, [activeRun, summaryError, selectedRunId])

  const { data: compareSummary } = useQuery({
    queryKey: ['results-summary', projectId, compareRunId],
    queryFn: () => resultsApi.getSummary(projectId!, compareRunId!),
    enabled: !!projectId && !!compareRunId && compareMode,
    retry: false,
  })

  // Fetch live results for active runs - must be before any early returns to satisfy Rules of Hooks
  const { data: liveSummary, isLoading: liveSummaryLoading } = useQuery({
    queryKey: ['live-results-summary', projectId, activeRun?.id],
    queryFn: () => resultsApi.getLiveSummary(projectId!, activeRun!.id),
    enabled: !!projectId && !!activeRun && activeRun.status === 'running',
    refetchInterval: pollingInterval,
    retry: false,
  })

  if (projectLoading || runsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (!project?.is_valid) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
        <BarChart3 className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-yellow-800 mb-2">No Results Available</h3>
        <p className="text-yellow-600">
          Upload and validate a model first, then run a simulation to view results.
        </p>
      </div>
    )
  }

  const completedRuns = (runs || []).filter(r => r.status === 'completed')
  const hasAnyRuns = (runs || []).length > 0

  // Show active run status with live results if available
  if (activeRun) {
    const hasLiveResults = liveSummary && liveSummary.heads_summary.kstpkper_list.length > 0

    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-slate-800">Results Dashboard</h2>
          {hasLiveResults && (
            <div className="flex items-center gap-2 text-green-600 text-sm">
              <Radio className="h-4 w-4 animate-pulse" />
              Live Results Available
            </div>
          )}
        </div>

        <ActiveRunStatus
          run={activeRun}
          projectId={projectId!}
          pollingInterval={pollingInterval}
          onPollingIntervalChange={setPollingInterval}
        />

        {/* Show live results charts if available */}
        {hasLiveResults && liveSummary && (
          <>
            <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-sm text-green-700">
              <div className="flex items-center gap-2">
                <Radio className="h-4 w-4 animate-pulse" />
                <span>
                  <strong>Live Results:</strong> {liveSummary.heads_summary.kstpkper_list.length} timestep(s) processed.
                  Results update as simulation progresses.
                </span>
              </div>
            </div>

            <StatCards summary={liveSummary} />

            <div>
              <ChartTabs tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />
              <div className="mt-4">
                <ExpandableChart title={`${TABS.find(t => t.id === activeTab)?.label || ''} (Live)`}>
                  {(expanded) => (
                    <>
                      {activeTab === 'contour' && (
                        <LiveHeadContourChart
                          projectId={projectId!}
                          runId={activeRun.id}
                          summary={liveSummary}
                          expanded={expanded}
                        />
                      )}
                      {activeTab === 'timeseries' && (
                        <HeadTimeSeriesChart
                          projectId={projectId!}
                          runId={activeRun.id}
                          summary={liveSummary}
                          expanded={expanded}
                        />
                      )}
                      {(activeTab === 'drawdown' || activeTab === 'budget' || activeTab === 'convergence') && (
                        <div className="h-64 flex items-center justify-center text-slate-400">
                          <div className="text-center">
                            <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2" />
                            <p>This view will be available after simulation completes</p>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </ExpandableChart>
              </div>
            </div>
          </>
        )}

        {!hasLiveResults && liveSummaryLoading && (
          <div className="bg-slate-100 rounded-lg p-8 text-center">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto mb-4" />
            <p className="text-slate-500">Waiting for live results...</p>
          </div>
        )}

        {completedRuns.length > 0 && (
          <div className="text-sm text-slate-500">
            Previous completed runs are available below once the current run finishes.
          </div>
        )}
      </div>
    )
  }

  if (completedRuns.length === 0) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-slate-800">Results Dashboard</h2>
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 text-center">
          <BarChart3 className="h-12 w-12 text-slate-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-slate-700 mb-2">No Model Results Yet</h3>
          <p className="text-slate-500 mb-4">
            {hasAnyRuns
              ? 'Previous simulation runs did not complete successfully.'
              : 'Run a simulation to generate results for visualization.'}
          </p>
          <Link
            to={`/projects/${projectId}/console`}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <PlayCircle className="h-4 w-4" />
            Go to Console
          </Link>
        </div>
      </div>
    )
  }

  const canCompare = completedRuns.length >= 2
  const activeTabTitle = TABS.find(t => t.id === activeTab)?.label || ''

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-800">Results Dashboard</h2>
        <div className="flex items-center gap-3">
          <RunSelector
            runs={runs || []}
            selectedRunId={selectedRunId}
            onSelect={setSelectedRunId}
          />
          {canCompare && (
            <button
              onClick={() => setCompareMode(!compareMode)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border transition-colors ${
                compareMode
                  ? 'bg-orange-50 border-orange-300 text-orange-600'
                  : 'border-slate-300 text-slate-500 hover:text-slate-700 hover:border-slate-400'
              }`}
              title="Compare two runs"
            >
              <GitCompareArrows className="h-4 w-4" />
              Compare
            </button>
          )}
          {compareMode && (
            <RunSelector
              runs={runs || []}
              selectedRunId={compareRunId}
              onSelect={setCompareRunId}
              label="vs:"
              excludeId={selectedRunId}
            />
          )}
        </div>
      </div>

      {summaryLoading ? (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
          <span className="ml-3 text-slate-500">Loading results...</span>
        </div>
      ) : summaryError || !summary ? (
        <PostProcessingStatus
          run={runs?.find(r => r.id === selectedRunId)}
          projectId={projectId!}
          pollingInterval={pollingInterval}
          onPollingIntervalChange={setPollingInterval}
          onRefresh={() => refetchSummary()}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      ) : (
        <>
          <StatCards summary={summary} compareSummary={compareMode ? compareSummary : undefined} />

          <div>
            <ChartTabs tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />
            <div className="mt-4">
              <ExpandableChart title={activeTabTitle}>
                {(expanded) => (
                  <>
                    {activeTab === 'contour' && (
                      <HeadContourChart
                        projectId={projectId!}
                        runId={selectedRunId!}
                        summary={summary}
                        expanded={expanded}
                        compareRunId={compareMode ? compareRunId : undefined}
                        compareSummary={compareMode ? compareSummary : undefined}
                      />
                    )}
                    {activeTab === 'drawdown' && (
                      <DrawdownChart
                        projectId={projectId!}
                        runId={selectedRunId!}
                        summary={summary}
                        expanded={expanded}
                      />
                    )}
                    {activeTab === 'budget' && (
                      <WaterBudgetChart
                        projectId={projectId!}
                        runId={selectedRunId!}
                        summary={summary}
                        expanded={expanded}
                        compareRunId={compareMode ? compareRunId : undefined}
                        convergenceInfo={runs?.find(r => r.id === selectedRunId)?.convergence_info}
                        startDate={summary?.metadata?.start_date}
                        timeUnit={summary?.metadata?.time_unit}
                        lengthUnit={summary?.metadata?.length_unit}
                        stressPeriodData={summary?.metadata?.stress_period_data ?? project?.stress_period_data}
                      />
                    )}
                    {activeTab === 'timeseries' && (
                      <HeadTimeSeriesChart
                        projectId={projectId!}
                        runId={selectedRunId!}
                        summary={summary}
                        expanded={expanded}
                        compareRunId={compareMode ? compareRunId : undefined}
                      />
                    )}
                    {activeTab === 'convergence' && (
                      <ConvergencePlot
                        summary={summary}
                        expanded={expanded}
                        compareSummary={compareMode ? compareSummary : undefined}
                      />
                    )}
                  </>
                )}
              </ExpandableChart>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
