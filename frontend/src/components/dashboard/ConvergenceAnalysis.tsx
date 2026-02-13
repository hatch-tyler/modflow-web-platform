import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Loader2, AlertTriangle, Settings, Zap } from 'lucide-react'
import { convergenceApi } from '../../services/api'
import type { ResultsSummary } from '../../types'
import IterationHeatmap from './IterationHeatmap'
import StressCorrelationChart from './StressCorrelationChart'
import ProblemCellsTable from './ProblemCellsTable'
import SolverSettingsCard from './SolverSettingsCard'
import RefinementPanel from './RefinementPanel'

interface ConvergenceAnalysisProps {
  projectId: string
  runId: string
  summary: ResultsSummary
  expanded: boolean
  compareRunId?: string
  compareSummary?: ResultsSummary
}

type SubTab = 'heatmap' | 'stress' | 'cells' | 'solver' | 'refinements'

const SUB_TABS: { id: SubTab; label: string }[] = [
  { id: 'heatmap', label: 'Iteration Heatmap' },
  { id: 'stress', label: 'Stress Correlation' },
  { id: 'cells', label: 'Problem Cells' },
  { id: 'solver', label: 'Solver Settings' },
  { id: 'refinements', label: 'Refinements' },
]

export default function ConvergenceAnalysis({
  projectId,
  runId,
  summary: _summary,
  expanded,
}: ConvergenceAnalysisProps) {
  const [activeSubTab, setActiveSubTab] = useState<SubTab>('heatmap')

  const { data: convergenceDetail, isLoading: detailLoading, error: detailError } = useQuery({
    queryKey: ['convergence-detail', projectId, runId],
    queryFn: () => convergenceApi.getDetail(projectId, runId),
    retry: false,
  })

  const { data: stressData, isLoading: stressLoading } = useQuery({
    queryKey: ['stress-data', projectId, runId],
    queryFn: () => convergenceApi.getStressData(projectId, runId),
    retry: false,
  })

  if (detailLoading) {
    return (
      <div className="flex items-center justify-center" style={{ height: expanded ? window.innerHeight - 300 : 300 }}>
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <span className="ml-3 text-slate-500">Loading convergence data...</span>
      </div>
    )
  }

  if (detailError || !convergenceDetail) {
    return (
      <div className="flex flex-col items-center justify-center text-slate-400" style={{ height: expanded ? window.innerHeight - 300 : 300 }}>
        <AlertTriangle className="h-8 w-8 mb-2" />
        <p>Convergence analysis data not available.</p>
        <p className="text-sm mt-1">This data is generated during post-processing for runs after this feature was added.</p>
      </div>
    )
  }

  const height = expanded ? window.innerHeight - 340 : 300

  return (
    <div>
      {/* Summary stats banner */}
      <div className="flex gap-4 mb-4">
        <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-lg text-sm">
          <span className="text-slate-500">Timesteps:</span>
          <span className="font-medium">{convergenceDetail.total_timesteps}</span>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm ${
          convergenceDetail.failed_timesteps > 0
            ? 'bg-red-50 text-red-700'
            : 'bg-green-50 text-green-700'
        }`}>
          <span>{convergenceDetail.failed_timesteps > 0 ? 'Failed:' : 'All converged'}</span>
          {convergenceDetail.failed_timesteps > 0 && (
            <span className="font-medium">{convergenceDetail.failed_timesteps}</span>
          )}
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-lg text-sm">
          <Settings className="h-3.5 w-3.5" />
          <span>{convergenceDetail.solver_settings.solver_type || 'Unknown solver'}</span>
          {convergenceDetail.solver_settings.complexity && (
            <span className="text-slate-400">({convergenceDetail.solver_settings.complexity})</span>
          )}
        </div>
        {convergenceDetail.problem_cells.length > 0 && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-amber-50 text-amber-700 rounded-lg text-sm">
            <Zap className="h-3.5 w-3.5" />
            <span>{convergenceDetail.problem_cells.length} problem cells</span>
          </div>
        )}
      </div>

      {/* Sub-tabs */}
      <div className="flex border-b border-slate-200 mb-4">
        {SUB_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveSubTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeSubTab === tab.id
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      {activeSubTab === 'heatmap' && (
        <IterationHeatmap
          convergenceDetail={convergenceDetail}
          height={height}
        />
      )}
      {activeSubTab === 'stress' && (
        <StressCorrelationChart
          convergenceDetail={convergenceDetail}
          stressData={stressData ?? null}
          height={height}
          loading={stressLoading}
        />
      )}
      {activeSubTab === 'cells' && (
        <ProblemCellsTable
          problemCells={convergenceDetail.problem_cells}
          height={height}
        />
      )}
      {activeSubTab === 'solver' && (
        <SolverSettingsCard
          solverSettings={convergenceDetail.solver_settings}
        />
      )}
      {activeSubTab === 'refinements' && (
        <RefinementPanel
          projectId={projectId}
          runId={runId}
        />
      )}
    </div>
  )
}
