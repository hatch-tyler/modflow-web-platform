import { Loader2, Map, BarChart2, TrendingUp, LineChart, Activity } from 'lucide-react'
import type { PostProcessProgress, PostProcessStage } from '../../types'

const STAGE_LABELS: Record<PostProcessStage, string> = {
  heads_budget: 'Processing heads & budget',
  heads: 'Heads complete',
  budget: 'Budget complete',
  listing: 'Parsing listing file',
  geometry: 'Extracting geometry',
  uploading: 'Uploading results',
  finalizing: 'Finalizing',
}

const DATA_TYPE_CONFIG: Record<string, { icon: typeof Map; message: string; needsStage: PostProcessStage[] }> = {
  heads: {
    icon: Map,
    message: 'Waiting for head data processing...',
    needsStage: ['heads'],
  },
  drawdown: {
    icon: TrendingUp,
    message: 'Waiting for head data to compute drawdown...',
    needsStage: ['heads'],
  },
  budget: {
    icon: BarChart2,
    message: 'Waiting for budget data processing...',
    needsStage: ['budget'],
  },
  timeseries: {
    icon: LineChart,
    message: 'Waiting for head data for time series...',
    needsStage: ['heads'],
  },
  listing: {
    icon: Activity,
    message: 'Waiting for listing file parsing...',
    needsStage: ['listing'],
  },
}

interface AwaitingDataProps {
  dataType: 'heads' | 'drawdown' | 'budget' | 'timeseries' | 'listing'
  progress?: PostProcessProgress
  height?: number | string
}

export default function AwaitingData({ dataType, progress, height = 300 }: AwaitingDataProps) {
  const config = DATA_TYPE_CONFIG[dataType]
  const Icon = config.icon

  const completedStages = progress?.postprocess_completed || []
  const currentStage = progress?.postprocess_stage
  const progressPercent = progress?.postprocess_progress || 0

  return (
    <div
      className="flex flex-col items-center justify-center bg-slate-50 rounded-lg border border-slate-200"
      style={{ height }}
    >
      <div className="flex items-center gap-3 mb-4">
        <div className="p-3 bg-blue-100 rounded-full">
          <Icon className="h-6 w-6 text-blue-600" />
        </div>
        <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
      </div>

      <p className="text-slate-600 font-medium mb-2">{config.message}</p>

      {progress && (
        <div className="w-64 mt-2">
          {/* Progress bar */}
          <div className="flex justify-between text-xs text-slate-500 mb-1">
            <span>{currentStage ? STAGE_LABELS[currentStage] : 'Processing...'}</span>
            <span>{progressPercent}%</span>
          </div>
          <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500"
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          {/* Completed stages chips */}
          {completedStages.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-3 justify-center">
              {completedStages.map(stage => (
                <span
                  key={stage}
                  className="px-2 py-0.5 text-xs bg-green-100 text-green-700 rounded-full"
                >
                  {STAGE_LABELS[stage]}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {!progress && (
        <p className="text-xs text-slate-400 mt-1">
          Post-processing will begin shortly...
        </p>
      )}
    </div>
  )
}
