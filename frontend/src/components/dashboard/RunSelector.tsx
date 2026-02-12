import type { Run } from '../../types'

interface RunSelectorProps {
  runs: Run[]
  selectedRunId: string | null
  onSelect: (runId: string) => void
  label?: string
  excludeId?: string | null
}

export default function RunSelector({ runs, selectedRunId, onSelect, label = 'Run:', excludeId }: RunSelectorProps) {
  let completedRuns = runs.filter(r => r.status === 'completed')
  if (excludeId) {
    completedRuns = completedRuns.filter(r => r.id !== excludeId)
  }

  if (completedRuns.length === 0) {
    return (
      <div className="text-sm text-slate-500">
        No completed runs available
      </div>
    )
  }

  return (
    <div className="flex items-center gap-3">
      <label htmlFor={`run-select-${label}`} className="text-sm font-medium text-slate-700">
        {label}
      </label>
      <select
        id={`run-select-${label}`}
        value={selectedRunId || ''}
        onChange={(e) => onSelect(e.target.value)}
        className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
      >
        {completedRuns.map((run) => (
          <option key={run.id} value={run.id}>
            {run.name || run.id.slice(0, 8)} — {run.completed_at
              ? new Date(run.completed_at).toLocaleString()
              : 'N/A'}
          </option>
        ))}
      </select>
    </div>
  )
}
