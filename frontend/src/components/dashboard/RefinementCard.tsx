import { ArrowRight } from 'lucide-react'
import type { Refinement } from '../../types'

interface RefinementCardProps {
  refinement: Refinement
  selected: boolean
  onToggle: () => void
}

const PRIORITY_STYLES = {
  high: 'bg-red-50 text-red-700 border-red-200',
  medium: 'bg-amber-50 text-amber-700 border-amber-200',
  low: 'bg-blue-50 text-blue-700 border-blue-200',
}

const CATEGORY_LABELS = {
  solver: 'Solver',
  temporal: 'Temporal',
  package: 'Package',
}

export default function RefinementCard({ refinement, selected, onToggle }: RefinementCardProps) {
  const canApply = refinement.file_modification !== null

  return (
    <div
      className={`border rounded-lg p-4 transition-colors ${
        selected
          ? 'border-blue-400 bg-blue-50/50'
          : 'border-slate-200 bg-white hover:border-slate-300'
      }`}
    >
      <div className="flex items-start gap-3">
        {/* Checkbox */}
        {canApply && (
          <label className="mt-0.5 flex-shrink-0">
            <input
              type="checkbox"
              checked={selected}
              onChange={onToggle}
              className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
            />
          </label>
        )}

        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-sm font-medium text-slate-800">{refinement.title}</h4>
            <span className={`inline-flex px-2 py-0.5 text-xs font-medium rounded-full border ${PRIORITY_STYLES[refinement.priority]}`}>
              {refinement.priority}
            </span>
            <span className="inline-flex px-2 py-0.5 text-xs font-medium rounded-full bg-slate-100 text-slate-600">
              {CATEGORY_LABELS[refinement.category] || refinement.category}
            </span>
          </div>

          {/* Description */}
          <p className="text-sm text-slate-600 mb-2">{refinement.description}</p>

          {/* Value change */}
          <div className="flex items-center gap-2 text-sm">
            <span className="font-mono px-2 py-0.5 bg-slate-100 rounded text-slate-600">
              {refinement.current_value}
            </span>
            <ArrowRight className="h-4 w-4 text-slate-400" />
            <span className="font-mono px-2 py-0.5 bg-green-100 rounded text-green-700 font-medium">
              {refinement.suggested_value}
            </span>
          </div>

          {/* File modification details */}
          {refinement.file_modification && (
            <div className="mt-2 text-xs text-slate-400">
              Modifies: {refinement.file_modification.package}
              {refinement.file_modification.block && ` > ${refinement.file_modification.block}`}
              {' > '}{refinement.file_modification.variable}
            </div>
          )}
          {!canApply && (
            <div className="mt-2 text-xs text-slate-400 italic">
              Manual intervention required
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
