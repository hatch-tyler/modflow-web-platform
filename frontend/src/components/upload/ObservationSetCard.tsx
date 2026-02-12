import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  FileSpreadsheet,
  Edit2,
  Trash2,
  ChevronDown,
  ChevronUp,
  Upload,
  FileText,
  Clock,
  Loader2,
} from 'lucide-react'
import clsx from 'clsx'
import { observationsApi } from '../../services/api'
import type { ObservationSet, ObservationSetSource } from '../../types'

interface ObservationSetCardProps {
  set: ObservationSet
  projectId: string
  onEdit?: (set: ObservationSet) => void
}

const SOURCE_LABELS: Record<ObservationSetSource, { label: string; icon: React.ElementType; color: string }> = {
  upload: { label: 'Uploaded', icon: Upload, color: 'text-blue-600 bg-blue-50' },
  zip_detected: { label: 'Auto-detected', icon: FileText, color: 'text-green-600 bg-green-50' },
  manual_marked: { label: 'From file', icon: FileSpreadsheet, color: 'text-orange-600 bg-orange-50' },
}

export default function ObservationSetCard({ set, projectId, onEdit }: ObservationSetCardProps) {
  const queryClient = useQueryClient()
  const [isExpanded, setIsExpanded] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  const deleteMutation = useMutation({
    mutationFn: () => observationsApi.deleteSet(projectId, set.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['observation-sets', projectId] })
      queryClient.invalidateQueries({ queryKey: ['observations', projectId] })
    },
  })

  const sourceMeta = SOURCE_LABELS[set.source]
  const SourceIcon = sourceMeta.icon

  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 bg-slate-50">
        <FileSpreadsheet className="h-5 w-5 text-orange-500 flex-shrink-0" />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-slate-800 truncate">{set.name}</span>
            <span
              className={clsx(
                'text-[10px] px-1.5 py-0.5 rounded font-medium flex items-center gap-1',
                sourceMeta.color
              )}
            >
              <SourceIcon className="h-3 w-3" />
              {sourceMeta.label}
            </span>
          </div>
          <div className="text-xs text-slate-500 flex items-center gap-3 mt-0.5">
            <span>{set.wells.length} wells</span>
            <span>{set.n_observations} observations</span>
            <span className="capitalize">{set.format} format</span>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          {onEdit && (
            <button
              onClick={() => onEdit(set)}
              className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded transition-colors"
              title="Edit column mapping"
            >
              <Edit2 className="h-4 w-4" />
            </button>
          )}

          {showDeleteConfirm ? (
            <div className="flex items-center gap-1">
              <button
                onClick={() => {
                  deleteMutation.mutate()
                  setShowDeleteConfirm(false)
                }}
                disabled={deleteMutation.isPending}
                className="px-2 py-1 text-xs bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50"
              >
                {deleteMutation.isPending ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  'Delete'
                )}
              </button>
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-2 py-1 text-xs border border-slate-300 rounded hover:bg-slate-100"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
              title="Delete observation set"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          )}

          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded transition-colors"
          >
            {isExpanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="px-4 py-3 border-t border-slate-100 text-sm">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-slate-500 mb-1">Wells</div>
              <div className="flex flex-wrap gap-1">
                {set.wells.slice(0, 8).map((well) => (
                  <span
                    key={well}
                    className="px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded text-xs"
                  >
                    {well}
                  </span>
                ))}
                {set.wells.length > 8 && (
                  <span className="text-xs text-slate-400">
                    +{set.wells.length - 8} more
                  </span>
                )}
              </div>
            </div>

            <div>
              <div className="text-xs text-slate-500 mb-1">Created</div>
              <div className="flex items-center gap-1 text-slate-600">
                <Clock className="h-3.5 w-3.5" />
                {new Date(set.created_at).toLocaleDateString()}
              </div>
            </div>
          </div>

          {set.column_mapping && (
            <div className="mt-3 pt-3 border-t border-slate-100">
              <div className="text-xs text-slate-500 mb-2">Column Mapping</div>
              <div className="grid grid-cols-3 gap-2 text-xs">
                {Object.entries(set.column_mapping).map(([key, value]) => {
                  if (value === null || value === undefined) return null
                  return (
                    <div key={key} className="flex items-center gap-1">
                      <span className="text-slate-400">{key}:</span>
                      <span className="font-mono text-slate-600">
                        {typeof value === 'number' ? `(${value})` : value}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

interface ObservationSetListProps {
  sets: ObservationSet[]
  projectId: string
  onEdit?: (set: ObservationSet) => void
  onUpload?: () => void
  onConfigure?: (set: ObservationSet) => void
}

export function ObservationSetList({ sets, projectId, onEdit, onUpload, onConfigure }: ObservationSetListProps) {
  // Separate configured sets from detected (unconfigured) sets
  const configuredSets = sets.filter(s => s.source !== 'zip_detected' || s.column_mapping)
  const detectedSets = sets.filter(s => s.source === 'zip_detected' && !s.column_mapping)

  const totalObs = configuredSets.reduce((sum, s) => sum + s.n_observations, 0)
  const totalWells = new Set(configuredSets.flatMap((s) => s.wells)).size

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-600">
          <span className="font-medium">{configuredSets.length}</span> observation sets,{' '}
          <span className="font-medium">{totalWells}</span> unique wells,{' '}
          <span className="font-medium">{totalObs}</span> total observations
        </div>
        {onUpload && (
          <button
            onClick={onUpload}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm border border-orange-300 text-orange-600 rounded-lg hover:bg-orange-50 transition-colors"
          >
            <Upload className="h-4 w-4" />
            Upload CSV
          </button>
        )}
      </div>

      {/* Detected observations that need configuration */}
      {detectedSets.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-amber-700 font-medium text-sm mb-3">
            <FileSpreadsheet className="h-4 w-4" />
            {detectedSets.length} observation file{detectedSets.length > 1 ? 's' : ''} detected - needs configuration
          </div>
          <div className="space-y-2">
            {detectedSets.map((set) => (
              <div
                key={set.id}
                className="flex items-center justify-between bg-white rounded px-3 py-2 border border-amber-100"
              >
                <div>
                  <div className="font-medium text-sm text-slate-700">{set.name}</div>
                  <div className="text-xs text-slate-500">
                    {set.format} format • ~{set.n_observations} observations detected
                  </div>
                </div>
                {onConfigure && (
                  <button
                    onClick={() => onConfigure(set)}
                    className="px-3 py-1 text-xs font-medium bg-amber-500 text-white rounded hover:bg-amber-600 transition-colors"
                  >
                    Configure
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Configured set cards */}
      {configuredSets.length === 0 ? (
        <div className="bg-slate-50 rounded-lg p-6 text-center text-slate-400">
          <FileSpreadsheet className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p>No observation sets configured</p>
          <p className="text-xs mt-1">
            {detectedSets.length > 0
              ? 'Configure the detected files above or upload a new CSV'
              : 'Upload a CSV or mark an existing file as observations'}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {configuredSets.map((set) => (
            <ObservationSetCard
              key={set.id}
              set={set}
              projectId={projectId}
              onEdit={onEdit}
            />
          ))}
        </div>
      )}
    </div>
  )
}
