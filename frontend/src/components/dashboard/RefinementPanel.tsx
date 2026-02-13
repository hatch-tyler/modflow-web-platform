import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Loader2, CheckCircle2, AlertTriangle, Undo2, Wrench } from 'lucide-react'
import { convergenceApi } from '../../services/api'
import RefinementCard from './RefinementCard'

interface RefinementPanelProps {
  projectId: string
  runId: string
}

export default function RefinementPanel({ projectId, runId }: RefinementPanelProps) {
  const queryClient = useQueryClient()
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [lastBackup, setLastBackup] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['convergence-recommendations', projectId, runId],
    queryFn: () => convergenceApi.getRecommendations(projectId, runId),
    retry: false,
  })

  const applyMutation = useMutation({
    mutationFn: () =>
      convergenceApi.applyRefinements(projectId, runId, Array.from(selectedIds)),
    onSuccess: (result) => {
      setLastBackup(result.backup_timestamp)
      setSelectedIds(new Set())
      queryClient.invalidateQueries({ queryKey: ['convergence-recommendations', projectId, runId] })
    },
  })

  const revertMutation = useMutation({
    mutationFn: (timestamp: string) =>
      convergenceApi.revertRefinements(projectId, runId, timestamp),
    onSuccess: () => {
      setLastBackup(null)
    },
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <span className="ml-3 text-slate-500">Generating recommendations...</span>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-slate-400">
        <AlertTriangle className="h-8 w-8 mb-2" />
        <p>Could not generate recommendations.</p>
      </div>
    )
  }

  const recommendations = data.recommendations
  const toggleSelection = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const applicableCount = recommendations.filter(r => r.file_modification).length

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Wrench className="h-5 w-5 text-slate-500" />
          <h3 className="text-lg font-medium text-slate-800">
            Refinement Recommendations
          </h3>
          <span className="text-sm text-slate-400">({recommendations.length} suggestions)</span>
        </div>

        <div className="flex items-center gap-2">
          {lastBackup && (
            <button
              onClick={() => revertMutation.mutate(lastBackup)}
              disabled={revertMutation.isPending}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-amber-700 bg-amber-50 border border-amber-200 rounded-lg hover:bg-amber-100 disabled:opacity-50"
            >
              {revertMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Undo2 className="h-4 w-4" />
              )}
              Revert Changes
            </button>
          )}

          {applicableCount > 0 && (
            <button
              onClick={() => applyMutation.mutate()}
              disabled={selectedIds.size === 0 || applyMutation.isPending}
              className="flex items-center gap-1.5 px-4 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {applyMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <CheckCircle2 className="h-4 w-4" />
              )}
              Apply Selected ({selectedIds.size})
            </button>
          )}
        </div>
      </div>

      {/* Success / error messages */}
      {applyMutation.isSuccess && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-sm text-green-700">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4" />
            Refinements applied successfully. You can now re-run the simulation.
          </div>
        </div>
      )}
      {applyMutation.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
          Failed to apply refinements. {(applyMutation.error as Error)?.message}
        </div>
      )}
      {revertMutation.isSuccess && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-sm text-green-700">
          Changes reverted successfully.
        </div>
      )}

      {/* Recommendation cards */}
      {recommendations.length === 0 ? (
        <div className="text-center py-12 text-slate-400">
          <CheckCircle2 className="h-8 w-8 mx-auto mb-2" />
          <p>No refinements needed. Model convergence looks good.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {recommendations.map((rec) => (
            <RefinementCard
              key={rec.id}
              refinement={rec}
              selected={selectedIds.has(rec.id)}
              onToggle={() => toggleSelection(rec.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
