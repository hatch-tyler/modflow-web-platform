import { useState, useEffect } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { MapPin, Calendar, Save, Loader2, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'
import { projectsApi } from '../../services/api'
import type { Project, ProjectUpdate } from '../../types'

interface SpatialReferenceEditorProps {
  project: Project
}

export default function SpatialReferenceEditor({ project }: SpatialReferenceEditorProps) {
  const queryClient = useQueryClient()
  const [expanded, setExpanded] = useState(false)

  const [xoff, setXoff] = useState(project.xoff?.toString() ?? '')
  const [yoff, setYoff] = useState(project.yoff?.toString() ?? '')
  const [angrot, setAngrot] = useState(project.angrot?.toString() ?? '')
  const [epsg, setEpsg] = useState(project.epsg?.toString() ?? '')
  const [startDate, setStartDate] = useState(project.start_date ?? '')

  // Sync state when project changes
  useEffect(() => {
    setXoff(project.xoff?.toString() ?? '')
    setYoff(project.yoff?.toString() ?? '')
    setAngrot(project.angrot?.toString() ?? '')
    setEpsg(project.epsg?.toString() ?? '')
    setStartDate(project.start_date ?? '')
  }, [project.id, project.xoff, project.yoff, project.angrot, project.epsg, project.start_date])

  const saveMutation = useMutation({
    mutationFn: (data: ProjectUpdate) => projectsApi.update(project.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', project.id] })
    },
  })

  const revalidateMutation = useMutation({
    mutationFn: () => projectsApi.revalidate(project.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', project.id] })
    },
  })

  const handleSave = () => {
    const update: ProjectUpdate = {}
    const xoffNum = xoff ? parseFloat(xoff) : undefined
    const yoffNum = yoff ? parseFloat(yoff) : undefined
    const angrotNum = angrot ? parseFloat(angrot) : undefined
    const epsgNum = epsg ? parseInt(epsg, 10) : undefined

    if (xoffNum !== undefined && !isNaN(xoffNum)) update.xoff = xoffNum
    if (yoffNum !== undefined && !isNaN(yoffNum)) update.yoff = yoffNum
    if (angrotNum !== undefined && !isNaN(angrotNum)) update.angrot = angrotNum
    if (epsgNum !== undefined && !isNaN(epsgNum)) update.epsg = epsgNum
    if (startDate) {
      update.start_date = startDate
    } else {
      update.start_date = null
    }

    saveMutation.mutate(update)
  }

  const hasChanges =
    (xoff || '') !== (project.xoff?.toString() ?? '') ||
    (yoff || '') !== (project.yoff?.toString() ?? '') ||
    (angrot || '') !== (project.angrot?.toString() ?? '') ||
    (epsg || '') !== (project.epsg?.toString() ?? '') ||
    (startDate || '') !== (project.start_date ?? '')

  const spd = project.stress_period_data
  const totalTime = spd ? spd.reduce((s, p) => s + p.perlen, 0) : null
  const totalSteps = spd ? spd.reduce((s, p) => s + p.nstp, 0) : null

  return (
    <div className="mt-8">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-lg font-semibold text-slate-800 mb-4 hover:text-slate-600 transition-colors"
      >
        <MapPin className="h-5 w-5 text-blue-500" />
        Spatial Reference & Timing
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-slate-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-slate-400" />
        )}
      </button>

      {expanded && (
        <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-4">
          {/* Editable coordinate fields */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">X Offset</label>
              <input
                type="number"
                step="any"
                value={xoff}
                onChange={(e) => setXoff(e.target.value)}
                placeholder="e.g., 500000"
                className="w-full text-sm border border-slate-300 rounded-md px-3 py-1.5 focus:ring-1 focus:ring-blue-400 focus:border-blue-400"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">Y Offset</label>
              <input
                type="number"
                step="any"
                value={yoff}
                onChange={(e) => setYoff(e.target.value)}
                placeholder="e.g., 4000000"
                className="w-full text-sm border border-slate-300 rounded-md px-3 py-1.5 focus:ring-1 focus:ring-blue-400 focus:border-blue-400"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">Rotation (degrees)</label>
              <input
                type="number"
                step="any"
                value={angrot}
                onChange={(e) => setAngrot(e.target.value)}
                placeholder="0"
                className="w-full text-sm border border-slate-300 rounded-md px-3 py-1.5 focus:ring-1 focus:ring-blue-400 focus:border-blue-400"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">EPSG Code</label>
              <input
                type="number"
                value={epsg}
                onChange={(e) => setEpsg(e.target.value)}
                placeholder="e.g., 32615"
                className="w-full text-sm border border-slate-300 rounded-md px-3 py-1.5 focus:ring-1 focus:ring-blue-400 focus:border-blue-400"
              />
            </div>
          </div>

          {(project.grid_type === 'vertex' || project.grid_type === 'unstructured') && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-xs text-blue-700">
              <span className="font-medium">Note:</span> This model uses a{' '}
              {project.grid_type === 'vertex' ? 'vertex (DISV)' : 'unstructured (DISU)'} grid.
              Cell coordinates may already be in real-world (e.g., UTM) coordinates. If so,
              X/Y Offset values of 0 are correct. Only set offsets if the model uses local
              coordinates with a separate XORIGIN/YORIGIN.
            </div>
          )}

          {/* Start date */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-500 mb-1">
                <Calendar className="h-3.5 w-3.5 inline mr-1" />
                Start Date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full text-sm border border-slate-300 rounded-md px-3 py-1.5 focus:ring-1 focus:ring-blue-400 focus:border-blue-400"
              />
            </div>
            <div className="flex items-end gap-2">
              <button
                onClick={handleSave}
                disabled={!hasChanges || saveMutation.isPending}
                className="flex items-center gap-1.5 px-4 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {saveMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                Save
              </button>
              <button
                onClick={() => revalidateMutation.mutate()}
                disabled={revalidateMutation.isPending}
                title="Re-parse model files to refresh spatial reference and metadata"
                className="flex items-center gap-1.5 px-3 py-1.5 text-sm border border-slate-300 text-slate-700 rounded-md hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {revalidateMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4" />
                )}
                Revalidate
              </button>
              {saveMutation.isSuccess && (
                <span className="text-xs text-green-600">Saved</span>
              )}
              {saveMutation.isError && (
                <span className="text-xs text-red-600">Error saving</span>
              )}
              {revalidateMutation.isSuccess && (
                <span className="text-xs text-green-600">Revalidated</span>
              )}
              {revalidateMutation.isError && (
                <span className="text-xs text-red-600">Revalidation failed</span>
              )}
            </div>
          </div>

          {/* Read-only info */}
          <div className="border-t border-slate-100 pt-3">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-500">
              {project.length_unit && (
                <div>
                  <span className="font-medium text-slate-600">Length Unit:</span> {project.length_unit}
                </div>
              )}
              {project.time_unit && (
                <div>
                  <span className="font-medium text-slate-600">Time Unit:</span> {project.time_unit}
                </div>
              )}
              {spd && (
                <>
                  <div>
                    <span className="font-medium text-slate-600">Stress Periods:</span> {spd.length}
                  </div>
                  <div>
                    <span className="font-medium text-slate-600">Total Steps:</span> {totalSteps}
                  </div>
                  <div className="col-span-2">
                    <span className="font-medium text-slate-600">Total Time:</span>{' '}
                    {totalTime?.toLocaleString()} {project.time_unit || 'time units'}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
