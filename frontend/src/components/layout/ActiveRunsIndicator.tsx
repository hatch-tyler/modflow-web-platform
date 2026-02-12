/**
 * Active Runs Indicator
 *
 * Shows a dropdown with all currently active runs across all projects.
 * Allows quick navigation to any running simulation.
 */

import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, Terminal, FlaskConical, ChevronDown, X } from 'lucide-react'
import clsx from 'clsx'
import { useRunManager, useActiveRunCount, type ActiveRun } from '../../store/runManager'

function RunTypeIcon({ runType }: { runType: ActiveRun['runType'] }) {
  switch (runType) {
    case 'simulation':
      return <Terminal className="h-4 w-4" />
    case 'pest_glm':
    case 'pest_ies':
      return <FlaskConical className="h-4 w-4" />
    default:
      return <Activity className="h-4 w-4" />
  }
}

function getStatusColor(status: ActiveRun['status']) {
  switch (status) {
    case 'running':
      return 'text-blue-500'
    case 'completed':
      return 'text-green-500'
    case 'failed':
      return 'text-red-500'
    case 'cancelled':
      return 'text-yellow-500'
    default:
      return 'text-slate-500'
  }
}

export default function ActiveRunsIndicator() {
  const navigate = useNavigate()
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const activeRunCount = useActiveRunCount()
  const { getAllActiveRuns, selectRun, stopRun, clearCompletedRuns } = useRunManager()
  const allRuns = getAllActiveRuns()

  // Sort runs: running first, then by start time
  const sortedRuns = [...allRuns].sort((a, b) => {
    if (a.status === 'running' && b.status !== 'running') return -1
    if (b.status === 'running' && a.status !== 'running') return 1
    return b.startedAt.getTime() - a.startedAt.getTime()
  })

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleNavigateToRun = (run: ActiveRun) => {
    selectRun(run.runId)
    if (run.runType === 'simulation') {
      navigate(`/projects/${run.projectId}/console`)
    } else {
      navigate(`/projects/${run.projectId}/pest`)
    }
    setIsOpen(false)
  }

  const handleRemoveRun = (e: React.MouseEvent, runId: string) => {
    e.stopPropagation()
    stopRun(runId)
  }

  if (allRuns.length === 0) {
    return null
  }

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Trigger button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={clsx(
          'flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors',
          activeRunCount > 0
            ? 'bg-blue-50 hover:bg-blue-100 text-blue-700'
            : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
        )}
      >
        <Activity
          className={clsx(
            'h-4 w-4',
            activeRunCount > 0 && 'animate-pulse'
          )}
        />
        <span className="text-sm font-medium">
          {activeRunCount > 0 ? (
            <>
              {activeRunCount} Running
            </>
          ) : (
            <>
              {allRuns.length} Recent
            </>
          )}
        </span>
        <ChevronDown className={clsx(
          'h-4 w-4 transition-transform',
          isOpen && 'rotate-180'
        )} />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border border-slate-200 z-50">
          <div className="p-3 border-b border-slate-100">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-slate-800">Active Runs</h3>
              {allRuns.some((r) => r.status !== 'running') && (
                <button
                  onClick={() => clearCompletedRuns()}
                  className="text-xs text-slate-500 hover:text-slate-700"
                >
                  Clear completed
                </button>
              )}
            </div>
          </div>

          <div className="max-h-96 overflow-y-auto">
            {sortedRuns.map((run) => (
              <div
                key={run.runId}
                onClick={() => handleNavigateToRun(run)}
                className="p-3 hover:bg-slate-50 cursor-pointer border-b border-slate-50 last:border-b-0"
              >
                <div className="flex items-start gap-3">
                  <div className={clsx('mt-0.5', getStatusColor(run.status))}>
                    <RunTypeIcon runType={run.runType} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm text-slate-800 truncate">
                        {run.projectName}
                      </span>
                      <span className={clsx(
                        'px-1.5 py-0.5 text-xs rounded',
                        run.status === 'running'
                          ? 'bg-blue-100 text-blue-700'
                          : run.status === 'completed'
                          ? 'bg-green-100 text-green-700'
                          : run.status === 'failed'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-slate-100 text-slate-600'
                      )}>
                        {run.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-slate-500 mt-0.5">
                      <span className="font-mono">{run.runId.slice(0, 8)}</span>
                      <span>|</span>
                      <span>
                        {run.runType === 'simulation' ? 'Simulation' :
                         run.runType === 'pest_glm' ? 'PEST++ GLM' : 'PEST++ IES'}
                      </span>
                    </div>
                    <div className="text-xs text-slate-400 mt-1">
                      Started {run.startedAt.toLocaleTimeString()}
                    </div>
                  </div>
                  {run.status !== 'running' && (
                    <button
                      onClick={(e) => handleRemoveRun(e, run.runId)}
                      className="p-1 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>

          {allRuns.length === 0 && (
            <div className="p-6 text-center text-slate-500 text-sm">
              No active runs
            </div>
          )}
        </div>
      )}
    </div>
  )
}
