/**
 * Global Run Manager Store
 *
 * Manages active simulation runs across all projects with:
 * - Persistent SSE connections that survive navigation
 * - Console output buffers for each run
 * - Quick access to any active run from anywhere in the app
 */

import { create } from 'zustand'
import { createOutputEventSource, createPestOutputEventSource } from '../services/api'

export type RunType = 'simulation' | 'pest_glm' | 'pest_ies'

export interface ActiveRun {
  runId: string
  projectId: string
  projectName: string
  runType: RunType
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  startedAt: Date
  output: string[]
  eventSource: EventSource | null
  reconnectAttempts: number
  reconnectTimer: ReturnType<typeof setTimeout> | null
}

interface RunManagerState {
  // Active runs indexed by runId
  activeRuns: Record<string, ActiveRun>

  // Currently selected run (for viewing in console)
  selectedRunId: string | null

  // Actions
  startRun: (params: {
    runId: string
    projectId: string
    projectName: string
    runType: RunType
    startedAt?: Date
    isReconnect?: boolean
  }) => void

  stopRun: (runId: string) => void

  selectRun: (runId: string | null) => void

  appendOutput: (runId: string, line: string) => void

  updateStatus: (runId: string, status: ActiveRun['status']) => void

  getRunOutput: (runId: string) => string[]

  getActiveRunsForProject: (projectId: string) => ActiveRun[]

  getAllActiveRuns: () => ActiveRun[]

  clearCompletedRuns: () => void
}

// Maximum output lines to keep per run (prevent memory issues)
const MAX_OUTPUT_LINES = 10000

// SSE reconnection settings
const MAX_RECONNECT_ATTEMPTS = 10
const INITIAL_RECONNECT_DELAY_MS = 1000
const MAX_RECONNECT_DELAY_MS = 30000

export const useRunManager = create<RunManagerState>((set, get) => ({
  activeRuns: {},
  selectedRunId: null,

  startRun: ({ runId, projectId, projectName, runType, startedAt, isReconnect }) => {
    const state = get()

    // If already tracking this run, check if we need to re-establish the SSE connection
    const existingRun = state.activeRuns[runId]
    if (existingRun) {
      // Only reconnect if the run is still "running" but SSE is dead (no eventSource, no pending timer)
      if (existingRun.status === 'running' && !existingRun.eventSource && !existingRun.reconnectTimer) {
        // Fall through to reconnect below
      } else {
        return
      }
    }

    // Helper to create SSE connection with event handlers
    const connectSSE = () => {
      const eventSource = runType === 'simulation'
        ? createOutputEventSource(projectId, runId)
        : createPestOutputEventSource(projectId, runId)

      eventSource.onmessage = (event) => {
        // Reset reconnect counter on successful message
        const currentRun = get().activeRuns[runId]
        if (currentRun && currentRun.reconnectAttempts > 0) {
          set((s) => ({
            activeRuns: {
              ...s.activeRuns,
              [runId]: { ...s.activeRuns[runId], reconnectAttempts: 0 },
            },
          }))
        }
        get().appendOutput(runId, event.data)
      }

      eventSource.addEventListener('status', (event) => {
        const status = (event as MessageEvent).data as string
        if (status === 'completed' || status === 'failed' || status === 'cancelled') {
          get().updateStatus(runId, status as ActiveRun['status'])
          eventSource.close()
        }
      })

      eventSource.onerror = () => {
        eventSource.close()
        const currentRun = get().activeRuns[runId]
        if (!currentRun || currentRun.status !== 'running') {
          return // Don't reconnect if run already finished
        }

        const attempts = currentRun.reconnectAttempts + 1
        if (attempts > MAX_RECONNECT_ATTEMPTS) {
          get().appendOutput(runId, '')
          get().appendOutput(runId, `[Connection lost after ${MAX_RECONNECT_ATTEMPTS} reconnect attempts. Refresh the page to retry.]`)
          set((s) => ({
            activeRuns: {
              ...s.activeRuns,
              [runId]: { ...s.activeRuns[runId], eventSource: null, reconnectAttempts: attempts },
            },
          }))
          return
        }

        // Exponential backoff: 1s, 2s, 4s, 8s, ... capped at 30s
        const delay = Math.min(
          INITIAL_RECONNECT_DELAY_MS * Math.pow(2, attempts - 1),
          MAX_RECONNECT_DELAY_MS,
        )

        get().appendOutput(runId, `[Connection lost. Reconnecting in ${(delay / 1000).toFixed(0)}s... (attempt ${attempts}/${MAX_RECONNECT_ATTEMPTS})]`)

        const timer = setTimeout(() => {
          const run = get().activeRuns[runId]
          if (run && run.status === 'running') {
            const newES = connectSSE()
            set((s) => ({
              activeRuns: {
                ...s.activeRuns,
                [runId]: {
                  ...s.activeRuns[runId],
                  eventSource: newES,
                  reconnectAttempts: attempts,
                  reconnectTimer: null,
                },
              },
            }))
          }
        }, delay)

        set((s) => ({
          activeRuns: {
            ...s.activeRuns,
            [runId]: {
              ...s.activeRuns[runId],
              eventSource: null,
              reconnectAttempts: attempts,
              reconnectTimer: timer,
            },
          },
        }))
      }

      return eventSource
    }

    const eventSource = connectSSE()

    // Re-establish SSE on an existing stale run
    if (existingRun) {
      get().appendOutput(runId, `[${new Date().toLocaleTimeString()}] Reconnected to simulation`)
      set((s) => ({
        activeRuns: {
          ...s.activeRuns,
          [runId]: {
            ...s.activeRuns[runId],
            eventSource,
            reconnectAttempts: 0,
            reconnectTimer: null,
          },
        },
        selectedRunId: runId,
      }))
      return
    }

    // Create new active run
    const actualStart = startedAt || new Date()

    // When reconnecting, start with empty output — the SSE endpoint will replay
    // full history. For new runs, show a "Run started" message.
    const initialOutput = isReconnect
      ? [`[${new Date().toLocaleTimeString()}] Reconnected to simulation (started ${actualStart.toLocaleTimeString()})`, '']
      : [`[${new Date().toLocaleTimeString()}] Run started...`, '']

    const newRun: ActiveRun = {
      runId,
      projectId,
      projectName,
      runType,
      status: 'running',
      startedAt: actualStart,
      output: initialOutput,
      eventSource,
      reconnectAttempts: 0,
      reconnectTimer: null,
    }

    set((state) => ({
      activeRuns: {
        ...state.activeRuns,
        [runId]: newRun,
      },
      selectedRunId: runId,
    }))
  },

  stopRun: (runId) => {
    const run = get().activeRuns[runId]
    if (run?.eventSource) {
      run.eventSource.close()
    }
    if (run?.reconnectTimer) {
      clearTimeout(run.reconnectTimer)
    }

    set((state) => {
      const { [runId]: removed, ...rest } = state.activeRuns
      return {
        activeRuns: rest,
        selectedRunId: state.selectedRunId === runId ? null : state.selectedRunId,
      }
    })
  },

  selectRun: (runId) => {
    set({ selectedRunId: runId })
  },

  appendOutput: (runId, line) => {
    set((state) => {
      const run = state.activeRuns[runId]
      if (!run) return state

      // Limit output buffer size
      const newOutput = [...run.output, line]
      if (newOutput.length > MAX_OUTPUT_LINES) {
        newOutput.splice(0, newOutput.length - MAX_OUTPUT_LINES)
      }

      return {
        activeRuns: {
          ...state.activeRuns,
          [runId]: {
            ...run,
            output: newOutput,
          },
        },
      }
    })
  },

  updateStatus: (runId, status) => {
    set((state) => {
      const run = state.activeRuns[runId]
      if (!run) return state

      // Add status message to output
      const statusMessage = `[${new Date().toLocaleTimeString()}] Run ${status}`
      const newOutput = [...run.output, '', statusMessage]

      return {
        activeRuns: {
          ...state.activeRuns,
          [runId]: {
            ...run,
            status,
            output: newOutput,
            eventSource: null, // Connection is closed
          },
        },
      }
    })
  },

  getRunOutput: (runId) => {
    return get().activeRuns[runId]?.output || []
  },

  getActiveRunsForProject: (projectId) => {
    return Object.values(get().activeRuns).filter(
      (run) => run.projectId === projectId
    )
  },

  getAllActiveRuns: () => {
    return Object.values(get().activeRuns)
  },

  clearCompletedRuns: () => {
    set((state) => {
      const activeOnly: Record<string, ActiveRun> = {}
      for (const [id, run] of Object.entries(state.activeRuns)) {
        if (run.status === 'running') {
          activeOnly[id] = run
        } else {
          // Close any lingering connections and timers
          run.eventSource?.close()
          if (run.reconnectTimer) clearTimeout(run.reconnectTimer)
        }
      }
      return { activeRuns: activeOnly }
    })
  },
}))

/**
 * Hook to get the count of active (running) runs
 */
export function useActiveRunCount(): number {
  return useRunManager((state) =>
    Object.values(state.activeRuns).filter((r) => r.status === 'running').length
  )
}

/**
 * Hook to check if a specific project has running simulations
 */
export function useProjectHasActiveRuns(projectId: string): boolean {
  return useRunManager((state) =>
    Object.values(state.activeRuns).some(
      (r) => r.projectId === projectId && r.status === 'running'
    )
  )
}
