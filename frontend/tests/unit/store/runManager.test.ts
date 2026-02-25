/**
 * Tests for runManager Zustand store
 *
 * Tests SSE connection lifecycle, reconnect logic, null guards, and cleanup.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import { act } from '@testing-library/react'
import { useRunManager } from '@/store/runManager'

// Mock EventSource
class MockEventSource {
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  readyState = 0
  url: string
  private listeners: Record<string, ((event: Event) => void)[]> = {}

  constructor(url: string) {
    this.url = url
    this.readyState = 1 // OPEN
  }

  addEventListener(type: string, handler: (event: Event) => void) {
    if (!this.listeners[type]) this.listeners[type] = []
    this.listeners[type].push(handler)
  }

  removeEventListener(type: string, handler: (event: Event) => void) {
    if (this.listeners[type]) {
      this.listeners[type] = this.listeners[type].filter((h) => h !== handler)
    }
  }

  dispatchEvent(event: Event): boolean {
    const handlers = this.listeners[event.type] || []
    handlers.forEach((h) => h(event))
    return true
  }

  close = vi.fn()

  // Test helpers
  simulateMessage(data: string) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data }))
    }
  }

  simulateStatus(status: string) {
    const handlers = this.listeners['status'] || []
    const event = new MessageEvent('status', { data: status })
    handlers.forEach((h) => h(event))
  }

  simulateError() {
    if (this.onerror) {
      this.onerror()
    }
  }
}

// Track created EventSources
let createdEventSources: MockEventSource[] = []

vi.mock('@/services/api', () => ({
  createOutputEventSource: vi.fn((_projectId: string, _runId: string) => {
    const es = new MockEventSource(`/api/v1/projects/${_projectId}/simulation/runs/${_runId}/output`)
    createdEventSources.push(es)
    return es
  }),
  createPestOutputEventSource: vi.fn((_projectId: string, _runId: string) => {
    const es = new MockEventSource(`/api/v1/projects/${_projectId}/pest/runs/${_runId}/output`)
    createdEventSources.push(es)
    return es
  }),
}))

const resetStore = () => {
  // Stop all runs and clear state
  const state = useRunManager.getState()
  Object.keys(state.activeRuns).forEach((runId) => {
    state.stopRun(runId)
  })
  useRunManager.setState({
    activeRuns: {},
    selectedRunId: null,
  })
}

describe('runManager', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    createdEventSources = []
    resetStore()
  })

  afterEach(() => {
    resetStore()
    vi.useRealTimers()
  })

  describe('startRun', () => {
    it('should create a new active run with SSE connection', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test Project',
          runType: 'simulation',
        })
      })

      const state = useRunManager.getState()
      expect(state.activeRuns['run-1']).toBeDefined()
      expect(state.activeRuns['run-1'].status).toBe('running')
      expect(state.activeRuns['run-1'].runType).toBe('simulation')
      expect(state.selectedRunId).toBe('run-1')
      expect(createdEventSources.length).toBe(1)
    })

    it('should not create duplicate run if already tracked', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test Project',
          runType: 'simulation',
        })
      })

      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test Project',
          runType: 'simulation',
        })
      })

      // Only one EventSource should be created
      expect(createdEventSources.length).toBe(1)
    })
  })

  describe('stopRun', () => {
    it('should close EventSource and remove run', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      const es = createdEventSources[0]

      act(() => {
        useRunManager.getState().stopRun('run-1')
      })

      expect(es.close).toHaveBeenCalled()
      expect(useRunManager.getState().activeRuns['run-1']).toBeUndefined()
    })

    it('should clear selectedRunId if stopping the selected run', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      expect(useRunManager.getState().selectedRunId).toBe('run-1')

      act(() => {
        useRunManager.getState().stopRun('run-1')
      })

      expect(useRunManager.getState().selectedRunId).toBeNull()
    })
  })

  describe('appendOutput', () => {
    it('should append lines to run output', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      act(() => {
        useRunManager.getState().appendOutput('run-1', 'Line 1')
        useRunManager.getState().appendOutput('run-1', 'Line 2')
      })

      const output = useRunManager.getState().getRunOutput('run-1')
      expect(output).toContain('Line 1')
      expect(output).toContain('Line 2')
    })

    it('should be a no-op for non-existent run', () => {
      act(() => {
        useRunManager.getState().appendOutput('nonexistent', 'Line')
      })

      const output = useRunManager.getState().getRunOutput('nonexistent')
      expect(output).toEqual([])
    })
  })

  describe('SSE message handling', () => {
    it('should append SSE messages to output', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      const es = createdEventSources[0]

      act(() => {
        es.simulateMessage('Simulation started')
        es.simulateMessage('Processing timestep 1')
      })

      const output = useRunManager.getState().getRunOutput('run-1')
      expect(output).toContain('Simulation started')
      expect(output).toContain('Processing timestep 1')
    })

    it('should update status on status event', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      const es = createdEventSources[0]

      act(() => {
        es.simulateStatus('completed')
      })

      const run = useRunManager.getState().activeRuns['run-1']
      expect(run.status).toBe('completed')
      expect(es.close).toHaveBeenCalled()
    })
  })

  describe('SSE reconnection (B7 fix)', () => {
    it('should attempt reconnect on error with backoff', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      const es = createdEventSources[0]

      // Simulate SSE error
      act(() => {
        es.simulateError()
      })

      // Run should still be tracked
      const run = useRunManager.getState().activeRuns['run-1']
      expect(run).toBeDefined()
      expect(run.status).toBe('running')
      expect(run.eventSource).toBeNull() // Closed after error
      expect(run.reconnectAttempts).toBe(1)
      expect(run.reconnectTimer).not.toBeNull()
    })

    it('should not spread undefined if run is stopped during reconnect timeout', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      const es = createdEventSources[0]

      // Trigger reconnect
      act(() => {
        es.simulateError()
      })

      // Stop the run while the reconnect timer is pending
      act(() => {
        useRunManager.getState().stopRun('run-1')
      })

      // Advance timers past the reconnect delay — this should NOT throw
      act(() => {
        vi.advanceTimersByTime(5000)
      })

      // Run should be gone, no crash
      expect(useRunManager.getState().activeRuns['run-1']).toBeUndefined()
    })

    it('should reconnect successfully after error', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      const es = createdEventSources[0]
      expect(createdEventSources.length).toBe(1)

      // Simulate error
      act(() => {
        es.simulateError()
      })

      // Advance past reconnect delay
      act(() => {
        vi.advanceTimersByTime(2000)
      })

      // A new EventSource should have been created
      expect(createdEventSources.length).toBe(2)

      const run = useRunManager.getState().activeRuns['run-1']
      expect(run.reconnectTimer).toBeNull() // Timer cleared after reconnect
    })
  })

  describe('updateStatus', () => {
    it('should update run status and nullify eventSource', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      act(() => {
        useRunManager.getState().updateStatus('run-1', 'failed')
      })

      const run = useRunManager.getState().activeRuns['run-1']
      expect(run.status).toBe('failed')
      expect(run.eventSource).toBeNull()
    })

    it('should be a no-op for non-existent run', () => {
      act(() => {
        useRunManager.getState().updateStatus('nonexistent', 'completed')
      })

      // Should not throw
      expect(useRunManager.getState().activeRuns['nonexistent']).toBeUndefined()
    })
  })

  describe('clearCompletedRuns', () => {
    it('should remove completed runs but keep running ones', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
        useRunManager.getState().startRun({
          runId: 'run-2',
          projectId: 'proj-1',
          projectName: 'Test',
          runType: 'simulation',
        })
      })

      // Mark run-1 as completed
      act(() => {
        useRunManager.getState().updateStatus('run-1', 'completed')
      })

      act(() => {
        useRunManager.getState().clearCompletedRuns()
      })

      expect(useRunManager.getState().activeRuns['run-1']).toBeUndefined()
      expect(useRunManager.getState().activeRuns['run-2']).toBeDefined()
    })
  })

  describe('helper hooks', () => {
    it('getActiveRunsForProject should filter by projectId', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test 1',
          runType: 'simulation',
        })
        useRunManager.getState().startRun({
          runId: 'run-2',
          projectId: 'proj-2',
          projectName: 'Test 2',
          runType: 'simulation',
        })
      })

      const proj1Runs = useRunManager.getState().getActiveRunsForProject('proj-1')
      expect(proj1Runs.length).toBe(1)
      expect(proj1Runs[0].runId).toBe('run-1')
    })

    it('getAllActiveRuns should return all runs', () => {
      act(() => {
        useRunManager.getState().startRun({
          runId: 'run-1',
          projectId: 'proj-1',
          projectName: 'Test 1',
          runType: 'simulation',
        })
        useRunManager.getState().startRun({
          runId: 'run-2',
          projectId: 'proj-2',
          projectName: 'Test 2',
          runType: 'simulation',
        })
      })

      const allRuns = useRunManager.getState().getAllActiveRuns()
      expect(allRuns.length).toBe(2)
    })
  })
})
