import { useState, useRef, useEffect, useCallback } from 'react'

interface UseTimestepAnimationOptions {
  /** Total number of timesteps available */
  totalSteps: number
  /** Initial index (0 = first, -1 = last). Default: 0 */
  initialIndex?: number | 'first' | 'last'
  /** Default playback interval in ms. Default: 500 */
  defaultIntervalMs?: number
  /** Dependencies that should reset playback (e.g. [runId, layer]) */
  resetDeps?: readonly unknown[]
  /** If true, auto-advance to latest timestep when new ones arrive (for live mode) */
  followLatest?: boolean
}

interface TimestepAnimationState {
  selectedIdx: number
  setSelectedIdx: React.Dispatch<React.SetStateAction<number>>
  playing: boolean
  setPlaying: React.Dispatch<React.SetStateAction<boolean>>
  intervalMs: number
  setIntervalMs: React.Dispatch<React.SetStateAction<number>>
  stepForward: () => void
  stepBack: () => void
}

/**
 * Shared hook for timestep animation across contour/drawdown/live charts.
 *
 * Handles:
 * - Play/pause with setInterval
 * - Proper cleanup on unmount or dependency change
 * - Step forward/back with wrapping
 * - Reset on run/layer change
 * - Optional "follow latest" for live mode
 */
export function useTimestepAnimation({
  totalSteps,
  initialIndex = 'first',
  defaultIntervalMs = 500,
  resetDeps = [],
  followLatest = false,
}: UseTimestepAnimationOptions): TimestepAnimationState {
  const resolveInitial = () => {
    if (initialIndex === 'last') return Math.max(0, totalSteps - 1)
    if (initialIndex === 'first') return 0
    return Math.max(0, Math.min(initialIndex, totalSteps - 1))
  }

  const [selectedIdx, setSelectedIdx] = useState(resolveInitial)
  const [playing, setPlaying] = useState(false)
  const [intervalMs, setIntervalMs] = useState(defaultIntervalMs)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stepForward = useCallback(() => {
    if (totalSteps <= 0) return
    setSelectedIdx(prev => (prev + 1) % totalSteps)
  }, [totalSteps])

  const stepBack = useCallback(() => {
    if (totalSteps <= 0) return
    setSelectedIdx(prev => (prev - 1 + totalSteps) % totalSteps)
  }, [totalSteps])

  // Animation interval — always cleaned up
  useEffect(() => {
    if (playing && totalSteps > 1) {
      timerRef.current = setInterval(stepForward, intervalMs)
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [playing, intervalMs, stepForward, totalSteps])

  // Reset on dependency change (run, layer, etc.)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    setPlaying(false)
    setSelectedIdx(resolveInitial())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, resetDeps)

  // Follow latest timestep in live mode (only when not playing)
  useEffect(() => {
    if (followLatest && !playing && totalSteps > 0) {
      setSelectedIdx(totalSteps - 1)
    }
  }, [followLatest, playing, totalSteps])

  return {
    selectedIdx,
    setSelectedIdx,
    playing,
    setPlaying,
    intervalMs,
    setIntervalMs,
    stepForward,
    stepBack,
  }
}
