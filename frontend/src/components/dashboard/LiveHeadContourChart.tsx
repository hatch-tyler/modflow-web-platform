import { useState, useEffect, useRef, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { Loader2, Play, Pause, SkipForward, SkipBack, Radio } from 'lucide-react'
import { resultsApi } from '../../services/api'
import type { LiveResultsSummary } from '../../types'
import { lengthAbbrev, labelWithUnit } from '../../utils/unitLabels'

interface LiveHeadContourChartProps {
  projectId: string
  runId: string
  summary: LiveResultsSummary
  expanded?: boolean
}

export default function LiveHeadContourChart({
  projectId,
  runId,
  summary,
  expanded = false,
}: LiveHeadContourChartProps) {
  const { metadata, heads_summary } = summary
  const kstpkperList = heads_summary.kstpkper_list || []
  const headLabel = labelWithUnit('Head', lengthAbbrev(metadata.length_unit))

  const [layer, setLayer] = useState(0)
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [intervalMs, setIntervalMs] = useState(500)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const kstp = kstpkperList[selectedIdx]?.[0] ?? 0
  const kper = kstpkperList[selectedIdx]?.[1] ?? 0

  // Fetch live head slice data
  const { data: headSlice, isLoading: sliceLoading } = useQuery({
    queryKey: ['live-heads', projectId, runId, layer, kper, kstp],
    queryFn: () => resultsApi.getLiveHeads(projectId, runId, layer, kper, kstp),
    enabled: kstpkperList.length > 0,
    staleTime: 5000, // Cache for 5 seconds
  })

  // Auto-select latest timestep when new ones become available
  useEffect(() => {
    if (kstpkperList.length > 0 && !playing) {
      setSelectedIdx(kstpkperList.length - 1)
    }
  }, [kstpkperList.length, playing])

  const stepForward = useCallback(() => {
    setSelectedIdx((prev) => (prev + 1) % kstpkperList.length)
  }, [kstpkperList.length])

  const stepBack = useCallback(() => {
    setSelectedIdx((prev) => (prev - 1 + kstpkperList.length) % kstpkperList.length)
  }, [kstpkperList.length])

  // Animation loop
  useEffect(() => {
    if (playing && kstpkperList.length > 1) {
      timerRef.current = setInterval(stepForward, intervalMs)
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [playing, intervalMs, stepForward, kstpkperList.length])

  // Stop playing when run changes
  useEffect(() => {
    setPlaying(false)
    setSelectedIdx(Math.max(0, kstpkperList.length - 1))
  }, [runId])

  if (kstpkperList.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-400">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Waiting for first timestep to complete...</p>
        </div>
      </div>
    )
  }

  const canAnimate = kstpkperList.length > 1
  const chartHeight = expanded ? window.innerHeight - 200 : 400

  return (
    <div>
      {/* Live indicator */}
      <div className="flex items-center gap-2 mb-2 text-green-600 text-xs">
        <Radio className="h-3 w-3 animate-pulse" />
        <span>Live - {kstpkperList.length} timestep(s) available</span>
      </div>

      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-slate-500">Layer:</label>
            <select
              value={layer}
              onChange={(e) => setLayer(Number(e.target.value))}
              className="text-xs border border-slate-300 rounded px-1.5 py-0.5"
            >
              {Array.from({ length: metadata.nlay }, (_, i) => (
                <option key={i} value={i}>
                  L{i + 1}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-slate-500">Step:</label>
            <select
              value={selectedIdx}
              onChange={(e) => {
                setPlaying(false)
                setSelectedIdx(Number(e.target.value))
              }}
              className="text-xs border border-slate-300 rounded px-1.5 py-0.5"
            >
              {kstpkperList.map(([ks, kp], i) => (
                <option key={i} value={i}>
                  SP{kp + 1} TS{ks + 1}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {canAnimate && (
        <div className="flex items-center gap-2 mb-2">
          <button
            onClick={stepBack}
            disabled={playing}
            className="p-1 rounded hover:bg-slate-100 disabled:opacity-30"
            title="Previous step"
          >
            <SkipBack className="h-4 w-4 text-slate-600" />
          </button>
          <button
            onClick={() => setPlaying(!playing)}
            className="p-1 rounded hover:bg-slate-100"
            title={playing ? 'Pause' : 'Play'}
          >
            {playing ? (
              <Pause className="h-4 w-4 text-slate-600" />
            ) : (
              <Play className="h-4 w-4 text-slate-600" />
            )}
          </button>
          <button
            onClick={stepForward}
            disabled={playing}
            className="p-1 rounded hover:bg-slate-100 disabled:opacity-30"
            title="Next step"
          >
            <SkipForward className="h-4 w-4 text-slate-600" />
          </button>
          <label className="text-xs text-slate-500 ml-2">Speed:</label>
          <input
            type="range"
            min={100}
            max={2000}
            step={100}
            value={2100 - intervalMs}
            onChange={(e) => setIntervalMs(2100 - Number(e.target.value))}
            className="w-20 h-1 accent-blue-500"
            title={`${intervalMs}ms per frame`}
          />
          <span className="text-xs text-slate-400 w-10">
            {selectedIdx + 1}/{kstpkperList.length}
          </span>
        </div>
      )}

      {sliceLoading ? (
        <div className="h-72 flex items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
        </div>
      ) : headSlice ? (
        <Plot
          data={[
            {
              z: headSlice.data,
              type: 'heatmap',
              colorscale: 'Viridis',
              colorbar: { title: { text: headLabel, side: 'right' } },
              hoverongaps: false,
              xgap: 0,
              ygap: 0,
              zmin: heads_summary.min_head ?? undefined,
              zmax: heads_summary.max_head ?? undefined,
            },
          ]}
          layout={{
            autosize: true,
            height: chartHeight,
            margin: { l: 50, r: 20, t: 10, b: 50 },
            plot_bgcolor: '#e2e8f0',
            xaxis: {
              title: { text: 'Column' },
              constrain: 'domain',
            },
            yaxis: {
              title: { text: 'Row' },
              autorange: 'reversed',
              scaleanchor: 'x',
              constrain: 'domain',
            },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      ) : (
        <div className="h-72 flex items-center justify-center text-slate-400">
          No data for this slice
        </div>
      )}
    </div>
  )
}
