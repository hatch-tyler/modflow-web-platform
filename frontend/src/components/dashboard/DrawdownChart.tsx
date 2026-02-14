import { memo, useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { Loader2, Play, Pause, SkipForward, SkipBack } from 'lucide-react'
import { resultsApi } from '../../services/api'
import type { ResultsSummary, PostProcessProgress } from '../../types'
import ExportButton from './ExportButton'
import AwaitingData from './AwaitingData'
import { lengthAbbrev, labelWithUnit } from '../../utils/unitLabels'

const NUM_COLOR_BINS = 25

// RdBu diverging colorscale for drawdown (blue=positive/decline, red=negative/rise)
const DRAWDOWN_COLORSCALE: [number, string][] = [
  [0, '#2166ac'],
  [0.25, '#67a9cf'],
  [0.5, '#f7f7f7'],
  [0.75, '#ef8a62'],
  [1, '#b2182b'],
]

interface DrawdownChartProps {
  projectId: string
  runId: string
  summary: ResultsSummary
  expanded?: boolean
  awaitingProgress?: PostProcessProgress
}

function DrawdownChart({ projectId, runId, summary, expanded = false, awaitingProgress }: DrawdownChartProps) {
  // Show awaiting data state if progress is provided and heads not yet complete
  if (awaitingProgress && !awaitingProgress.postprocess_completed?.includes('heads')) {
    return <AwaitingData dataType="drawdown" progress={awaitingProgress} height={expanded ? window.innerHeight - 200 : 400} />
  }
  const { metadata, heads_summary } = summary
  const kstpkperList = heads_summary.kstpkper_list || []
  const gridType = metadata.grid_type || 'structured'
  const isPolygonGrid = gridType === 'unstructured' || gridType === 'vertex'
  const lu = lengthAbbrev(metadata.length_unit)
  const ddLabel = labelWithUnit('Drawdown', lu)

  const [layer, setLayer] = useState(0)
  const [selectedIdx, setSelectedIdx] = useState(kstpkperList.length > 1 ? kstpkperList.length - 1 : 0)
  const [playing, setPlaying] = useState(false)
  const [intervalMs, setIntervalMs] = useState(500)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const kstp = kstpkperList[selectedIdx]?.[0] ?? 0
  const kper = kstpkperList[selectedIdx]?.[1] ?? 0
  const initialKstp = kstpkperList[0]?.[0] ?? 0
  const initialKper = kstpkperList[0]?.[1] ?? 0

  // Fetch initial head slice (first time step)
  const { data: initialSlice, isLoading: initialLoading } = useQuery({
    queryKey: ['heads', projectId, runId, layer, initialKper, initialKstp],
    queryFn: () => resultsApi.getHeads(projectId, runId, layer, initialKper, initialKstp),
    enabled: kstpkperList.length > 0,
    staleTime: Infinity,
  })

  // Fetch current head slice
  const { data: currentSlice, isLoading: currentLoading } = useQuery({
    queryKey: ['heads', projectId, runId, layer, kper, kstp],
    queryFn: () => resultsApi.getHeads(projectId, runId, layer, kper, kstp),
    enabled: kstpkperList.length > 0 && selectedIdx > 0,
  })

  // Fetch grid geometry for polygon grids
  const { data: gridGeometry, isLoading: geomLoading } = useQuery({
    queryKey: ['grid-geometry', projectId, runId],
    queryFn: () => resultsApi.getGridGeometry(projectId, runId),
    enabled: isPolygonGrid,
    staleTime: Infinity,
  })

  // Compute drawdown data: initial_head - current_head (positive = decline)
  const drawdownData = useMemo(() => {
    if (selectedIdx === 0 && initialSlice) {
      // At initial time step, drawdown is zero
      return initialSlice.data.map(row =>
        row.map(v => (v === null || Math.abs(v!) > 1e20 || v! < -880 ? null : 0))
      )
    }
    if (!initialSlice || !currentSlice) return null

    const data: (number | null)[][] = []
    for (let r = 0; r < initialSlice.data.length; r++) {
      const row: (number | null)[] = []
      for (let c = 0; c < (initialSlice.data[r]?.length || 0); c++) {
        const initVal = initialSlice.data[r][c]
        const curVal = currentSlice.data[r]?.[c]
        if (
          initVal === null || curVal === null ||
          Math.abs(initVal!) > 1e20 || Math.abs(curVal!) > 1e20 ||
          initVal! < -880 || curVal! < -880
        ) {
          row.push(null)
        } else {
          row.push(initVal! - curVal!)
        }
      }
      data.push(row)
    }
    return data
  }, [initialSlice, currentSlice, selectedIdx])

  // Compute drawdown range for colorbar
  const drawdownRange = useMemo(() => {
    if (!drawdownData) return { min: -1, max: 1 }
    let mn = Infinity
    let mx = -Infinity
    for (const row of drawdownData) {
      for (const v of row) {
        if (v !== null) {
          if (v < mn) mn = v
          if (v > mx) mx = v
        }
      }
    }
    if (!isFinite(mn)) mn = -1
    if (!isFinite(mx)) mx = 1
    // Make symmetric around zero
    const absMax = Math.max(Math.abs(mn), Math.abs(mx), 0.01)
    return { min: -absMax, max: absMax }
  }, [drawdownData])

  // Polygon grid traces for drawdown
  const polygonTraces = useMemo(() => {
    if (!isPolygonGrid || !gridGeometry || !drawdownData) return null

    const layerKey = String(layer)
    const layerData = gridGeometry.layers[layerKey] || gridGeometry.layers['0']
    if (!layerData) return null

    const polygons = layerData.polygons

    // Flatten drawdown data
    const values: (number | null)[] = []
    for (const row of drawdownData) {
      if (Array.isArray(row)) {
        values.push(...row)
      } else {
        values.push(row)
      }
    }

    const { min: vmin, max: vmax } = drawdownRange
    const range = vmax - vmin || 1

    const binAssignments: number[] = []
    for (let i = 0; i < polygons.length; i++) {
      const v = i < values.length ? values[i] : null
      if (v === null) {
        binAssignments.push(-1)
      } else {
        const t = (v - vmin) / range
        binAssignments.push(Math.max(0, Math.min(NUM_COLOR_BINS - 1, Math.floor(t * NUM_COLOR_BINS))))
      }
    }

    // Interpolate RdBu colorscale
    function drawdownColor(t: number): string {
      const clamped = Math.max(0, Math.min(1, t))
      for (let i = 0; i < DRAWDOWN_COLORSCALE.length - 1; i++) {
        const [t0, c0] = DRAWDOWN_COLORSCALE[i]
        const [t1, c1] = DRAWDOWN_COLORSCALE[i + 1]
        if (clamped >= t0 && clamped <= t1) {
          const f = (clamped - t0) / (t1 - t0)
          // Simple hex interpolation
          const r0 = parseInt(c0.slice(1, 3), 16)
          const g0 = parseInt(c0.slice(3, 5), 16)
          const b0 = parseInt(c0.slice(5, 7), 16)
          const r1 = parseInt(c1.slice(1, 3), 16)
          const g1 = parseInt(c1.slice(3, 5), 16)
          const b1 = parseInt(c1.slice(5, 7), 16)
          const r = Math.round(r0 + (r1 - r0) * f)
          const g = Math.round(g0 + (g1 - g0) * f)
          const b = Math.round(b0 + (b1 - b0) * f)
          return `rgb(${r},${g},${b})`
        }
      }
      return DRAWDOWN_COLORSCALE[DRAWDOWN_COLORSCALE.length - 1][1]
    }

    const traces: Plotly.Data[] = []
    for (let bin = 0; bin < NUM_COLOR_BINS; bin++) {
      const xs: (number | null)[] = []
      const ys: (number | null)[] = []
      for (let i = 0; i < polygons.length; i++) {
        if (binAssignments[i] !== bin) continue
        const poly = polygons[i]
        if (!poly || poly.length < 3) continue
        for (const pt of poly) {
          xs.push(pt[0])
          ys.push(pt[1])
        }
        xs.push(null)
        ys.push(null)
      }
      if (xs.length === 0) continue
      const color = drawdownColor(bin / (NUM_COLOR_BINS - 1))
      traces.push({
        type: 'scatter',
        x: xs,
        y: ys,
        fill: 'toself',
        fillcolor: color,
        line: { color, width: 0.3 },
        hoverinfo: 'skip',
        showlegend: false,
      } as Plotly.Data)
    }

    // Hover points at centroids
    const cx: number[] = []
    const cy: number[] = []
    const hoverText: string[] = []
    for (let i = 0; i < polygons.length; i++) {
      const v = i < values.length ? values[i] : null
      if (v === null) continue
      const poly = polygons[i]
      if (!poly || poly.length < 3) continue
      let sumX = 0, sumY = 0, n = 0
      for (const pt of poly) {
        sumX += pt[0]
        sumY += pt[1]
        n++
      }
      if (n > 1 && poly[0][0] === poly[n - 1][0] && poly[0][1] === poly[n - 1][1]) {
        sumX -= poly[n - 1][0]
        sumY -= poly[n - 1][1]
        n--
      }
      cx.push(sumX / n)
      cy.push(sumY / n)
      hoverText.push(`Drawdown: ${v.toFixed(3)}${lu ? ' ' + lu : ''}<br>Cell: ${i + 1}`)
    }

    traces.push({
      type: 'scattergl',
      x: cx,
      y: cy,
      mode: 'markers',
      marker: { size: 6, color: 'rgba(0,0,0,0)' },
      text: hoverText,
      hoverinfo: 'text',
      showlegend: false,
    } as Plotly.Data)

    // Colorbar dummy trace
    traces.push({
      type: 'scatter',
      x: [null],
      y: [null],
      mode: 'markers',
      marker: {
        size: 0,
        color: [vmin],
        colorscale: DRAWDOWN_COLORSCALE.map(([t, c]) => [t, c]),
        cmin: vmin,
        cmax: vmax,
        colorbar: {
          title: { text: ddLabel, side: 'right' },
          thickness: 15,
          len: 0.9,
        },
        showscale: true,
      },
      showlegend: false,
      hoverinfo: 'skip',
    } as Plotly.Data)

    return traces
  }, [isPolygonGrid, gridGeometry, drawdownData, drawdownRange, layer])

  const stepForward = useCallback(() => {
    setSelectedIdx(prev => (prev + 1) % kstpkperList.length)
  }, [kstpkperList.length])

  const stepBack = useCallback(() => {
    setSelectedIdx(prev => (prev - 1 + kstpkperList.length) % kstpkperList.length)
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

  useEffect(() => {
    setPlaying(false)
    setSelectedIdx(kstpkperList.length > 1 ? kstpkperList.length - 1 : 0)
  }, [runId, layer, kstpkperList.length])

  if (kstpkperList.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-400">
        No head data available
      </div>
    )
  }

  if (kstpkperList.length < 2) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-400">
        Drawdown requires at least 2 time steps
      </div>
    )
  }

  const chartHeight = expanded ? window.innerHeight - 200 : 500
  const isLoading = initialLoading || (selectedIdx > 0 && currentLoading) || (isPolygonGrid && geomLoading)

  return (
    <div>
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
                <option key={i} value={i}>L{i + 1}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-slate-500">Step:</label>
            <select
              value={selectedIdx}
              onChange={(e) => { setPlaying(false); setSelectedIdx(Number(e.target.value)) }}
              className="text-xs border border-slate-300 rounded px-1.5 py-0.5"
            >
              {kstpkperList.map(([ks, kp], i) => (
                <option key={i} value={i}>SP{kp + 1} TS{ks + 1}</option>
              ))}
            </select>
          </div>
          <span className="text-xs text-slate-400">
            Drawdown = Initial Head - Current Head
          </span>
          <ExportButton url={resultsApi.exportDrawdownCsvUrl(projectId, runId, layer, kper, kstp)} />
        </div>
      </div>

      {/* Animation controls */}
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
          {playing
            ? <Pause className="h-4 w-4 text-slate-600" />
            : <Play className="h-4 w-4 text-slate-600" />}
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

      {isLoading ? (
        <div className="h-72 flex items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
        </div>
      ) : isPolygonGrid && polygonTraces ? (
        <Plot
          data={polygonTraces}
          layout={{
            autosize: true,
            height: chartHeight,
            margin: { l: 50, r: 20, t: 10, b: 50 },
            plot_bgcolor: '#ffffff',
            xaxis: {
              title: { text: 'X' },
              constrain: 'domain',
              scaleanchor: 'y',
              range: gridGeometry ? [gridGeometry.extent[0], gridGeometry.extent[1]] : undefined,
            },
            yaxis: {
              title: { text: 'Y' },
              constrain: 'domain',
              range: gridGeometry ? [gridGeometry.extent[2], gridGeometry.extent[3]] : undefined,
            },
            hovermode: 'closest',
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      ) : drawdownData ? (
        <Plot
          data={[
            {
              z: drawdownData,
              type: 'heatmap',
              colorscale: DRAWDOWN_COLORSCALE as any,
              colorbar: { title: { text: ddLabel, side: 'right' } },
              hoverongaps: false,
              xgap: 0,
              ygap: 0,
              zmin: drawdownRange.min,
              zmax: drawdownRange.max,
              zmid: 0,
            } as any,
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

export default memo(DrawdownChart)
