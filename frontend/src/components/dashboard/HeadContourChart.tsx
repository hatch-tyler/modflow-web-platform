import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { Loader2, Play, Pause, SkipForward, SkipBack, Map as MapIcon, Grid3X3 } from 'lucide-react'
import { resultsApi } from '../../services/api'
import type { ResultsSummary, PostProcessProgress } from '../../types'
import { timesToDates } from '../../utils/dateUtils'
import HeadDeckView from './HeadDeckView'
import ExportButton from './ExportButton'
import AwaitingData from './AwaitingData'
import { lengthAbbrev, labelWithUnit } from '../../utils/unitLabels'
import { viridisColor, DIFF_COLORSCALE } from '../../utils/colorScales'

const NUM_COLOR_BINS = 25

interface HeadContourChartProps {
  projectId: string
  runId: string
  summary: ResultsSummary
  expanded?: boolean
  compareRunId?: string | null
  compareSummary?: ResultsSummary
  awaitingProgress?: PostProcessProgress
}

export default function HeadContourChart({ projectId, runId, summary, expanded = false, compareRunId, compareSummary, awaitingProgress }: HeadContourChartProps) {
  // Show awaiting data state if progress is provided and heads not yet complete
  if (awaitingProgress && !awaitingProgress.postprocess_completed?.includes('heads')) {
    return <AwaitingData dataType="heads" progress={awaitingProgress} height={expanded ? window.innerHeight - 200 : 400} />
  }
  const { metadata, heads_summary } = summary
  const kstpkperList = heads_summary.kstpkper_list || []
  const gridType = metadata.grid_type || 'structured'
  const isPolygonGrid = gridType === 'unstructured' || gridType === 'vertex'
  const lu = lengthAbbrev(metadata.length_unit)
  const headLabel = labelWithUnit('Head', lu)

  const [layer, setLayer] = useState(0)
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [intervalMs, setIntervalMs] = useState(500)
  const [showMapView, setShowMapView] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const kstp = kstpkperList[selectedIdx]?.[0] ?? 0
  const kper = kstpkperList[selectedIdx]?.[1] ?? 0

  // Compute date labels for step selector when start_date and stress_period_data are available
  const stepDateLabels = useMemo(() => {
    const spData = metadata.stress_period_data
    const startDate = metadata.start_date
    if (!spData || !startDate || kstpkperList.length === 0) return null

    // Build cumulative times for each kstpkper entry
    const times: number[] = []
    for (const [ks, kp] of kstpkperList) {
      let cumTime = 0
      // Sum all complete stress periods before kp
      for (let p = 0; p < kp && p < spData.length; p++) {
        cumTime += spData[p].perlen
      }
      // Add fraction of current stress period for the timestep
      if (kp < spData.length) {
        const sp = spData[kp]
        const { perlen, nstp, tsmult } = sp
        if (tsmult === 1 || !tsmult) {
          // Equal timesteps
          cumTime += perlen * ((ks + 1) / nstp)
        } else {
          // Geometric series
          const dt1 = perlen * (tsmult - 1) / (Math.pow(tsmult, nstp) - 1)
          for (let s = 0; s <= ks; s++) {
            cumTime += dt1 * Math.pow(tsmult, s)
          }
        }
      }
      times.push(cumTime)
    }

    const dates = timesToDates(startDate, times, metadata.time_unit)
    return dates
  }, [metadata.stress_period_data, metadata.start_date, metadata.time_unit, kstpkperList])

  // Fetch head slice data for Plotly rendering (not in map mode)
  const { data: headSlice, isLoading: sliceLoading } = useQuery({
    queryKey: ['heads', projectId, runId, layer, kper, kstp],
    queryFn: () => resultsApi.getHeads(projectId, runId, layer, kper, kstp),
    enabled: kstpkperList.length > 0 && !showMapView,
  })

  // Fetch grid geometry for polygon grids (cached per run)
  const { data: gridGeometry, isLoading: geomLoading } = useQuery({
    queryKey: ['grid-geometry', projectId, runId],
    queryFn: () => resultsApi.getGridGeometry(projectId, runId),
    enabled: isPolygonGrid && !showMapView,
    staleTime: Infinity,
  })

  // Fetch comparison head slice for difference map
  const { data: compareSlice, isLoading: compareSliceLoading } = useQuery({
    queryKey: ['heads', projectId, compareRunId, layer, kper, kstp],
    queryFn: () => resultsApi.getHeads(projectId, compareRunId!, layer, kper, kstp),
    enabled: !!compareRunId && kstpkperList.length > 0 && !showMapView,
  })

  // Compute head difference (A - B) for comparison mode
  const diffData = useMemo(() => {
    if (!compareRunId || !headSlice || !compareSlice) return null

    const rawA = headSlice.data
    const rawB = compareSlice.data

    function isInvalid(v: any) {
      return v === null || v === undefined || (typeof v === 'number' && (Math.abs(v) > 1e20 || v < -880))
    }

    // Handle both structured (2D array) and unstructured (flat or nested)
    const result: (number | null)[][] = []
    for (let r = 0; r < rawA.length; r++) {
      const row: (number | null)[] = []
      const rowA = rawA[r]
      const rowB = rawB[r] || []
      if (Array.isArray(rowA)) {
        for (let c = 0; c < rowA.length; c++) {
          const a = rowA[c]
          const b = Array.isArray(rowB) ? rowB[c] : null
          if (isInvalid(a) || isInvalid(b)) {
            row.push(null)
          } else {
            row.push((a as number) - (b as number))
          }
        }
      } else {
        // For unstructured grids, rowA might be a single value
        const a = rowA as unknown as number | null
        const b = rawB[r] as unknown as number | null
        if (isInvalid(a) || isInvalid(b)) {
          row.push(null)
        } else {
          row.push((a as number) - (b as number))
        }
      }
      result.push(row)
    }
    return result
  }, [compareRunId, headSlice, compareSlice])

  // Compute symmetric range for difference colorbar
  const diffRange = useMemo(() => {
    if (!diffData) return { min: -1, max: 1 }
    let mn = Infinity
    let mx = -Infinity
    for (const row of diffData) {
      for (const v of row) {
        if (v !== null) {
          if (v < mn) mn = v
          if (v > mx) mx = v
        }
      }
    }
    if (!isFinite(mn)) mn = -1
    if (!isFinite(mx)) mx = 1
    const absMax = Math.max(Math.abs(mn), Math.abs(mx), 0.01)
    return { min: -absMax, max: absMax }
  }, [diffData])

  // Build polygon traces for difference map
  const diffPolygonTraces = useMemo(() => {
    if (!isPolygonGrid || !gridGeometry || !diffData) return null

    const layerKey = String(layer)
    const layerData = gridGeometry.layers[layerKey] || gridGeometry.layers['0']
    if (!layerData) return null

    const polygons = layerData.polygons

    // Flatten diff data
    const values: (number | null)[] = []
    for (const row of diffData) {
      if (Array.isArray(row)) {
        values.push(...row)
      } else {
        values.push(row)
      }
    }

    const { min: vmin, max: vmax } = diffRange
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

    function diffColor(t: number): string {
      const clamped = Math.max(0, Math.min(1, t))
      for (let i = 0; i < DIFF_COLORSCALE.length - 1; i++) {
        const [t0, c0] = DIFF_COLORSCALE[i]
        const [t1, c1] = DIFF_COLORSCALE[i + 1]
        if (clamped >= t0 && clamped <= t1) {
          const f = (clamped - t0) / (t1 - t0)
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
      return DIFF_COLORSCALE[DIFF_COLORSCALE.length - 1][1]
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
      const color = diffColor(bin / (NUM_COLOR_BINS - 1))
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

    // Hover points
    const cx: number[] = []
    const cy: number[] = []
    const hoverText: string[] = []
    for (let i = 0; i < polygons.length; i++) {
      const v = i < values.length ? values[i] : null
      if (v === null) continue
      const poly = polygons[i]
      if (!poly || poly.length < 3) continue
      let sumX = 0, sumY = 0, n = 0
      for (const pt of poly) { sumX += pt[0]; sumY += pt[1]; n++ }
      if (n > 1 && poly[0][0] === poly[n - 1][0] && poly[0][1] === poly[n - 1][1]) {
        sumX -= poly[n - 1][0]; sumY -= poly[n - 1][1]; n--
      }
      cx.push(sumX / n)
      cy.push(sumY / n)
      hoverText.push(`Diff: ${v.toFixed(3)}${lu ? ' ' + lu : ''}<br>Cell: ${i + 1}`)
    }
    traces.push({
      type: 'scattergl',
      x: cx, y: cy,
      mode: 'markers',
      marker: { size: 6, color: 'rgba(0,0,0,0)' },
      text: hoverText, hoverinfo: 'text', showlegend: false,
    } as Plotly.Data)

    // Colorbar
    traces.push({
      type: 'scatter', x: [null], y: [null], mode: 'markers',
      marker: {
        size: 0, color: [vmin],
        colorscale: DIFF_COLORSCALE.map(([t, c]) => [t, c]),
        cmin: vmin, cmax: vmax,
        colorbar: { title: { text: labelWithUnit('Head Diff (A-B)', lu), side: 'right' }, thickness: 15, len: 0.9 },
        showscale: true,
      },
      showlegend: false, hoverinfo: 'skip',
    } as Plotly.Data)

    return traces
  }, [isPolygonGrid, gridGeometry, diffData, diffRange, layer])

  // Build Plotly traces for polygon grids
  const polygonTraces = useMemo(() => {
    if (!isPolygonGrid || !gridGeometry || !headSlice) return null

    const layerKey = String(layer)
    const layerData = gridGeometry.layers[layerKey] || gridGeometry.layers['0']
    if (!layerData) return null

    const polygons = layerData.polygons

    // Flatten head data to 1D
    const rawData = headSlice.data
    const values: (number | null)[] = []
    for (const row of rawData) {
      if (Array.isArray(row)) {
        values.push(...row)
      } else {
        values.push(row)
      }
    }

    const vmin = heads_summary.min_head ?? 0
    const vmax = heads_summary.max_head ?? 1
    const range = vmax - vmin || 1

    // Assign each cell to a color bin
    const binAssignments: number[] = []
    for (let i = 0; i < polygons.length; i++) {
      const v = i < values.length ? values[i] : null
      if (v === null || Math.abs(v) > 1e20 || v < -880) {
        binAssignments.push(-1)
      } else {
        const t = (v - vmin) / range
        binAssignments.push(Math.max(0, Math.min(NUM_COLOR_BINS - 1, Math.floor(t * NUM_COLOR_BINS))))
      }
    }

    // Create one scatter trace per color bin with fill='toself'
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
      const color = viridisColor(bin / (NUM_COLOR_BINS - 1))
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

    // Add scatter trace at cell centroids for hover interactivity
    const cx: number[] = []
    const cy: number[] = []
    const hoverText: string[] = []
    const cellIndices: number[] = []
    for (let i = 0; i < polygons.length; i++) {
      const v = i < values.length ? values[i] : null
      if (v === null || Math.abs(v) > 1e20 || v < -880) continue
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
      hoverText.push(`Head: ${v.toFixed(3)}${lu ? ' ' + lu : ''}<br>Cell: ${i + 1}`)
      cellIndices.push(i)
    }

    // Hover points (invisible markers at centroids)
    traces.push({
      type: 'scattergl',
      x: cx,
      y: cy,
      mode: 'markers',
      marker: {
        size: 6,
        color: 'rgba(0,0,0,0)',
      },
      text: hoverText,
      hoverinfo: 'text',
      showlegend: false,
      customdata: cellIndices,
    } as Plotly.Data)

    // Dummy trace for colorbar
    traces.push({
      type: 'scatter',
      x: [null],
      y: [null],
      mode: 'markers',
      marker: {
        size: 0,
        color: [vmin],
        colorscale: 'Viridis',
        cmin: vmin,
        cmax: vmax,
        colorbar: {
          title: { text: headLabel, side: 'right' },
          thickness: 15,
          len: 0.9,
        },
        showscale: true,
      },
      showlegend: false,
      hoverinfo: 'skip',
    } as Plotly.Data)

    return traces
  }, [isPolygonGrid, gridGeometry, headSlice, layer, heads_summary.min_head, heads_summary.max_head])

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

  // Stop playing when run/layer changes
  useEffect(() => {
    setPlaying(false)
    setSelectedIdx(0)
  }, [runId, layer])

  if (kstpkperList.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-400">
        No head data available
      </div>
    )
  }

  const canAnimate = kstpkperList.length > 1
  const chartHeight = expanded ? window.innerHeight - 200 : 500
  const polygonLoading = isPolygonGrid && (sliceLoading || geomLoading)

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
                <option key={i} value={i}>
                  SP{kp + 1} TS{ks + 1}{stepDateLabels?.[i] ? ` (${stepDateLabels[i]})` : ''}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={() => setShowMapView(!showMapView)}
            className={`flex items-center gap-1 px-2 py-1 text-xs rounded border transition-colors ${
              showMapView
                ? 'bg-blue-50 border-blue-300 text-blue-600'
                : 'border-slate-300 text-slate-500 hover:text-slate-700'
            }`}
            title={showMapView ? 'Switch to grid view' : 'Switch to map view'}
          >
            {showMapView ? (
              <>
                <Grid3X3 className="h-3.5 w-3.5" />
                <span>Grid View</span>
              </>
            ) : (
              <>
                <MapIcon className="h-3.5 w-3.5" />
                <span>Map View</span>
              </>
            )}
          </button>
          <ExportButton url={resultsApi.exportHeadsCsvUrl(projectId, runId, layer, kper, kstp)} />
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
      )}

      {compareRunId && (
        <div className="text-xs text-orange-500 mb-2 font-medium">
          Showing Head Difference (Run A - Run B)
        </div>
      )}

      <div>
        {showMapView ? (
          <HeadDeckView
            projectId={projectId}
            runId={runId}
            summary={summary}
            layer={layer}
            kper={kper}
            kstp={kstp}
            expanded={expanded}
            compareRunId={compareRunId}
            compareSummary={compareSummary}
          />
        ) : compareRunId && (sliceLoading || compareSliceLoading || (isPolygonGrid && geomLoading)) ? (
          <div className="h-72 flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
          </div>
        ) : compareRunId && diffData && isPolygonGrid && diffPolygonTraces ? (
          <Plot
            data={diffPolygonTraces}
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
        ) : compareRunId && diffData && !isPolygonGrid ? (
          <Plot
            data={[
              {
                z: diffData,
                type: 'heatmap',
                colorscale: DIFF_COLORSCALE as any,
                colorbar: { title: { text: labelWithUnit('Head Diff (A-B)', lu), side: 'right' } },
                hoverongaps: false,
                xgap: 0,
                ygap: 0,
                zmin: diffRange.min,
                zmax: diffRange.max,
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
        ) : polygonLoading ? (
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
        ) : sliceLoading ? (
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
    </div>
  )
}
