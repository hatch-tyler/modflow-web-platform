import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import type { ResultsSummary, PostProcessProgress } from '../../types'
import AwaitingData from './AwaitingData'
import { timeAbbrev, labelWithUnit } from '../../utils/unitLabels'

// Internal cell-to-cell flow records excluded from mass balance totals
const INTERNAL_FLOWS = new Set([
  'FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
  'FLOW_RIGHT_FACE', 'FLOW_FRONT_FACE', 'FLOW_LOWER_FACE',
  'FLOW JA FACE', 'FLOW_JA_FACE', 'FLOW-JA-FACE',
  'DATA-SPDIS', 'DATA-SAT', 'DATA-STOSS', 'DATA-STOSY',
])

function isInternalFlow(name: string): boolean {
  return INTERNAL_FLOWS.has(name.toUpperCase())
}

/** Compute mass balance error excluding internal face flows from a budget period. */
function computeMassBalanceError(period: { in: Record<string, number>; out: Record<string, number> }): number {
  let totalIn = 0
  let totalOut = 0
  for (const [name, val] of Object.entries(period.in)) {
    if (!isInternalFlow(name)) totalIn += val
  }
  for (const [name, val] of Object.entries(period.out)) {
    if (!isInternalFlow(name)) totalOut += val
  }
  const avg = (totalIn + totalOut) / 2
  return avg > 0 ? ((totalIn - totalOut) / avg) * 100 : 0
}

interface ConvergencePlotProps {
  summary: ResultsSummary
  expanded?: boolean
  compareSummary?: ResultsSummary
  awaitingProgress?: PostProcessProgress
}

export default function ConvergencePlot({ summary, expanded = false, compareSummary, awaitingProgress }: ConvergencePlotProps) {
  // Show awaiting data state if progress is provided and listing not yet complete
  if (awaitingProgress && !awaitingProgress.postprocess_completed?.includes('listing')) {
    return <AwaitingData dataType="listing" progress={awaitingProgress} height={expanded ? window.innerHeight - 160 : 350} />
  }

  const kstpkperList = summary.heads_summary?.kstpkper_list || []
  if (kstpkperList.length < 2) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-400">
        Mass balance plot requires at least 2 time steps
      </div>
    )
  }

  const tu = timeAbbrev(summary.metadata?.time_unit)

  // Build mass balance data from the best available source
  const { xValues, yValues } = useMemo(() => {
    const { budget, heads_summary, convergence } = summary
    const times = heads_summary.times || []

    // Source 1: percent_discrepancies from listing file (most reliable, always per-stress-period)
    const listingDiscrepancies = convergence?.percent_discrepancies
    if (listingDiscrepancies && listingDiscrepancies.length > 0 && times.length > 0) {
      // The listing file reports one discrepancy per stress period.
      // We need to find the stress-period-end times. If model has 1 kstp per kper,
      // times aligns 1:1. If multiple kstp per kper, we pick the last kstp in each kper.
      const kstpkper = heads_summary.kstpkper_list || []

      // Find the time index for the last kstp in each kper
      const kperLastIndex = new Map<number, number>()
      for (let i = 0; i < kstpkper.length; i++) {
        const [, kper] = kstpkper[i]
        kperLastIndex.set(kper, i) // overwrite keeps the last occurrence
      }

      // Build arrays — pair each discrepancy with its stress period end time
      const xs: number[] = []
      const ys: number[] = []

      // Get sorted kper values
      const sortedKpers = [...kperLastIndex.keys()].sort((a, b) => a - b)

      for (let i = 0; i < sortedKpers.length && i < listingDiscrepancies.length; i++) {
        const idx = kperLastIndex.get(sortedKpers[i])!
        if (idx < times.length) {
          xs.push(times[idx])
          ys.push(listingDiscrepancies[i])
        }
      }

      if (xs.length > 0) return { xValues: xs, yValues: ys }
    }

    // Source 2: Budget periods (when full budget is available, not quick_mode)
    const periods = budget?.periods || {}
    const periodKeys = Object.keys(periods)
    if (periodKeys.length > 1 && times.length > 0) {
      // Build kper → time map
      const kstpkper = heads_summary.kstpkper_list || []
      const kperTimeMap = new Map<number, number>()
      for (let i = 0; i < kstpkper.length; i++) {
        const [, kper] = kstpkper[i]
        if (i < times.length) kperTimeMap.set(kper, times[i])
      }

      const sorted = [...periodKeys].sort((a, b) => (periods[a].kper ?? 0) - (periods[b].kper ?? 0))
      const xs: number[] = []
      const ys: number[] = []
      sorted.forEach((key) => {
        const period = periods[key]
        const kper = period.kper ?? 0
        xs.push(kperTimeMap.get(kper) ?? kper + 1)
        ys.push(computeMassBalanceError(period))
      })
      return { xValues: xs, yValues: ys }
    }

    return { xValues: [] as number[], yValues: [] as number[] }
  }, [summary])

  // Same for comparison summary
  const compareData = useMemo(() => {
    if (!compareSummary) return null

    const { budget, heads_summary, convergence } = compareSummary
    const times = heads_summary.times || []

    // Source 1: listing file discrepancies
    const listingDiscrepancies = convergence?.percent_discrepancies
    if (listingDiscrepancies && listingDiscrepancies.length > 0 && times.length > 0) {
      const kstpkper = heads_summary.kstpkper_list || []
      const kperLastIndex = new Map<number, number>()
      for (let i = 0; i < kstpkper.length; i++) {
        const [, kper] = kstpkper[i]
        kperLastIndex.set(kper, i)
      }
      const sortedKpers = [...kperLastIndex.keys()].sort((a, b) => a - b)
      const xs: number[] = []
      const ys: number[] = []
      for (let i = 0; i < sortedKpers.length && i < listingDiscrepancies.length; i++) {
        const idx = kperLastIndex.get(sortedKpers[i])!
        if (idx < times.length) {
          xs.push(times[idx])
          ys.push(listingDiscrepancies[i])
        }
      }
      if (xs.length > 0) return { xValues: xs, yValues: ys }
    }

    // Source 2: Budget periods
    const periods = budget?.periods || {}
    const periodKeys = Object.keys(periods)
    if (periodKeys.length > 1 && times.length > 0) {
      const kstpkper = heads_summary.kstpkper_list || []
      const kperTimeMap = new Map<number, number>()
      for (let i = 0; i < kstpkper.length; i++) {
        const [, kper] = kstpkper[i]
        if (i < times.length) kperTimeMap.set(kper, times[i])
      }
      const sorted = [...periodKeys].sort((a, b) => (periods[a].kper ?? 0) - (periods[b].kper ?? 0))
      const xs: number[] = []
      const ys: number[] = []
      sorted.forEach((key) => {
        const period = periods[key]
        const kper = period.kper ?? 0
        xs.push(kperTimeMap.get(kper) ?? kper + 1)
        ys.push(computeMassBalanceError(period))
      })
      return { xValues: xs, yValues: ys }
    }

    return null
  }, [compareSummary])

  if (xValues.length === 0) {
    return (
      <div className="h-72 flex flex-col items-center justify-center text-slate-400">
        <p className="text-lg font-medium text-slate-500 mb-2">No mass balance data available</p>
        <p className="text-sm text-slate-400 max-w-md text-center">
          Mass balance data will be available after re-running the simulation
          with an updated post-processor.
        </p>
      </div>
    )
  }

  const traces: Plotly.Data[] = [
    {
      x: xValues,
      y: yValues,
      type: 'scatter',
      mode: 'lines+markers',
      name: compareSummary ? 'Run A' : '% Discrepancy',
      line: { color: '#8b5cf6', width: 2 },
      marker: { size: 3 },
    },
    {
      x: [xValues[0], xValues[xValues.length - 1]],
      y: [0, 0],
      type: 'scatter',
      mode: 'lines',
      name: 'Zero',
      line: { color: '#94a3b8', width: 1, dash: 'dash' },
      showlegend: false,
    },
  ]

  // Add comparison trace
  if (compareData && compareData.xValues.length > 0) {
    traces.push({
      x: compareData.xValues,
      y: compareData.yValues,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Run B',
      line: { color: '#f97316', width: 2 },
      marker: { size: 3 },
    })
  }

  const chartHeight = expanded ? window.innerHeight - 160 : 350

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height: chartHeight,
        margin: { l: 60, r: 20, t: 10, b: 50 },
        xaxis: { title: { text: labelWithUnit('Time', tu) } },
        yaxis: { title: { text: 'Mass Balance Error (%)' } },
        showlegend: !!compareSummary,
        legend: { orientation: 'h', y: 1.05 },
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
    />
  )
}
