import { useState, useMemo, useCallback, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { Loader2, RefreshCw } from 'lucide-react'
import { resultsApi, zoneDefinitionsApi, zoneBudgetApi } from '../../services/api'
import type { ResultsSummary, PostProcessProgress, StressPeriodData } from '../../types'
import { timesToDates, kperToEndTimes } from '../../utils/dateUtils'
import ExportButton from './ExportButton'
import AwaitingData from './AwaitingData'
import ZonePainterModal from './ZonePainterModal'
import type { LayerZoneAssignments } from '../../utils/zoneColors'
import { flowRateLabel, volumeLabel, timeAbbrev, labelWithUnit } from '../../utils/unitLabels'

// Colorblind-safe palette (Okabe-Ito / Wong 2011 + Paul Tol extensions)
const COLORS = [
  '#0072B2', // blue
  '#E69F00', // orange
  '#009E73', // bluish green
  '#D55E00', // vermillion
  '#CC79A7', // reddish purple
  '#56B4E9', // sky blue
  '#F0E442', // yellow
  '#332288', // indigo
  '#882255', // wine
  '#44AA99', // teal
]

// Inflow area alpha (80%)
const INFLOW_ALPHA = 'CC'
// Outflow area alpha (50%)
const OUTFLOW_ALPHA = '80'

// MODFLOW package acronyms that should stay uppercase
const ACRONYMS = new Set([
  'ET', 'EVT', 'ETS', 'CHD', 'WEL', 'DRN', 'GHB', 'RCH', 'RIV',
  'SFR', 'UZF', 'LAK', 'MAW', 'MVR', 'STO', 'SS', 'SY', 'MNW',
  'NPF', 'HFB', 'CSUB', 'BUY', 'VSC', 'API', 'OC',
])

function prettifyBudgetName(name: string): string {
  const clean = name
    .replace(/^FROM_/, '')
    .replace(/^TO_/, '')
    .replace(/_/g, ' ')
  return clean
    .split(/\s+/)
    .map(word => {
      if (ACRONYMS.has(word.toUpperCase())) return word.toUpperCase()
      // Hyphenated terms like STO-SS: check each part
      if (word.includes('-')) {
        return word.split('-').map(p =>
          ACRONYMS.has(p.toUpperCase()) ? p.toUpperCase() : p.charAt(0).toUpperCase() + p.slice(1).toLowerCase()
        ).join('-')
      }
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    })
    .join(' ')
}

// Terms to hide from charts (noise from listing/CBC output)
const IGNORED = new Set([
  'IN-OUT', 'IN_-_OUT', 'PERCENT_DISCREPANCY', 'PERCENT DISCREPANCY',
  'NUMBER_OF_TIME_STEPS', 'MULTIPLIER_FOR_DELT', 'INITIAL_TIME_STEP_SIZE',
  'TOTAL_IN', 'TOTAL_OUT', 'TOTAL IN', 'TOTAL OUT', 'TOTAL',
  // Internal cell-to-cell flows (not real sources/sinks)
  'FLOW_RIGHT_FACE', 'FLOW RIGHT FACE',
  'FLOW_FRONT_FACE', 'FLOW FRONT FACE',
  'FLOW_LOWER_FACE', 'FLOW LOWER FACE',
  'FLOW_JA_FACE', 'FLOW JA FACE',
])
const isIgnored = (n: string) => {
  const stripped = n.replace(/^FROM_/, '').replace(/^TO_/, '')
  const normalized = stripped.replace(/-/g, '_')
  return IGNORED.has(stripped) || IGNORED.has(normalized) || IGNORED.has(normalized.replace(/_/g, ' '))
}

// Storage component detection
function isStorageComponent(name: string): boolean {
  const stripped = name.replace(/^FROM_/, '').replace(/^TO_/, '').toUpperCase()
  return stripped.startsWith('STO') || stripped.startsWith('STORAGE')
}

function buildZoneBudgetTraces(zbResult: any): { traces: Plotly.Data[]; horizontal: boolean } | null {
  if (!zbResult?.records?.length || !zbResult?.columns?.length) return null

  const records = zbResult.records as Record<string, any>[]
  const columns = zbResult.columns as string[]

  const hasName = columns.includes('name')
  const hasTotim = columns.includes('totim')

  if (!hasName) return null

  // Unique budget component names
  const allNames: string[] = []
  for (const rec of records) {
    const n = rec.name as string
    if (n && !allNames.includes(n) && !isIgnored(n)) allNames.push(n)
  }

  // Unique time values
  const timeKey = hasTotim ? 'totim' : null
  const uniqueTimes: number[] = []
  if (timeKey) {
    for (const rec of records) {
      const t = rec[timeKey] as number
      if (t !== null && t !== undefined && !uniqueTimes.includes(t)) uniqueTimes.push(t)
    }
    uniqueTimes.sort((a, b) => a - b)
  }

  // Zone data columns (everything that's not metadata)
  const metaCols = new Set(['totim', 'time_step', 'stress_period', 'name', 'kper', 'kstp'])
  const zoneCols = columns.filter(c => !metaCols.has(c))
  if (zoneCols.length === 0) return null

  // Classify IN vs OUT by name prefix
  const inNames = allNames.filter(n =>
    n.startsWith('FROM_') || n.includes('_IN') || n === 'TOTAL_IN'
  )
  const outNames = allNames.filter(n =>
    n.startsWith('TO_') || n.includes('_OUT') || n === 'TOTAL_OUT'
  )
  const otherNames = allNames.filter(n => !inNames.includes(n) && !outNames.includes(n))
  const budgetInNames = [...inNames, ...otherNames]

  // Assign a consistent color per prettified component name so the same
  // component shares a color across the Inflow and Outflow bars.
  const uniquePretty: string[] = []
  for (const n of allNames) {
    const p = prettifyBudgetName(n)
    if (!uniquePretty.includes(p)) uniquePretty.push(p)
  }

  const traces: Plotly.Data[] = []
  // Track which prettified names actually produced an inflow trace
  // so outflow traces only hide from legend when the inflow trace truly exists.
  const createdInflowLabels = new Set<string>()

  if (uniqueTimes.length > 1) {
    // Multi-period: vertical stacked bars with in/out offset groups
    budgetInNames.forEach((bName) => {
      zoneCols.forEach((zCol) => {
        const xVals: number[] = []
        const yVals: number[] = []
        for (const t of uniqueTimes) {
          const rec = records.find(r => r.name === bName && r[timeKey!] === t)
          if (rec) {
            xVals.push(t)
            yVals.push(Math.abs(rec[zCol] as number ?? 0))
          }
        }
        if (yVals.some(v => v !== 0)) {
          const pretty = prettifyBudgetName(bName)
          const colorIdx = uniquePretty.indexOf(pretty)
          const label = zoneCols.length === 1 ? pretty : `${pretty} ${zCol}`
          createdInflowLabels.add(label)
          traces.push({
            x: xVals,
            y: yVals,
            type: 'bar',
            name: label,
            offsetgroup: 'in',
            legendgroup: label,
            marker: { color: COLORS[colorIdx % COLORS.length], opacity: 0.85 },
          } as Plotly.Data)
        }
      })
    })

    outNames.forEach((bName) => {
      zoneCols.forEach((zCol) => {
        const xVals: number[] = []
        const yVals: number[] = []
        for (const t of uniqueTimes) {
          const rec = records.find(r => r.name === bName && r[timeKey!] === t)
          if (rec) {
            xVals.push(t)
            yVals.push(Math.abs(rec[zCol] as number ?? 0))
          }
        }
        if (yVals.some(v => v !== 0)) {
          const pretty = prettifyBudgetName(bName)
          const colorIdx = uniquePretty.indexOf(pretty)
          const label = zoneCols.length === 1 ? pretty : `${pretty} ${zCol}`
          traces.push({
            x: xVals,
            y: yVals,
            type: 'bar',
            name: label,
            offsetgroup: 'out',
            legendgroup: label,
            showlegend: !createdInflowLabels.has(label),
            marker: {
              color: COLORS[colorIdx % COLORS.length],
              opacity: 0.85,
              pattern: { shape: '/' },
            },
          } as Plotly.Data)
        }
      })
    })
  } else {
    // Single period: horizontal stacked bars (Inflow / Outflow rows)
    budgetInNames.forEach((bName) => {
      zoneCols.forEach((zCol) => {
        const matchingRecs = records.filter(r => r.name === bName)
        const val = matchingRecs.reduce((sum, r) => sum + Math.abs(r[zCol] as number ?? 0), 0)
        if (val !== 0) {
          const pretty = prettifyBudgetName(bName)
          const colorIdx = uniquePretty.indexOf(pretty)
          createdInflowLabels.add(pretty)
          traces.push({
            y: [zoneCols.length === 1 ? 'Inflow' : `${zCol} Inflow`],
            x: [val],
            type: 'bar',
            orientation: 'h',
            name: pretty,
            legendgroup: pretty,
            marker: { color: COLORS[colorIdx % COLORS.length], opacity: 0.85 },
          })
        }
      })
    })

    outNames.forEach((bName) => {
      zoneCols.forEach((zCol) => {
        const matchingRecs = records.filter(r => r.name === bName)
        const val = matchingRecs.reduce((sum, r) => sum + Math.abs(r[zCol] as number ?? 0), 0)
        if (val !== 0) {
          const pretty = prettifyBudgetName(bName)
          const colorIdx = uniquePretty.indexOf(pretty)
          traces.push({
            y: [zoneCols.length === 1 ? 'Outflow' : `${zCol} Outflow`],
            x: [val],
            type: 'bar',
            orientation: 'h',
            name: pretty,
            legendgroup: pretty,
            showlegend: !createdInflowLabels.has(pretty),
            marker: {
              color: COLORS[colorIdx % COLORS.length],
              opacity: 0.85,
              pattern: { shape: '/' },
            },
          })
        }
      })
    })
  }

  if (traces.length === 0) return null
  return { traces, horizontal: uniqueTimes.length <= 1 }
}

// ─── Zone Budget Timeseries Transform ────────────────────────────────────────

interface TransformedTimestep {
  kper: number
  time: number
  date: string | null
  inflows: Record<string, Record<string, number>>   // component → zone → value
  outflows: Record<string, Record<string, number>>
}

interface TransformedZoneBudget {
  timesteps: TransformedTimestep[]
  zones: string[]              // ["ZONE_1", "ZONE_2"]
  zoneNames: string[]          // ["Zone 1", "Zone 2"]
  inflowComponents: string[]   // prettified names
  outflowComponents: string[]
  rawInflowNames: string[]     // original FROM_ names
  rawOutflowNames: string[]    // original TO_ names
  cumulativeStorage: Record<string, number[]>  // zone → running sum array
  hasStorage: boolean
}

function transformZoneBudgetRecords(
  zbResult: any,
  stressPeriodData?: StressPeriodData[],
  startDate?: string | null,
  timeUnit?: string,
): TransformedZoneBudget | null {
  if (!zbResult?.records?.length || !zbResult?.columns?.length) return null

  const records = zbResult.records as Record<string, any>[]
  const columns = zbResult.columns as string[]

  if (!columns.includes('name')) return null

  // Handle both naming conventions: MF6 uses 'kper'/'kstp', classic uses 'stress_period'/'time_step'
  const kperCol = columns.includes('kper') ? 'kper' : columns.includes('stress_period') ? 'stress_period' : null
  const kstpCol = columns.includes('kstp') ? 'kstp' : columns.includes('time_step') ? 'time_step' : null
  const hasTotim = columns.includes('totim')
  if (!kperCol && !hasTotim) return null

  // Zone data columns (exclude ZONE_0 which is unassigned/inactive cells)
  const metaCols = new Set(['totim', 'time_step', 'stress_period', 'name', 'kper', 'kstp'])
  const zoneCols = columns.filter(c => !metaCols.has(c) && c !== 'ZONE_0')
  if (zoneCols.length === 0) return null

  // Detect kper indexing: if min >= 1, normalize to 0-based
  let kperOffset = 0
  if (kperCol) {
    const allKper = records.map(r => r[kperCol] as number).filter(k => k !== null && k !== undefined)
    if (allKper.length > 0 && Math.min(...allKper) >= 1) {
      kperOffset = 1
    }
  }

  // Collect all unique budget component names
  const allNames: string[] = []
  for (const rec of records) {
    const n = rec.name as string
    if (n && !allNames.includes(n) && !isIgnored(n)) allNames.push(n)
  }

  // Classify IN vs OUT
  const inNames = allNames.filter(n => n.startsWith('FROM_'))
  const outNames = allNames.filter(n => n.startsWith('TO_'))

  // Filter out storage from main chart components
  const nonStorageIn = inNames.filter(n => !isStorageComponent(n))
  const nonStorageOut = outNames.filter(n => !isStorageComponent(n))
  const storageIn = inNames.filter(n => isStorageComponent(n))
  const storageOut = outNames.filter(n => isStorageComponent(n))
  const hasStorage = storageIn.length > 0 || storageOut.length > 0

  // Group records by stress period; take last timestep per period as representative value
  const kperMap = new Map<number, Map<string, Record<string, any>>>()
  for (const rec of records) {
    // Use kper/stress_period column, or derive from totim index
    const rawKper = kperCol ? (rec[kperCol] as number) : null
    const kper = rawKper !== null ? rawKper - kperOffset : 0
    const name = rec.name as string
    if (!name || isIgnored(name)) continue

    // For totim-only data without kper, use totim as the grouping key
    const groupKey = rawKper !== null ? kper : (rec.totim as number ?? 0)

    if (!kperMap.has(groupKey)) kperMap.set(groupKey, new Map())
    const nameMap = kperMap.get(groupKey)!
    const existingRec = nameMap.get(name)
    const kstp = kstpCol ? (rec[kstpCol] as number ?? 0) : 0
    const existingKstp = existingRec && kstpCol ? (existingRec[kstpCol] as number ?? 0) : 0
    if (!existingRec || kstp >= existingKstp) {
      nameMap.set(name, rec)
    }
  }

  // Sorted unique period keys
  const sortedKpers = Array.from(kperMap.keys()).sort((a, b) => a - b)

  // Compute times from stress period data, or use totim if available
  const endTimes = stressPeriodData && stressPeriodData.length > 0 ? kperToEndTimes(stressPeriodData) : null
  // If no stressPeriodData but records have totim, use totim values directly
  const totimValues = hasTotim && !endTimes
    ? sortedKpers.map(k => {
        const nameMap = kperMap.get(k)!
        const firstRec = nameMap.values().next().value
        return firstRec?.totim as number ?? k
      })
    : null
  const effectiveTimes = endTimes ?? totimValues
  const dates = (effectiveTimes && startDate)
    ? timesToDates(startDate, effectiveTimes, timeUnit)
    : null

  // Build timestep array
  const timesteps: TransformedTimestep[] = sortedKpers.map((kper, idx) => {
    const nameMap = kperMap.get(kper)!
    const time = effectiveTimes && idx < effectiveTimes.length ? effectiveTimes[idx] : kper
    const date = dates && idx < dates.length ? dates[idx] : null

    const inflows: Record<string, Record<string, number>> = {}
    const outflows: Record<string, Record<string, number>> = {}

    for (const bName of nonStorageIn) {
      const rec = nameMap.get(bName)
      if (!rec) continue
      const pretty = prettifyBudgetName(bName)
      if (!inflows[pretty]) inflows[pretty] = {}
      for (const z of zoneCols) {
        inflows[pretty][z] = Math.abs(rec[z] as number ?? 0)
      }
    }

    for (const bName of nonStorageOut) {
      const rec = nameMap.get(bName)
      if (!rec) continue
      const pretty = prettifyBudgetName(bName)
      if (!outflows[pretty]) outflows[pretty] = {}
      for (const z of zoneCols) {
        outflows[pretty][z] = Math.abs(rec[z] as number ?? 0)
      }
    }

    return { kper, time, date, inflows, outflows }
  })

  // Compute cumulative storage per zone
  const cumulativeStorage: Record<string, number[]> = {}
  if (hasStorage) {
    for (const z of zoneCols) {
      const cumSum: number[] = []
      let running = 0
      for (const kper of sortedKpers) {
        const nameMap = kperMap.get(kper)!
        let stoIn = 0
        let stoOut = 0
        for (const sName of storageIn) {
          const rec = nameMap.get(sName)
          if (rec) stoIn += Math.abs(rec[z] as number ?? 0)
        }
        for (const sName of storageOut) {
          const rec = nameMap.get(sName)
          if (rec) stoOut += Math.abs(rec[z] as number ?? 0)
        }
        running += stoIn - stoOut
        cumSum.push(running)
      }
      cumulativeStorage[z] = cumSum
    }
  }

  // Collect unique prettified component names
  const inflowComponents = [...new Set(nonStorageIn.map(prettifyBudgetName))]
  const outflowComponents = [...new Set(nonStorageOut.map(prettifyBudgetName))]

  // Zone display names
  const zoneNames = zoneCols.map(z => z.replace(/_/g, ' ').replace(/^ZONE /i, 'Zone '))

  return {
    timesteps,
    zones: zoneCols,
    zoneNames,
    inflowComponents,
    outflowComponents,
    rawInflowNames: nonStorageIn,
    rawOutflowNames: nonStorageOut,
    cumulativeStorage,
    hasStorage,
  }
}

// ─── Zone Budget Timeseries Trace Builder ────────────────────────────────────

function buildZoneTimeseriesTraces(
  transformed: TransformedZoneBudget,
  selectedZone: string | null,
  yLabel: string,
  chartHeight: number,
): { traces: Plotly.Data[]; layout: Partial<Plotly.Layout> } {
  const { timesteps, zones, zoneNames, inflowComponents, outflowComponents, cumulativeStorage, hasStorage } = transformed

  const hasDate = timesteps.length > 0 && timesteps[0].date !== null
  const xValues = timesteps.map(ts => hasDate ? ts.date! : ts.time)

  // Build unique component color map (same component shares color across in/out)
  const allComponents = [...new Set([...inflowComponents, ...outflowComponents])]
  const componentColorIdx = new Map<string, number>()
  allComponents.forEach((c, i) => componentColorIdx.set(c, i))

  const traces: Plotly.Data[] = []

  if (selectedZone === null) {
    // "All Zones" summary: total inflow/outflow line per zone
    zones.forEach((zone, zi) => {
      const color = COLORS[zi % COLORS.length]
      const zoneName = zoneNames[zi]

      // Total inflow per timestep for this zone
      const inVals = timesteps.map(ts => {
        let sum = 0
        for (const comp of inflowComponents) {
          sum += ts.inflows[comp]?.[zone] ?? 0
        }
        return sum
      })

      // Total outflow per timestep for this zone (negated)
      const outVals = timesteps.map(ts => {
        let sum = 0
        for (const comp of outflowComponents) {
          sum += ts.outflows[comp]?.[zone] ?? 0
        }
        return -sum
      })

      traces.push({
        x: xValues,
        y: inVals,
        type: 'scatter',
        mode: 'lines',
        name: `${zoneName} In`,
        legendgroup: zoneName,
        line: { color, width: 2 },
      })

      traces.push({
        x: xValues,
        y: outVals,
        type: 'scatter',
        mode: 'lines',
        name: `${zoneName} Out`,
        legendgroup: zoneName,
        showlegend: false,
        line: { color, width: 2, dash: 'dash' },
      })
    })
  } else {
    // Per-zone stacked area: inflow above zero, outflow below
    const zoneIdx = zones.indexOf(selectedZone)
    if (zoneIdx < 0) return { traces: [], layout: {} }

    // Inflow stacked area traces
    inflowComponents.forEach((comp) => {
      const ci = componentColorIdx.get(comp) ?? 0
      const baseColor = COLORS[ci % COLORS.length]
      const yVals = timesteps.map(ts => ts.inflows[comp]?.[selectedZone] ?? 0)

      if (yVals.some(v => v !== 0)) {
        traces.push({
          x: xValues,
          y: yVals,
          type: 'scatter',
          name: comp,
          legendgroup: comp,
          stackgroup: 'inflow',
          fillcolor: baseColor + INFLOW_ALPHA,
          line: { color: baseColor, width: 0.5 },
        })
      }
    })

    // Outflow stacked area traces (negated y-values)
    outflowComponents.forEach((comp) => {
      const ci = componentColorIdx.get(comp) ?? 0
      const baseColor = COLORS[ci % COLORS.length]
      const yVals = timesteps.map(ts => -(ts.outflows[comp]?.[selectedZone] ?? 0))
      const hasInflow = inflowComponents.includes(comp)

      if (yVals.some(v => v !== 0)) {
        traces.push({
          x: xValues,
          y: yVals,
          type: 'scatter',
          name: comp,
          legendgroup: comp,
          showlegend: !hasInflow,
          stackgroup: 'outflow',
          fillcolor: baseColor + OUTFLOW_ALPHA,
          line: { color: baseColor, width: 0.5, dash: 'dot' },
        })
      }
    })
  }

  // Zero reference line
  traces.push({
    x: [xValues[0], xValues[xValues.length - 1]],
    y: [0, 0],
    type: 'scatter',
    mode: 'lines',
    name: '',
    showlegend: false,
    line: { color: '#94a3b8', width: 1, dash: 'dash' },
    hoverinfo: 'skip',
  })

  // Layout
  const showStorageSubplot = hasStorage && selectedZone !== null
  const mainDomain: [number, number] = showStorageSubplot ? [0.25, 1.0] : [0, 1.0]
  const storageDomain: [number, number] = [0, 0.18]

  const layout: Partial<Plotly.Layout> = {
    autosize: true,
    height: chartHeight,
    margin: { l: 60, r: 20, t: 10, b: 80 },
    xaxis: {
      title: { text: hasDate ? 'Date' : undefined },
      type: hasDate ? 'date' : undefined,
      domain: [0, 1],
    },
    yaxis: {
      title: { text: yLabel },
      domain: mainDomain,
      zeroline: false,
    },
    legend: {
      orientation: 'h',
      y: -0.35,
      x: 0.5,
      xanchor: 'center',
      font: { size: 10 },
    },
  }

  // Storage subplot
  if (showStorageSubplot && selectedZone) {
    const stoVals = cumulativeStorage[selectedZone]
    if (stoVals && stoVals.some(v => v !== 0)) {
      traces.push({
        x: xValues,
        y: stoVals,
        type: 'scatter',
        mode: 'lines',
        name: 'Cumulative \u0394Storage',
        showlegend: true,
        yaxis: 'y2',
        xaxis: 'x2',
        line: { color: '#6366f1', width: 2 },
      })

      Object.assign(layout, {
        xaxis2: {
          anchor: 'y2',
          matches: 'x',
          showticklabels: false,
        },
        yaxis2: {
          title: { text: '\u0394Storage', font: { size: 10 } },
          domain: storageDomain,
        },
        grid: { rows: 2, columns: 1, roworder: 'top to bottom' as const },
      })
    }
  }

  // All Zones summary with storage lines
  if (hasStorage && selectedZone === null) {
    zones.forEach((zone, zi) => {
      const stoVals = cumulativeStorage[zone]
      if (stoVals && stoVals.some(v => v !== 0)) {
        const color = COLORS[zi % COLORS.length]
        const zoneName = zoneNames[zi]
        traces.push({
          x: xValues,
          y: stoVals,
          type: 'scatter',
          mode: 'lines',
          name: `${zoneName} \u0394Sto`,
          legendgroup: zoneName,
          showlegend: false,
          yaxis: 'y2',
          xaxis: 'x2',
          line: { color, width: 1.5, dash: 'dot' },
        })
      }
    })

    // Add storage subplot for All Zones view too
    const hasAnyStorage = zones.some(z => cumulativeStorage[z]?.some(v => v !== 0))
    if (hasAnyStorage) {
      layout.yaxis!.domain = [0.25, 1.0]
      Object.assign(layout, {
        xaxis2: {
          anchor: 'y2',
          matches: 'x',
          showticklabels: false,
        },
        yaxis2: {
          title: { text: '\u0394Storage', font: { size: 10 } },
          domain: storageDomain,
        },
        grid: { rows: 2, columns: 1, roworder: 'top to bottom' as const },
      })
    }
  }

  return { traces, layout }
}

// Comparison palette — desaturated/lighter variants for Run B
const COMPARE_COLORS = [
  '#5DA5D5', // light blue
  '#F2C75C', // light orange
  '#5DC09E', // light green
  '#E8895C', // light vermillion
  '#DDA4C4', // light purple
  '#8DD3F0', // pale sky blue
  '#F5EF84', // pale yellow
  '#7766BB', // light indigo
  '#BB6699', // light wine
  '#77CCBB', // light teal
]

interface WaterBudgetChartProps {
  projectId: string
  runId: string
  summary: ResultsSummary
  expanded?: boolean
  compareRunId?: string | null
  awaitingProgress?: PostProcessProgress
  convergenceInfo?: Record<string, unknown>
  startDate?: string | null
  timeUnit?: string
  lengthUnit?: string
  stressPeriodData?: StressPeriodData[]
}

export default function WaterBudgetChart({ projectId, runId, summary, expanded = false, compareRunId, awaitingProgress, convergenceInfo, startDate, timeUnit, lengthUnit, stressPeriodData }: WaterBudgetChartProps) {
  // Show awaiting data state if progress is provided and budget not yet complete
  if (awaitingProgress && !awaitingProgress.postprocess_completed?.includes('budget')) {
    return <AwaitingData dataType="budget" progress={awaitingProgress} height={expanded ? window.innerHeight - 160 : 400} />
  }

  const [viewMode, setViewMode] = useState<'model' | 'zone'>('model')
  const [selectedZone, setSelectedZone] = useState<string | null>(null)

  // Zone budget state
  const [zoneAssignments, setZoneAssignments] = useState<LayerZoneAssignments>({})
  const [painterOpen, setPainterOpen] = useState(false)
  const [customZoneBudgetResult, setCustomZoneBudgetResult] = useState<any>(null)

  const hasCustomZoneBudget = customZoneBudgetResult?.records?.length > 0
  const queryClient = useQueryClient()
  const [refreshing, setRefreshing] = useState(false)
  const [savedComputing, setSavedComputing] = useState(false)
  const [savedComputeError, setSavedComputeError] = useState<string | null>(null)
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Fetch saved zone definitions for quick access
  const { data: savedDefs } = useQuery({
    queryKey: ['zone-definitions', projectId],
    queryFn: () => zoneDefinitionsApi.list(projectId),
    staleTime: 30_000,
  })

  // Convert API zone definition format to LayerZoneAssignments
  const defnToAssignments = (defn: { zone_layers: Record<string, Record<string, number[]>> }): LayerZoneAssignments => {
    const loaded: LayerZoneAssignments = {}
    for (const [layStr, zones] of Object.entries(defn.zone_layers)) {
      const layerMap: Record<number, number> = {}
      for (const [zoneName, cellIndices] of Object.entries(zones)) {
        const zoneNum = parseInt(zoneName.replace('Zone ', ''), 10)
        if (!isNaN(zoneNum)) {
          for (const ci of cellIndices) {
            layerMap[ci] = zoneNum
          }
        }
      }
      loaded[Number(layStr)] = layerMap
    }
    return loaded
  }

  // Load a saved zone definition and directly compute the zone budget
  const handleLoadSavedAndCompute = async (name: string) => {
    setSavedComputing(true)
    setSavedComputeError(null)

    try {
      const defn = await zoneDefinitionsApi.get(projectId, name)
      setZoneAssignments(defnToAssignments(defn))

      // Compute zone budget directly using the zone layers from the definition
      const response = await zoneBudgetApi.compute(projectId, runId, defn.zone_layers, false)

      if (response.status === 'completed' && response.result) {
        setCustomZoneBudgetResult(response.result)
        setViewMode('zone')
        setSavedComputing(false)
        return
      }

      if (response.status === 'queued' && response.task_id) {
        // Poll for async result
        const taskId = response.task_id
        const poll = async () => {
          try {
            const progress = await zoneBudgetApi.getStatus(projectId, runId, taskId)
            if (progress.status === 'completed') {
              const result = await zoneBudgetApi.getResult(projectId, runId, taskId)
              setCustomZoneBudgetResult(result)
              setViewMode('zone')
              setSavedComputing(false)
              return
            }
            if (progress.status === 'failed') {
              setSavedComputeError(progress.error || 'Computation failed')
              setSavedComputing(false)
              return
            }
            pollTimerRef.current = setTimeout(poll, 2000)
          } catch (err: any) {
            if (err?.response?.status === 404) {
              setSavedComputeError('Task was lost. Please retry.')
              setSavedComputing(false)
              return
            }
            pollTimerRef.current = setTimeout(poll, 3000)
          }
        }
        poll()
        return
      }

      setSavedComputing(false)
    } catch (err: any) {
      console.error('Zone budget load/compute failed:', err)
      setSavedComputeError(err instanceof Error ? err.message : 'Failed to compute zone budget')
      setSavedComputing(false)
    }
  }

  // Fetch the whole-model zone budget (single zone = entire model)
  const { data: modelZoneBudget, isLoading } = useQuery({
    queryKey: ['model-zone-budget', projectId, runId],
    queryFn: () => resultsApi.getModelZoneBudget(projectId, runId),
  })

  // Fetch comparison run's model zone budget
  const { data: compareZoneBudget, isLoading: compareLoading } = useQuery({
    queryKey: ['model-zone-budget', projectId, compareRunId],
    queryFn: () => resultsApi.getModelZoneBudget(projectId, compareRunId!),
    enabled: !!compareRunId,
  })

  const handleRefreshBudget = useCallback(async () => {
    setRefreshing(true)
    try {
      // Fetch with refresh=true to bust server cache, then update query cache
      const fresh = await resultsApi.getModelZoneBudget(projectId, runId, true)
      queryClient.setQueryData(['model-zone-budget', projectId, runId], fresh)
    } finally {
      setRefreshing(false)
    }
  }, [projectId, runId, queryClient])

  // Parse model zone budget into traces
  const modelData = useMemo(() => buildZoneBudgetTraces(modelZoneBudget), [modelZoneBudget])

  // Parse comparison zone budget into traces (with orange colors)
  const compareData = useMemo(() => {
    if (!compareZoneBudget) return null
    const result = buildZoneBudgetTraces(compareZoneBudget)
    if (!result) return null
    // Re-color all comparison traces to orange family and prefix names
    const recolored = result.traces.map((trace, i) => ({
      ...trace,
      name: `B: ${(trace as any).name || ''}`,
      marker: {
        ...((trace as any).marker || {}),
        color: COMPARE_COLORS[i % COMPARE_COLORS.length],
        opacity: 0.85,
      },
      legendgroup: `cmp_${(trace as any).legendgroup || i}`,
    }))
    return { ...result, traces: recolored }
  }, [compareZoneBudget])

  // Parse custom (painted) zone budget into traces (bar charts)
  const customData = useMemo(() => buildZoneBudgetTraces(customZoneBudgetResult), [customZoneBudgetResult])

  // Transform zone budget records for timeseries view
  const transformedZoneBudget = useMemo(() => {
    if (!customZoneBudgetResult) return null
    return transformZoneBudgetRecords(customZoneBudgetResult, stressPeriodData, startDate, timeUnit)
  }, [customZoneBudgetResult, stressPeriodData, startDate, timeUnit])

  // Determine if we should show timeseries (multi-timestep zone budget)
  const hasMultipleTimesteps = transformedZoneBudget && transformedZoneBudget.timesteps.length > 1
  const showZoneTimeseries = viewMode === 'zone' && hasCustomZoneBudget && hasMultipleTimesteps


  const chartHeight = expanded ? window.innerHeight - 160 : 400
  const showCustomView = viewMode === 'zone' && hasCustomZoneBudget

  // y-axis label: listing file source gives cumulative volumes, CBC gives flow rates
  const isListingSource = modelZoneBudget?.source === 'listing_file'
  const yLabel = isListingSource ? volumeLabel(lengthUnit) : flowRateLabel(lengthUnit, timeUnit)
  const tu = timeAbbrev(timeUnit)
  const xTimeLabel = labelWithUnit('Time', tu)

  // Build timeseries traces/layout if applicable
  const timeseriesChart = useMemo(() => {
    if (!showZoneTimeseries || !transformedZoneBudget) return null
    return buildZoneTimeseriesTraces(transformedZoneBudget, selectedZone, yLabel, chartHeight)
  }, [showZoneTimeseries, transformedZoneBudget, selectedZone, yLabel, chartHeight])

  if (isLoading || (compareRunId && compareLoading)) {
    return (
      <div className="h-72 flex items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
      </div>
    )
  }

  const activeData = showCustomView && !showZoneTimeseries ? customData : (!showCustomView ? modelData : null)

  const hasSavedDefs = savedDefs && savedDefs.length > 0

  if (!activeData && !painterOpen && !showZoneTimeseries && !hasSavedDefs) {
    return (
      <div className="h-72 flex flex-col items-center justify-center text-slate-400">
        <p className="text-lg font-medium text-slate-500 mb-2">
          {showCustomView ? 'Zone budget data could not be charted' : 'No budget data available'}
        </p>
        {!showCustomView && (
          <p className="text-sm text-slate-400 max-w-md text-center">
            Cell budget file (CBC) not found in simulation results.
            Enable "Save Water Budget" option when starting a new simulation to generate detailed budget data.
          </p>
        )}
      </div>
    )
  }

  // Show warning if budget was parsed from listing file or processing failed
  const budgetWarning = modelZoneBudget?.warning || (convergenceInfo?.budget_warning as string | undefined)

  // When comparing, prefix Run A trace names and combine with Run B traces
  const isHorizontal = activeData?.horizontal ?? false
  let plotTraces = activeData?.traces ?? []
  if (compareRunId && compareData && !showCustomView && activeData) {
    const aTraces = activeData.traces.map(t => ({
      ...t,
      name: `A: ${(t as any).name || ''}`,
      legendgroup: `a_${(t as any).legendgroup || ''}`,
      ...(isHorizontal
        ? { y: [(t as any).y?.[0] ? `A ${(t as any).y[0]}` : 'A'] }
        : { offsetgroup: `a_${(t as any).offsetgroup || ''}` }
      ),
    }))
    const bTraces = compareData.traces.map((t, i) => ({
      ...t,
      name: `B: ${(t as any).name || ''}`,
      legendgroup: `b_${(t as any).legendgroup || ''}`,
      marker: {
        ...((t as any).marker || {}),
        color: COMPARE_COLORS[i % COMPARE_COLORS.length],
        opacity: 0.85,
      },
      ...(isHorizontal
        ? { y: [(t as any).y?.[0] ? `B ${(t as any).y[0]}` : 'B'] }
        : { offsetgroup: `b_${(t as any).offsetgroup || ''}` }
      ),
    }))
    plotTraces = [...aTraces, ...bTraces]
  }

  // Convert numeric time x-values to dates when startDate is available (multi-period bar chart only)
  const useDateAxis = !isHorizontal && !!startDate && !showZoneTimeseries
  if (useDateAxis && activeData) {
    const allTimes = new Set<number>()
    for (const trace of plotTraces) {
      const xs = (trace as any).x
      if (Array.isArray(xs)) {
        for (const x of xs) {
          if (typeof x === 'number') allTimes.add(x)
        }
      }
    }
    const timeArr = Array.from(allTimes).sort((a, b) => a - b)
    const dateArr = timesToDates(startDate, timeArr, timeUnit)
    if (dateArr) {
      const timeToDate = new Map<number, string>()
      timeArr.forEach((t, i) => timeToDate.set(t, dateArr[i]))
      for (const trace of plotTraces) {
        const xs = (trace as any).x
        if (Array.isArray(xs)) {
          ;(trace as any).x = xs.map((x: any) =>
            typeof x === 'number' ? (timeToDate.get(x) ?? x) : x
          )
        }
      }
    }
  }

  // Zone selector UI for timeseries view
  const zoneSelector = showZoneTimeseries && transformedZoneBudget ? (
    <ZoneSelector
      zones={transformedZoneBudget.zones}
      zoneNames={transformedZoneBudget.zoneNames}
      selectedZone={selectedZone}
      onSelect={setSelectedZone}
    />
  ) : null

  return (
    <div>
      {budgetWarning && (
        <div className="mb-3 px-3 py-2 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-700">
          <span className="font-medium">Note:</span> {budgetWarning}
        </div>
      )}
      <div className="flex items-center gap-2 mb-2">
        {hasCustomZoneBudget && (
          <>
            <button
              onClick={() => setViewMode('model')}
              className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                viewMode === 'model'
                  ? 'bg-blue-50 border-blue-300 text-blue-600'
                  : 'border-slate-300 text-slate-500 hover:text-slate-700'
              }`}
            >
              Model Budget
            </button>
            <button
              onClick={() => setViewMode('zone')}
              className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                viewMode === 'zone'
                  ? 'bg-orange-50 border-orange-300 text-orange-600'
                  : 'border-slate-300 text-slate-500 hover:text-slate-700'
              }`}
            >
              Zone Budget
            </button>
          </>
        )}
        <button
          onClick={() => setPainterOpen(true)}
          className="flex items-center gap-1 px-2 py-1 text-xs rounded border border-slate-300 text-slate-500 hover:text-slate-700 transition-colors"
        >
          {hasCustomZoneBudget ? 'Edit Zones' : 'Zone Budget'}
        </button>
        {savedDefs && savedDefs.length > 0 && (
          <select
            value=""
            disabled={savedComputing}
            onChange={(e) => {
              if (e.target.value) handleLoadSavedAndCompute(e.target.value)
            }}
            className="px-2 py-1 text-xs border border-slate-300 rounded text-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-400 disabled:opacity-50"
          >
            <option value="">
              {savedComputing ? 'Computing...' : 'Load Saved...'}
            </option>
            {savedDefs.map(d => (
              <option key={d.name} value={d.name}>{d.name}</option>
            ))}
          </select>
        )}
        <div className="ml-auto flex items-center gap-1">
          <button
            onClick={handleRefreshBudget}
            disabled={refreshing}
            title="Refresh budget data (clears cached results)"
            className="p-1.5 text-slate-400 hover:text-slate-600 disabled:opacity-50 transition-colors"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? 'animate-spin' : ''}`} />
          </button>
          <ExportButton url={resultsApi.exportBudgetCsvUrl(projectId, runId)} />
        </div>
      </div>

      {savedComputing && (
        <div className="flex items-center gap-2 mb-2 px-3 py-2 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-700">
          <Loader2 className="h-4 w-4 animate-spin" />
          Computing zone budget...
        </div>
      )}

      {savedComputeError && (
        <div className="mb-2 px-3 py-2 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {savedComputeError}
        </div>
      )}

      {zoneSelector}

      {painterOpen && (
        <ZonePainterModal
          projectId={projectId}
          runId={runId}
          summary={summary}
          zoneAssignments={zoneAssignments}
          onZoneAssignmentsChange={setZoneAssignments}
          onComputeResult={(result) => {
            setCustomZoneBudgetResult(result)
            setViewMode('zone')
          }}
          onClose={() => setPainterOpen(false)}
        />
      )}

      {showZoneTimeseries && timeseriesChart ? (
        <Plot
          data={timeseriesChart.traces}
          layout={timeseriesChart.layout}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      ) : activeData && (
        <Plot
          data={plotTraces}
          layout={{
            autosize: true,
            height: chartHeight,
            margin: isHorizontal
              ? { l: 80, r: 20, t: 10, b: 80 }
              : { l: 60, r: 20, t: 10, b: 80 },
            barmode: 'stack',
            xaxis: isHorizontal
              ? { title: { text: yLabel } }
              : {
                  title: { text: useDateAxis ? 'Date' : xTimeLabel },
                  type: useDateAxis ? 'date' : undefined,
                },
            yaxis: isHorizontal
              ? { autorange: 'reversed' as const }
              : { title: { text: yLabel } },
            legend: {
              orientation: 'h',
              y: -0.35,
              x: 0.5,
              xanchor: 'center',
              font: { size: 10 },
            },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      )}
    </div>
  )
}

// ─── Zone Selector Component ─────────────────────────────────────────────────

function ZoneSelector({
  zones,
  zoneNames,
  selectedZone,
  onSelect,
}: {
  zones: string[]
  zoneNames: string[]
  selectedZone: string | null
  onSelect: (zone: string | null) => void
}) {
  if (zones.length <= 1) return null

  // 2-4 zones: tabs. 5+: dropdown.
  if (zones.length <= 4) {
    return (
      <div className="flex items-center gap-1 mb-2">
        <button
          onClick={() => onSelect(null)}
          className={`px-2 py-0.5 text-xs rounded transition-colors ${
            selectedZone === null
              ? 'bg-indigo-50 border border-indigo-300 text-indigo-600'
              : 'border border-slate-200 text-slate-500 hover:text-slate-700'
          }`}
        >
          All Zones
        </button>
        {zones.map((z, i) => (
          <button
            key={z}
            onClick={() => onSelect(z)}
            className={`px-2 py-0.5 text-xs rounded transition-colors ${
              selectedZone === z
                ? 'bg-indigo-50 border border-indigo-300 text-indigo-600'
                : 'border border-slate-200 text-slate-500 hover:text-slate-700'
            }`}
          >
            {zoneNames[i]}
          </button>
        ))}
      </div>
    )
  }

  // 5+ zones: dropdown
  return (
    <div className="flex items-center gap-2 mb-2">
      <span className="text-xs text-slate-500">Zone:</span>
      <select
        value={selectedZone ?? '__all__'}
        onChange={(e) => onSelect(e.target.value === '__all__' ? null : e.target.value)}
        className="px-2 py-0.5 text-xs border border-slate-200 rounded text-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-400"
      >
        <option value="__all__">All Zones</option>
        {zones.map((z, i) => (
          <option key={z} value={z}>{zoneNames[i]}</option>
        ))}
      </select>
    </div>
  )
}
