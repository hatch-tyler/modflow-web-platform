import { memo, useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import { Loader2, FileSpreadsheet, ChevronDown } from 'lucide-react'
import { resultsApi, observationsApi } from '../../services/api'
import type { ResultsSummary, ObservationData, PostProcessProgress } from '../../types'
import { timesToDates } from '../../utils/dateUtils'
import ExportButton from './ExportButton'
import AwaitingData from './AwaitingData'
import { lengthAbbrev, timeAbbrev, labelWithUnit } from '../../utils/unitLabels'

const OBS_COLORS = ['#ef4444', '#f97316', '#22c55e', '#8b5cf6', '#06b6d4', '#ec4899', '#eab308', '#14b8a6']

interface HeadTimeSeriesChartProps {
  projectId: string
  runId: string
  summary: ResultsSummary
  expanded?: boolean
  compareRunId?: string | null
  awaitingProgress?: PostProcessProgress
}

function HeadTimeSeriesChart({ projectId, runId, summary, expanded = false, compareRunId, awaitingProgress }: HeadTimeSeriesChartProps) {
  // Show awaiting data state if progress is provided and heads not yet complete
  if (awaitingProgress && !awaitingProgress.postprocess_completed?.includes('heads')) {
    return <AwaitingData dataType="timeseries" progress={awaitingProgress} height={expanded ? window.innerHeight - 160 : 350} />
  }
  const { metadata, heads_summary } = summary
  const kstpkperList = heads_summary.kstpkper_list || []

  if (kstpkperList.length < 2) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-400">
        Head time series requires at least 2 time steps
      </div>
    )
  }
  const gridType = metadata.grid_type || 'structured'
  const isUnstructured = gridType === 'unstructured' || gridType === 'vertex'
  const lu = lengthAbbrev(metadata.length_unit)
  const tu = timeAbbrev(metadata.time_unit)
  const headLabel = labelWithUnit('Head', lu)
  const timeLabel = labelWithUnit('Time', tu)

  const [layer, setLayer] = useState(0)
  const [row, setRow] = useState(Math.floor((metadata.nrow - 1) / 2))
  const [col, setCol] = useState(Math.floor((metadata.ncol - 1) / 2))
  const [node, setNode] = useState(0)
  const [selectedWell, setSelectedWell] = useState<string>('')
  const [selectedSetId, setSelectedSetId] = useState<string>('all')
  const [showSetDropdown, setShowSetDropdown] = useState(false)

  const queryParams = isUnstructured
    ? { layer, node }
    : { layer, row, col }

  const { data: tsData, isLoading, isError } = useQuery({
    queryKey: ['timeseries', projectId, runId, queryParams],
    queryFn: () => resultsApi.getTimeseries(projectId, runId, queryParams),
  })

  // Query for observation sets
  const { data: observationSets = [] } = useQuery({
    queryKey: ['observation-sets', projectId],
    queryFn: () => observationsApi.listSets(projectId),
    retry: false,
  })

  // Query for merged observation data (for backwards compatibility)
  const { data: obsData } = useQuery({
    queryKey: ['observations', projectId],
    queryFn: () => observationsApi.get(projectId),
    retry: false,
  })

  // Query for specific set data when selected
  const { data: selectedSetData } = useQuery({
    queryKey: ['observation-set-data', projectId, selectedSetId],
    queryFn: () => observationsApi.getSet(projectId, selectedSetId),
    enabled: selectedSetId !== 'all' && selectedSetId !== '',
    retry: false,
  })

  const { data: compareTsData } = useQuery({
    queryKey: ['timeseries', projectId, compareRunId, queryParams],
    queryFn: () => resultsApi.getTimeseries(projectId, compareRunId!, queryParams),
    enabled: !!compareRunId,
  })

  const cellLabel = isUnstructured
    ? `L${layer + 1} N${node + 1}`
    : `L${layer + 1} R${row + 1} C${col + 1}`

  const chartHeight = expanded ? window.innerHeight - 160 : 350

  // Determine which observation data to use based on selection
  const activeObsData: ObservationData | undefined = useMemo(() => {
    if (selectedSetId === 'all') {
      return obsData
    }
    return selectedSetData
  }, [selectedSetId, obsData, selectedSetData])

  const handleWellSelect = (wellName: string) => {
    setSelectedWell(wellName)
    if (wellName && activeObsData?.wells[wellName]) {
      const well = activeObsData.wells[wellName]
      setLayer(well.layer)
      if (well.node !== undefined) {
        setNode(well.node)
      } else {
        if (well.row !== undefined) setRow(well.row)
        if (well.col !== undefined) setCol(well.col)
      }
    }
  }

  const wellNames = useMemo(() => {
    if (!activeObsData?.wells) return []
    return Object.keys(activeObsData.wells)
  }, [activeObsData])

  const hasObs = wellNames.length > 0
  const hasSets = observationSets.length > 0

  // Build observation overlay traces
  const obsTraces = useMemo(() => {
    if (!activeObsData?.wells) return []
    const traces: Plotly.Data[] = []
    let colorIdx = 0

    for (const [name, well] of Object.entries(activeObsData.wells)) {
      // Show if: well has no location (wide format), or location matches current cell
      const hasLocation = well.row !== undefined || well.node !== undefined
      let matches = !hasLocation // wide format → always show

      if (hasLocation) {
        if (isUnstructured) {
          matches = well.layer === layer && well.node === node
        } else {
          matches = well.layer === layer && well.row === row && well.col === col
        }
      }

      if (!matches) continue

      // Add set indicator to name if showing specific set
      const displayName = selectedSetId !== 'all'
        ? `${name}`
        : `Obs: ${name}`

      traces.push({
        x: well.times,
        y: well.heads,
        type: 'scatter',
        mode: 'markers',
        name: displayName,
        marker: {
          size: 8,
          symbol: 'diamond',
          color: OBS_COLORS[colorIdx % OBS_COLORS.length],
        },
      } as Plotly.Data)
      colorIdx++
    }
    return traces
  }, [activeObsData, layer, row, col, node, isUnstructured, selectedSetId])

  return (
    <div>
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <div className="flex items-center gap-1">
          <label className="text-xs text-slate-500">L:</label>
          <input
            type="number"
            min={1}
            max={metadata.nlay}
            value={layer + 1}
            onChange={(e) => setLayer(Math.max(0, Math.min(metadata.nlay - 1, Number(e.target.value) - 1)))}
            className="w-16 text-xs border border-slate-300 rounded px-2 py-1"
          />
        </div>
        {isUnstructured ? (
          <div className="flex items-center gap-1">
            <label className="text-xs text-slate-500">Node:</label>
            <input
              type="number"
              min={1}
              max={metadata.ncol}
              value={node + 1}
              onChange={(e) => setNode(Math.max(0, Math.min(metadata.ncol - 1, Number(e.target.value) - 1)))}
              className="w-20 text-xs border border-slate-300 rounded px-2 py-1"
            />
          </div>
        ) : (
          <>
            <div className="flex items-center gap-1">
              <label className="text-xs text-slate-500">R:</label>
              <input
                type="number"
                min={1}
                max={metadata.nrow}
                value={row + 1}
                onChange={(e) => setRow(Math.max(0, Math.min(metadata.nrow - 1, Number(e.target.value) - 1)))}
                className="w-16 text-xs border border-slate-300 rounded px-2 py-1"
              />
            </div>
            <div className="flex items-center gap-1">
              <label className="text-xs text-slate-500">C:</label>
              <input
                type="number"
                min={1}
                max={metadata.ncol}
                value={col + 1}
                onChange={(e) => setCol(Math.max(0, Math.min(metadata.ncol - 1, Number(e.target.value) - 1)))}
                className="w-16 text-xs border border-slate-300 rounded px-2 py-1"
              />
            </div>
          </>
        )}

        <div className="ml-auto flex items-center gap-2">
          {/* Observation set selector */}
          {hasSets && (
            <div className="relative">
              <button
                onClick={() => setShowSetDropdown(!showSetDropdown)}
                className="flex items-center gap-1.5 px-2 py-1 text-xs rounded border border-orange-300 text-orange-600 hover:bg-orange-50 transition-colors"
              >
                <FileSpreadsheet className="h-3.5 w-3.5" />
                <span>
                  {selectedSetId === 'all'
                    ? 'All Observations'
                    : observationSets.find(s => s.id === selectedSetId)?.name || 'Select Set'}
                </span>
                <ChevronDown className="h-3 w-3" />
              </button>

              {showSetDropdown && (
                <div className="absolute right-0 top-full mt-1 bg-white border border-slate-200 rounded-lg shadow-lg z-10 min-w-[180px]">
                  <button
                    onClick={() => {
                      setSelectedSetId('all')
                      setShowSetDropdown(false)
                    }}
                    className={`w-full text-left px-3 py-2 text-xs hover:bg-slate-50 ${
                      selectedSetId === 'all' ? 'bg-orange-50 text-orange-700' : ''
                    }`}
                  >
                    All Observations
                  </button>
                  {observationSets.map(set => (
                    <button
                      key={set.id}
                      onClick={() => {
                        setSelectedSetId(set.id)
                        setShowSetDropdown(false)
                      }}
                      className={`w-full text-left px-3 py-2 text-xs hover:bg-slate-50 border-t border-slate-100 ${
                        selectedSetId === set.id ? 'bg-orange-50 text-orange-700' : ''
                      }`}
                    >
                      <div className="font-medium">{set.name}</div>
                      <div className="text-slate-400">
                        {set.wells.length} wells, {set.n_observations} obs
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Well selector */}
          {hasObs && (
            <div className="flex items-center gap-1">
              <label className="text-xs text-slate-500">Well:</label>
              <select
                value={selectedWell}
                onChange={(e) => handleWellSelect(e.target.value)}
                className="text-xs border border-slate-300 rounded px-1.5 py-0.5"
              >
                <option value="">--</option>
                {wellNames.map(name => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </div>
          )}

          <ExportButton url={resultsApi.exportTimeseriesCsvUrl(projectId, runId, queryParams)} />
        </div>
      </div>

      {isLoading ? (
        <div className="h-72 flex items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
        </div>
      ) : isError ? (
        <div className="h-72 flex items-center justify-center text-red-400 text-sm">
          Error loading time series for this cell
        </div>
      ) : tsData ? (() => {
        const dates = timesToDates(
          summary.metadata.start_date,
          tsData.times,
          summary.metadata.time_unit,
        )
        const xValues = dates || tsData.times
        const compareDates = compareTsData
          ? timesToDates(summary.metadata.start_date, compareTsData.times, summary.metadata.time_unit)
          : null
        const compareXValues = compareTsData
          ? (compareDates || compareTsData.times)
          : undefined

        // Convert observation times to dates if applicable
        const datedObsTraces = obsTraces.map((trace: any) => {
          if (dates && trace.x) {
            const obsDates = timesToDates(summary.metadata.start_date, trace.x as number[], summary.metadata.time_unit)
            if (obsDates) return { ...trace, x: obsDates }
          }
          return trace
        })

        return (
          <Plot
            data={[
              {
                x: xValues,
                y: tsData.heads,
                type: 'scatter',
                mode: 'lines+markers',
                name: compareRunId ? `Run A ${cellLabel}` : `Simulated ${cellLabel}`,
                line: { color: '#3b82f6', width: 2 },
                marker: { size: 4 },
                connectgaps: false,
              },
              ...(compareTsData && compareXValues ? [{
                x: compareXValues,
                y: compareTsData.heads,
                type: 'scatter' as const,
                mode: 'lines+markers' as const,
                name: `Run B ${cellLabel}`,
                line: { color: '#f97316', width: 2 },
                marker: { size: 4 },
                connectgaps: false,
              }] : []),
              ...datedObsTraces,
            ]}
            layout={{
              autosize: true,
              height: chartHeight,
              margin: { l: 60, r: 20, t: 10, b: 50 },
              xaxis: {
                title: { text: dates ? 'Date' : timeLabel },
                type: dates ? 'date' : undefined,
              },
              yaxis: { title: { text: headLabel } },
              showlegend: true,
              legend: { orientation: 'h', y: 1.05 },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        )
      })()
       : (
        <div className="h-72 flex items-center justify-center text-slate-400">
          Select a cell to view its head time series
        </div>
      )}
    </div>
  )
}

export default memo(HeadTimeSeriesChart)
