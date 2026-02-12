import { useState, useMemo, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import DeckGL from '@deck.gl/react'
import { PolygonLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import { Map } from 'react-map-gl/maplibre'
import { Loader2, Layers } from 'lucide-react'
import { resultsApi } from '../../services/api'
import type { ResultsSummary } from '../../types'
import { getProjection } from '../../utils/projection'
import {
  valueToColor,
  diffValueToColor,
  VIRIDIS_GRADIENT_CSS,
  DIFF_GRADIENT_CSS,
} from '../../utils/colorScales'
import {
  buildPolygonGridFeatures,
  buildStructuredGridFeatures,
  flattenHeadData,
  isNoData,
  type CellFeature,
} from '../../utils/gridPolygons'
import { lengthAbbrev, labelWithUnit } from '../../utils/unitLabels'
import { BASEMAPS, DEFAULT_BASEMAP, type BasemapOption } from '../../utils/basemaps'
import 'maplibre-gl/dist/maplibre-gl.css'

interface HeadDeckViewProps {
  projectId: string
  runId: string
  summary: ResultsSummary
  layer: number
  kper: number
  kstp: number
  expanded?: boolean
  compareRunId?: string | null
  compareSummary?: ResultsSummary
}

export default function HeadDeckView({
  projectId,
  runId,
  summary,
  layer,
  kper,
  kstp,
  expanded = false,
  compareRunId,
}: HeadDeckViewProps) {
  const { metadata, heads_summary } = summary
  const epsg = metadata.epsg
  const gridType = metadata.grid_type || 'structured'
  const isPolygonGrid = gridType === 'unstructured' || gridType === 'vertex'
  const lu = lengthAbbrev(metadata.length_unit)

  const [opacity, setOpacity] = useState(180)
  const [basemap, setBasemap] = useState<BasemapOption>(DEFAULT_BASEMAP)
  const [showBasemapPicker, setShowBasemapPicker] = useState(false)

  // Build projection converter
  const converter = useMemo(() => {
    if (!epsg) return null
    return getProjection(epsg)
  }, [epsg])

  // Fetch grid geometry (polygon grids)
  const { data: gridGeometry, isLoading: geomLoading } = useQuery({
    queryKey: ['grid-geometry', projectId, runId],
    queryFn: () => resultsApi.getGridGeometry(projectId, runId),
    enabled: isPolygonGrid,
    staleTime: Infinity,
  })

  // Fetch structured grid info
  const { data: gridInfo, isLoading: gridInfoLoading } = useQuery({
    queryKey: ['structured-grid-info', projectId, runId],
    queryFn: () => resultsApi.getStructuredGridInfo(projectId, runId),
    enabled: !isPolygonGrid,
    staleTime: Infinity,
  })

  // Fetch head slice
  const { data: headSlice, isLoading: headLoading } = useQuery({
    queryKey: ['heads', projectId, runId, layer, kper, kstp],
    queryFn: () => resultsApi.getHeads(projectId, runId, layer, kper, kstp),
  })

  // Fetch comparison head slice for diff mode
  const { data: compareSlice, isLoading: compareLoading } = useQuery({
    queryKey: ['heads', projectId, compareRunId, layer, kper, kstp],
    queryFn: () => resultsApi.getHeads(projectId, compareRunId!, layer, kper, kstp),
    enabled: !!compareRunId,
  })

  // Build cell features
  const { cellFeatures, mapViewState, orthoViewState } = useMemo(() => {
    if (!headSlice) return { cellFeatures: [], mapViewState: null, orthoViewState: null }

    if (isPolygonGrid && gridGeometry) {
      return buildPolygonGridFeatures({
        gridGeometry,
        headData: headSlice.data,
        layer,
        xoff: metadata.xoff || 0,
        yoff: metadata.yoff || 0,
        angrot: metadata.angrot || 0,
        converter,
      })
    }

    if (!isPolygonGrid) {
      return buildStructuredGridFeatures({
        gridInfo: gridInfo ?? null,
        headData: headSlice.data,
        xoff: gridInfo?.xoff ?? (metadata.xoff || 0),
        yoff: gridInfo?.yoff ?? (metadata.yoff || 0),
        angrot: gridInfo?.angrot ?? (metadata.angrot || 0),
        converter,
      })
    }

    return { cellFeatures: [], mapViewState: null, orthoViewState: null }
  }, [headSlice, gridGeometry, gridInfo, layer, metadata, converter, isPolygonGrid])

  // Compute per-cell diff values when in comparison mode
  const { diffValues, diffAbsMax } = useMemo(() => {
    if (!compareRunId || !headSlice || !compareSlice) return { diffValues: null, diffAbsMax: 0 }

    const flatA = flattenHeadData(headSlice.data)
    const flatB = flattenHeadData(compareSlice.data)
    const diffs: (number | null)[] = []
    let absMax = 0

    for (let i = 0; i < Math.max(flatA.length, flatB.length); i++) {
      const a = i < flatA.length ? flatA[i] : null
      const b = i < flatB.length ? flatB[i] : null
      if (isNoData(a) || isNoData(b)) {
        diffs.push(null)
      } else {
        const d = (a as number) - (b as number)
        diffs.push(d)
        if (Math.abs(d) > absMax) absMax = Math.abs(d)
      }
    }
    if (absMax < 0.01) absMax = 0.01
    return { diffValues: diffs, diffAbsMax: absMax }
  }, [compareRunId, headSlice, compareSlice])

  const isDiff = !!compareRunId && !!diffValues
  const vmin = heads_summary.min_head ?? 0
  const vmax = heads_summary.max_head ?? 1

  // Tooltip
  const getTooltip = useCallback(
    ({ object }: { object?: CellFeature }) => {
      if (!object || object.value === null) return null

      const label = object.row !== undefined
        ? `R${object.row + 1} C${object.col! + 1}`
        : `Cell ${object.cellIndex}`

      let valueStr: string
      if (isDiff && diffValues) {
        const d = diffValues[object.cellIndex]
        valueStr = d !== null ? `Diff: ${d.toFixed(3)}${lu ? ' ' + lu : ''}` : 'N/A'
      } else {
        valueStr = `Head: ${object.value.toFixed(3)}${lu ? ' ' + lu : ''}`
      }

      return {
        html: `<div style="padding:4px 8px"><b>${label}</b><br/>${valueStr}</div>`,
        style: {
          backgroundColor: 'rgba(0,0,0,0.8)',
          color: '#fff',
          fontSize: '12px',
          borderRadius: '4px',
        },
      }
    },
    [isDiff, diffValues, lu],
  )

  // Loading
  const isLoading = headLoading || (isPolygonGrid ? geomLoading : gridInfoLoading) || (compareRunId ? compareLoading : false)

  if (isLoading) {
    return (
      <div className="h-96 flex items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
        <span className="ml-2 text-sm text-slate-500">Loading map data...</span>
      </div>
    )
  }

  if (!mapViewState && !orthoViewState) {
    return (
      <div className="h-96 flex items-center justify-center text-slate-400">
        No mappable data available
      </div>
    )
  }

  if (cellFeatures.length === 0) {
    return (
      <div className="h-96 flex items-center justify-center text-slate-400">
        No mappable data available
      </div>
    )
  }

  // Build polygon layer
  const polygonLayer = new PolygonLayer<CellFeature>({
    id: 'head-deck-contour',
    data: cellFeatures,
    getPolygon: (d) => d.polygon,
    getFillColor: (d) => {
      if (d.value === null) return [200, 200, 200, 40] as [number, number, number, number]
      if (isDiff && diffValues) {
        const diff = diffValues[d.cellIndex]
        if (diff === null) return [200, 200, 200, 40] as [number, number, number, number]
        return diffValueToColor(diff, diffAbsMax, opacity)
      }
      return valueToColor(d.value, vmin, vmax, opacity)
    },
    getLineColor: [100, 100, 100, 50],
    getLineWidth: 1,
    lineWidthMinPixels: 0.5,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 80],
    updateTriggers: {
      getFillColor: [opacity, isDiff, diffValues, diffAbsMax, vmin, vmax],
    },
  })

  const mapHeight = expanded ? 'calc(100vh - 160px)' : '500px'
  const legendLabel = isDiff ? labelWithUnit('Head Diff (A-B)', lu) : labelWithUnit('Head', lu)
  const legendGradient = isDiff ? DIFF_GRADIENT_CSS : VIRIDIS_GRADIENT_CSS
  const legendMin = isDiff ? (-diffAbsMax).toFixed(1) : vmin?.toFixed(1)
  const legendMax = isDiff ? diffAbsMax.toFixed(1) : vmax?.toFixed(1)

  return (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <label className="text-xs text-slate-500">Opacity:</label>
        <input
          type="range"
          min={50}
          max={255}
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="w-24 h-1 accent-blue-500"
        />
      </div>

      <div style={{ height: mapHeight, position: 'relative' }}>
        {mapViewState ? (
          <DeckGL
            initialViewState={mapViewState}
            controller={true}
            layers={[polygonLayer]}
            getTooltip={getTooltip as any}
          >
            <Map mapStyle={basemap.style as any} />
          </DeckGL>
        ) : orthoViewState ? (
          <DeckGL
            views={[new OrthographicView({ id: 'ortho', flipY: false })]}
            initialViewState={{ ortho: orthoViewState }}
            controller={true}
            layers={[polygonLayer]}
            getTooltip={getTooltip as any}
          />
        ) : null}

        {/* Basemap picker (only for geo-referenced models) */}
        {mapViewState && (
          <div className="absolute top-3 right-3 z-10">
            <button
              onClick={() => setShowBasemapPicker(!showBasemapPicker)}
              className="bg-white/90 hover:bg-white rounded-lg p-1.5 shadow-md border border-slate-200 transition-colors"
              title="Change basemap"
            >
              <Layers className="h-4 w-4 text-slate-600" />
            </button>
            {showBasemapPicker && (
              <div className="absolute top-9 right-0 bg-white rounded-lg shadow-lg border border-slate-200 p-2 flex gap-2">
                {BASEMAPS.map((bm) => (
                  <button
                    key={bm.id}
                    onClick={() => { setBasemap(bm); setShowBasemapPicker(false) }}
                    className={`flex flex-col items-center gap-1 p-1 rounded transition-all ${
                      basemap.id === bm.id
                        ? 'ring-2 ring-blue-500 ring-offset-1'
                        : 'hover:bg-slate-50'
                    }`}
                    title={bm.label}
                  >
                    <div
                      className="w-14 h-10 rounded border border-slate-300"
                      style={{ background: bm.thumbnail }}
                    />
                    <span className="text-[10px] text-slate-600 font-medium">{bm.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Color scale legend */}
        <div className="absolute bottom-4 right-4 bg-white/90 rounded-lg p-2 shadow-md">
          <div className="text-xs text-slate-600 mb-1 font-medium">{legendLabel}</div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-slate-500">{legendMin}</span>
            <div
              className="w-24 h-3 rounded"
              style={{ background: legendGradient }}
            />
            <span className="text-xs text-slate-500">{legendMax}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
