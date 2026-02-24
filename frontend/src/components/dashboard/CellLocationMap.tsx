import { useMemo, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import DeckGL from '@deck.gl/react'
import { PolygonLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import { Map } from 'react-map-gl/maplibre'
import { Loader2 } from 'lucide-react'
import { resultsApi } from '../../services/api'
import type { ResultsSummary } from '../../types'
import { getProjection } from '../../utils/projection'
import {
  buildPolygonGridFeatures,
  buildStructuredGridFeatures,
  type CellFeature,
} from '../../utils/gridPolygons'
import { DEFAULT_BASEMAP } from '../../utils/basemaps'
import 'maplibre-gl/dist/maplibre-gl.css'

interface CellLocationMapProps {
  projectId: string
  runId: string
  summary: ResultsSummary
  layer: number
  row: number
  col: number
  node: number
  isUnstructured: boolean
  onCellSelect: (cell: { layer: number; row?: number; col?: number; node?: number }) => void
}

export default function CellLocationMap({
  projectId,
  runId,
  summary,
  layer,
  row,
  col,
  node,
  isUnstructured,
  onCellSelect,
}: CellLocationMapProps) {
  const { metadata } = summary
  const epsg = metadata.epsg
  const gridType = metadata.grid_type || 'structured'
  const isPolygonGrid = gridType === 'unstructured' || gridType === 'vertex'

  const converter = useMemo(() => {
    if (!epsg) return null
    return getProjection(epsg)
  }, [epsg])

  // Fetch grid geometry (polygon grids) — same cache key as HeadDeckView
  const { data: gridGeometry, isLoading: geomLoading } = useQuery({
    queryKey: ['grid-geometry', projectId, runId],
    queryFn: () => resultsApi.getGridGeometry(projectId, runId),
    enabled: isPolygonGrid,
    staleTime: Infinity,
  })

  // Fetch structured grid info — same cache key as HeadDeckView
  const { data: gridInfo, isLoading: gridInfoLoading } = useQuery({
    queryKey: ['structured-grid-info', projectId, runId],
    queryFn: () => resultsApi.getStructuredGridInfo(projectId, runId),
    enabled: !isPolygonGrid,
    staleTime: Infinity,
  })

  // Build cell features with dummy (all-null) head data — we only need polygons
  const { cellFeatures, mapViewState, orthoViewState } = useMemo(() => {
    if (isPolygonGrid && gridGeometry) {
      const layerKey = String(layer)
      const layerData = gridGeometry.layers[layerKey] || gridGeometry.layers['0']
      const ncpl = layerData?.polygons?.length || metadata.ncol
      const dummyData = [Array(ncpl).fill(null)]
      return buildPolygonGridFeatures({
        gridGeometry,
        headData: dummyData,
        layer,
        xoff: metadata.xoff || 0,
        yoff: metadata.yoff || 0,
        angrot: metadata.angrot || 0,
        converter,
      })
    }

    if (!isPolygonGrid) {
      const nrow = metadata.nrow || 1
      const ncol = metadata.ncol || 1
      const dummyData = Array.from({ length: nrow }, () => Array(ncol).fill(null))
      return buildStructuredGridFeatures({
        gridInfo: gridInfo ?? null,
        headData: dummyData,
        xoff: gridInfo?.xoff ?? (metadata.xoff || 0),
        yoff: gridInfo?.yoff ?? (metadata.yoff || 0),
        angrot: gridInfo?.angrot ?? (metadata.angrot || 0),
        converter,
      })
    }

    return { cellFeatures: [], mapViewState: null, orthoViewState: null }
  }, [gridGeometry, gridInfo, layer, metadata, converter, isPolygonGrid])

  // Find the highlighted cell feature
  const highlightFeature = useMemo(() => {
    if (cellFeatures.length === 0) return null
    if (isUnstructured) {
      return cellFeatures.find((f) => f.cellIndex === node) || null
    }
    return cellFeatures.find((f) => f.row === row && f.col === col) || null
  }, [cellFeatures, row, col, node, isUnstructured])

  // Center view on the highlighted cell (new object ref triggers deck.gl update)
  const centeredMapViewState = useMemo(() => {
    if (!mapViewState) return null
    if (!highlightFeature) return mapViewState
    const [lng, lat] = highlightFeature.centroid
    return { ...mapViewState, longitude: lng, latitude: lat }
  }, [mapViewState, highlightFeature])

  const centeredOrthoViewState = useMemo(() => {
    if (!orthoViewState) return null
    if (!highlightFeature) return orthoViewState
    const [cx, cy] = highlightFeature.centroid
    return { ...orthoViewState, target: [cx, cy, 0] as [number, number, number] }
  }, [orthoViewState, highlightFeature])

  // Click handler
  const handleClick = useCallback(
    (info: any) => {
      const obj = info.object as CellFeature | undefined
      if (!obj) return
      if (isUnstructured) {
        onCellSelect({ layer, node: obj.cellIndex })
      } else {
        onCellSelect({ layer, row: obj.row, col: obj.col })
      }
    },
    [layer, isUnstructured, onCellSelect],
  )

  // Tooltip
  const getTooltip = useCallback(
    ({ object }: { object?: CellFeature }) => {
      if (!object) return null
      const label =
        object.row !== undefined
          ? `R${object.row + 1} C${object.col! + 1}`
          : `Cell ${object.cellIndex + 1}`
      return {
        html: `<div style="padding:2px 6px;font-size:11px">${label}</div>`,
        style: {
          backgroundColor: 'rgba(0,0,0,0.75)',
          color: '#fff',
          borderRadius: '3px',
        },
      }
    },
    [],
  )

  const isLoading = isPolygonGrid ? geomLoading : gridInfoLoading

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
      </div>
    )
  }

  if (cellFeatures.length === 0 || (!centeredMapViewState && !centeredOrthoViewState)) {
    return (
      <div className="h-full flex items-center justify-center text-slate-400 text-xs">
        No grid data
      </div>
    )
  }

  // Base grid layer — light gray fill, thin outline, pickable
  const baseLayer = new PolygonLayer<CellFeature>({
    id: 'cell-map-base',
    data: cellFeatures,
    getPolygon: (d) => d.polygon,
    getFillColor: [230, 230, 230, 120],
    getLineColor: [160, 160, 160, 100],
    getLineWidth: 1,
    lineWidthMinPixels: 0.5,
    pickable: true,
    autoHighlight: true,
    highlightColor: [59, 130, 246, 60],
    onClick: handleClick,
  })

  // Highlight layer — selected cell in blue
  const highlightLayer = highlightFeature
    ? new PolygonLayer<CellFeature>({
        id: 'cell-map-highlight',
        data: [highlightFeature],
        getPolygon: (d) => d.polygon,
        getFillColor: [59, 130, 246, 180],
        getLineColor: [255, 255, 255, 255],
        getLineWidth: 2,
        lineWidthMinPixels: 2,
        pickable: false,
      })
    : null

  const layers = highlightLayer ? [baseLayer, highlightLayer] : [baseLayer]

  return (
    <div className="h-full w-full relative">
      {centeredMapViewState ? (
        <DeckGL
          initialViewState={centeredMapViewState}
          controller={true}
          layers={layers}
          getTooltip={getTooltip as any}
        >
          <Map mapStyle={DEFAULT_BASEMAP.style as any} />
        </DeckGL>
      ) : centeredOrthoViewState ? (
        <DeckGL
          views={[new OrthographicView({ id: 'ortho', flipY: false })]}
          initialViewState={{ ortho: centeredOrthoViewState }}
          controller={true}
          layers={layers}
          getTooltip={getTooltip as any}
        />
      ) : null}
    </div>
  )
}
