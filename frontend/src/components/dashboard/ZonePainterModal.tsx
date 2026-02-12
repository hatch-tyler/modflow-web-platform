import { useState, useCallback, useMemo, useEffect, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import DeckGL from '@deck.gl/react'
import { PolygonLayer, PathLayer } from '@deck.gl/layers'
import { OrthographicView } from '@deck.gl/core'
import { Map } from 'react-map-gl/maplibre'
import { Loader2, X, MousePointer2, Square, Pentagon, Plus, Minus, AlertTriangle, Save, FolderOpen, Zap, Calculator } from 'lucide-react'
import { resultsApi, zoneBudgetApi, zoneDefinitionsApi } from '../../services/api'
import type { ResultsSummary, ZoneBudgetProgress } from '../../types'
import { MAX_ZONES, zoneColor, type LayerZoneAssignments } from '../../utils/zoneColors'
import { pointInPolygon } from '../../utils/pointInPolygon'
import { getProjection } from '../../utils/projection'
import {
  buildPolygonGridFeatures,
  buildStructuredGridFeatures,
  type CellFeature,
} from '../../utils/gridPolygons'
import { DEFAULT_BASEMAP } from '../../utils/basemaps'
import 'maplibre-gl/dist/maplibre-gl.css'

type Tool = 'click' | 'rect' | 'polygon'

interface ZonePainterModalProps {
  projectId: string
  runId: string
  summary: ResultsSummary
  zoneAssignments: LayerZoneAssignments
  onZoneAssignmentsChange: (a: LayerZoneAssignments) => void
  onComputeResult: (result: any) => void
  onClose: () => void
}

function hslToRgba(hsl: string, alpha: number): [number, number, number, number] {
  // Parse "hsl(H, S%, L%)" or "hsla(H, S%, L%, A)"
  const m = hsl.match(/hsla?\(\s*([\d.]+),\s*([\d.]+)%?,\s*([\d.]+)%?/)
  if (!m) return [128, 128, 128, alpha]
  const h = Number(m[1]) / 360
  const s = Number(m[2]) / 100
  const l = Number(m[3]) / 100

  let r: number, g: number, b: number
  if (s === 0) {
    r = g = b = l
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1
      if (t > 1) t -= 1
      if (t < 1 / 6) return p + (q - p) * 6 * t
      if (t < 1 / 2) return q
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
      return p
    }
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s
    const p = 2 * l - q
    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), alpha]
}

export default function ZonePainterModal({
  projectId,
  runId,
  summary,
  zoneAssignments,
  onZoneAssignmentsChange,
  onComputeResult,
  onClose,
}: ZonePainterModalProps) {
  const { metadata, heads_summary } = summary
  const epsg = metadata.epsg
  const gridType = metadata.grid_type || 'structured'
  const isPolygonGrid = gridType === 'unstructured' || gridType === 'vertex'

  const kstpkperList = heads_summary.kstpkper_list || []
  const lastIdx = kstpkperList.length - 1
  const kstp = kstpkperList[lastIdx]?.[0] ?? 0
  const kper = kstpkperList[lastIdx]?.[1] ?? 0

  // Internal state (previously in WaterBudgetChart)
  const [activeZone, setActiveZone] = useState(1)
  const [numZones, setNumZones] = useState(6)
  const [applyAllLayers, setApplyAllLayers] = useState(false)
  const [computing, setComputing] = useState(false)

  const [layer, setLayer] = useState(0)
  const [tool, setTool] = useState<Tool>('click')
  const [polygonVertices, setPolygonVertices] = useState<[number, number][]>([])
  const [cursorPos, setCursorPos] = useState<[number, number] | null>(null)
  const [lastVertexPixel, setLastVertexPixel] = useState<[number, number] | null>(null)
  const [rectStart, setRectStart] = useState<[number, number] | null>(null)
  const [deckKey, setDeckKey] = useState(0)

  // Async compute state
  const [taskId, setTaskId] = useState<string | null>(null)
  const [computeProgress, setComputeProgress] = useState<ZoneBudgetProgress | null>(null)
  const [computeError, setComputeError] = useState<string | null>(null)
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Save/load zone definitions
  const [saveName, setSaveName] = useState('')
  const [showSaveInput, setShowSaveInput] = useState(false)
  const [showLoadDropdown, setShowLoadDropdown] = useState(false)
  const queryClient = useQueryClient()

  // Keep activeZone within numZones
  useEffect(() => {
    if (activeZone > numZones) setActiveZone(numZones)
  }, [numZones, activeZone])

  const currentLayerZones = zoneAssignments[layer] || {}

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (polygonVertices.length > 0) {
          setPolygonVertices([])
          setLastVertexPixel(null)
        } else if (rectStart) {
          setRectStart(null)
        } else {
          onClose()
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, polygonVertices.length, rectStart])

  // Fetch head slice
  const { data: headSlice, isLoading: headLoading, isError: headError, refetch: refetchHeads } = useQuery({
    queryKey: ['heads', projectId, runId, layer, kper, kstp],
    queryFn: () => resultsApi.getHeads(projectId, runId, layer, kper, kstp),
    enabled: kstpkperList.length > 0,
    retry: 2,
    retryDelay: 2000,
  })

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

  // Build projection converter
  const converter = useMemo(() => {
    if (!epsg) return null
    return getProjection(epsg)
  }, [epsg])

  const hasProjection = !!converter && !!epsg

  // Build cell features using shared grid polygon builders
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

  // Set of active cell indices (cells with valid head data)
  const activeCellIndices = useMemo(() => {
    const s = new Set<number>()
    for (const c of cellFeatures) {
      if (c.value !== null) s.add(c.cellIndex)
    }
    return s
  }, [cellFeatures])

  // Paint a single cell (click tool)
  const paintCell = useCallback((cellIdx: number, toggle: boolean) => {
    if (!activeCellIndices.has(cellIdx)) return
    const targetLayers = applyAllLayers
      ? Array.from({ length: metadata.nlay }, (_, i) => i)
      : [layer]

    const updated = { ...zoneAssignments }
    for (const lay of targetLayers) {
      const layerZones = { ...(updated[lay] || {}) }
      if (toggle && layerZones[cellIdx] === activeZone) {
        delete layerZones[cellIdx]
      } else {
        layerZones[cellIdx] = activeZone
      }
      updated[lay] = layerZones
    }
    onZoneAssignmentsChange(updated)
  }, [activeZone, layer, applyAllLayers, metadata.nlay, zoneAssignments, onZoneAssignmentsChange, activeCellIndices])

  // Assign all active cells inside polygon
  const assignPolygon = useCallback((vertices: [number, number][]) => {
    if (vertices.length < 3) return
    const targetLayers = applyAllLayers
      ? Array.from({ length: metadata.nlay }, (_, i) => i)
      : [layer]

    const updated = { ...zoneAssignments }
    for (const lay of targetLayers) {
      const layerZones = { ...(updated[lay] || {}) }
      for (const cell of cellFeatures) {
        if (cell.value !== null && pointInPolygon(cell.centroid, vertices)) {
          layerZones[cell.cellIndex] = activeZone
        }
      }
      updated[lay] = layerZones
    }
    onZoneAssignmentsChange(updated)
  }, [cellFeatures, activeZone, layer, applyAllLayers, metadata.nlay, zoneAssignments, onZoneAssignmentsChange])

  // Assign all active cells inside a rectangle (defined by two corners)
  const assignRect = useCallback((corner1: [number, number], corner2: [number, number]) => {
    const minX = Math.min(corner1[0], corner2[0])
    const maxX = Math.max(corner1[0], corner2[0])
    const minY = Math.min(corner1[1], corner2[1])
    const maxY = Math.max(corner1[1], corner2[1])

    const targetLayers = applyAllLayers
      ? Array.from({ length: metadata.nlay }, (_, i) => i)
      : [layer]

    const updated = { ...zoneAssignments }
    for (const lay of targetLayers) {
      const layerZones = { ...(updated[lay] || {}) }
      for (const cell of cellFeatures) {
        if (cell.value === null) continue
        const [cx, cy] = cell.centroid
        if (cx >= minX && cx <= maxX && cy >= minY && cy <= maxY) {
          layerZones[cell.cellIndex] = activeZone
        }
      }
      updated[lay] = layerZones
    }
    onZoneAssignmentsChange(updated)
  }, [cellFeatures, activeZone, layer, applyAllLayers, metadata.nlay, zoneAssignments, onZoneAssignmentsChange])

  // Close polygon and assign zone
  const closePolygon = useCallback(() => {
    if (polygonVertices.length >= 3) {
      assignPolygon(polygonVertices)
      setPolygonVertices([])
      setCursorPos(null)
      setLastVertexPixel(null)
    }
  }, [polygonVertices, assignPolygon])

  const totalAssignedAll = Object.values(zoneAssignments).reduce(
    (sum, layZones) => sum + Object.keys(layZones).length, 0,
  )

  // Copy current layer's zones to all other layers
  const copyToAllLayers = useCallback(() => {
    const srcLayer = Object.entries(zoneAssignments).find(([, z]) => Object.keys(z).length > 0)
    if (!srcLayer) return
    const src = srcLayer[1]
    const updated = { ...zoneAssignments }
    for (let i = 0; i < metadata.nlay; i++) {
      updated[i] = { ...src }
    }
    onZoneAssignmentsChange(updated)
  }, [zoneAssignments, metadata.nlay, onZoneAssignmentsChange])

  // Build zone layers payload from assignments
  const buildZoneLayers = useCallback(() => {
    const zoneLayers: Record<string, Record<string, number[]>> = {}
    for (const [layStr, layerZonesData] of Object.entries(zoneAssignments)) {
      const groups: Record<string, number[]> = {}
      for (const [cellStr, zoneNum] of Object.entries(layerZonesData)) {
        const name = `Zone ${zoneNum}`
        if (!groups[name]) groups[name] = []
        groups[name].push(Number(cellStr))
      }
      if (Object.keys(groups).length > 0) {
        zoneLayers[layStr] = groups
      }
    }
    return zoneLayers
  }, [zoneAssignments])

  // Poll for task completion
  const pollTaskStatus = useCallback((tid: string) => {
    const poll = async () => {
      try {
        const progress = await zoneBudgetApi.getStatus(projectId, runId, tid)
        setComputeProgress(progress)

        if (progress.status === 'completed') {
          // Fetch result
          const result = await zoneBudgetApi.getResult(projectId, runId, tid)
          setComputing(false)
          setTaskId(null)
          setComputeProgress(null)
          onComputeResult(result)
          onClose()
          return
        }

        if (progress.status === 'failed') {
          setComputeError(progress.error || 'Computation failed')
          setComputing(false)
          setTaskId(null)
          return
        }

        // Continue polling
        pollTimerRef.current = setTimeout(poll, 2000)
      } catch (err: any) {
        // If status endpoint returns 404, the task was lost (Redis key expired)
        if (err?.response?.status === 404) {
          setComputeError('Task was lost — the worker may not have received it. Please retry.')
          setComputing(false)
          setTaskId(null)
          return
        }
        // Retry on transient network error
        pollTimerRef.current = setTimeout(poll, 3000)
      }
    }
    poll()
  }, [projectId, runId, onComputeResult, onClose])

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
    }
  }, [])

  const handleComputeZoneBudget = async (quickMode = false) => {
    const zoneLayers = buildZoneLayers()
    if (Object.keys(zoneLayers).length === 0) return

    setComputing(true)
    setComputeError(null)
    setComputeProgress(null)

    try {
      const response = await zoneBudgetApi.compute(projectId, runId, zoneLayers, quickMode)

      if (response.status === 'completed' && response.result) {
        // Cache hit — immediate result
        setComputing(false)
        onComputeResult(response.result)
        onClose()
        return
      }

      if (response.status === 'queued' && response.task_id) {
        setTaskId(response.task_id)
        setComputeProgress({ status: 'queued', progress: 0, message: 'Queued for computation...' })
        pollTaskStatus(response.task_id)
        return
      }

      // Unexpected response
      setComputing(false)
    } catch (err) {
      console.error('Zone budget computation failed:', err)
      setComputeError(err instanceof Error ? err.message : 'Computation failed')
      setComputing(false)
    }
  }

  // Fetch saved zone definitions
  const { data: savedDefs, refetch: refetchDefs } = useQuery({
    queryKey: ['zone-definitions', projectId],
    queryFn: () => zoneDefinitionsApi.list(projectId),
    staleTime: 30_000,
  })

  const handleSaveZones = async () => {
    if (!saveName.trim()) return
    const zoneLayers = buildZoneLayers()
    if (Object.keys(zoneLayers).length === 0) return

    try {
      await zoneDefinitionsApi.save(projectId, {
        name: saveName.trim(),
        zone_layers: zoneLayers,
        num_zones: numZones,
      })
      setShowSaveInput(false)
      setSaveName('')
      refetchDefs()
      queryClient.invalidateQueries({ queryKey: ['zone-definitions', projectId] })
    } catch (err) {
      console.error('Failed to save zone definition:', err)
    }
  }

  const handleLoadZones = async (name: string) => {
    setShowLoadDropdown(false)
    try {
      const defn = await zoneDefinitionsApi.get(projectId, name)
      // Convert API format back to LayerZoneAssignments
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
      onZoneAssignmentsChange(loaded)
      if (defn.num_zones) setNumZones(defn.num_zones)
    } catch (err) {
      console.error('Failed to load zone definition:', err)
    }
  }

  // Controller config per tool
  const controllerConfig = useMemo(() => {
    if (tool === 'polygon') return { dragPan: false, scrollZoom: true, doubleClickZoom: false }
    // click and rect tools use click-based interaction; allow panning via drag
    return { dragPan: true, scrollZoom: true, doubleClickZoom: false }
  }, [tool])

  // DeckGL onClick
  const handleDeckClick = useCallback((info: any, event: any) => {
    if (tool === 'polygon') {
      const coord = info.coordinate
      if (!coord) return
      const pt: [number, number] = [coord[0], coord[1]]
      const pixel: [number, number] | null = info.pixel ? [info.pixel[0], info.pixel[1]] : null

      // Double-click closes polygon
      if (event?.srcEvent?.detail >= 2 && polygonVertices.length >= 2) {
        assignPolygon([...polygonVertices, pt])
        setPolygonVertices([])
        setCursorPos(null)
        setLastVertexPixel(null)
        return
      }
      setPolygonVertices(prev => [...prev, pt])
      setLastVertexPixel(pixel)
      return
    }

    if (tool === 'rect') {
      const coord = info.coordinate
      if (!coord) return
      const pt: [number, number] = [coord[0], coord[1]]
      if (!rectStart) {
        // First click: set starting corner
        setRectStart(pt)
      } else {
        // Second click: assign cells within the rectangle
        assignRect(rectStart, pt)
        setRectStart(null)
        setCursorPos(null)
      }
      return
    }

    if (tool === 'click') {
      const obj = info.object as CellFeature | undefined
      if (!obj) return
      paintCell(obj.cellIndex, true)
    }
  }, [tool, polygonVertices, paintCell, assignPolygon, rectStart, assignRect])

  // DeckGL onHover — for polygon/rect preview cursor tracking
  const handleDeckHover = useCallback((info: any) => {
    if ((tool === 'polygon' || (tool === 'rect' && rectStart)) && info.coordinate) {
      setCursorPos([info.coordinate[0], info.coordinate[1]])
    }
  }, [tool, rectStart])

  // Build layers
  const layers = useMemo(() => {
    const result: any[] = []

    // 1. Grid background — white fill with visible borders
    result.push(
      new PolygonLayer<CellFeature>({
        id: 'grid-bg',
        data: cellFeatures,
        getPolygon: (d) => d.polygon,
        getFillColor: (d) => d.value !== null ? [255, 255, 255, 230] as [number, number, number, number] : [220, 220, 220, 40] as [number, number, number, number],
        getLineColor: [140, 150, 160, 140],
        getLineWidth: 1,
        lineWidthMinPixels: 0.5,
        pickable: tool !== 'polygon',
        autoHighlight: tool === 'click',
        highlightColor: [200, 220, 255, 100],
      }),
    )

    // 2. Zone overlay layer
    const zonedCells = cellFeatures.filter(c => currentLayerZones[c.cellIndex] !== undefined)
    if (zonedCells.length > 0) {
      result.push(
        new PolygonLayer<CellFeature & { zone: number }>({
          id: 'zone-overlay',
          data: zonedCells.map(c => ({ ...c, zone: currentLayerZones[c.cellIndex] })),
          getPolygon: (d) => d.polygon,
          getFillColor: (d) => {
            const zc = zoneColor(d.zone - 1, numZones)
            return hslToRgba(zc.border, 160)
          },
          getLineColor: (d) => {
            const zc = zoneColor(d.zone - 1, numZones)
            return hslToRgba(zc.border, 230)
          },
          getLineWidth: 2,
          lineWidthMinPixels: 1,
          pickable: false,
          updateTriggers: {
            getFillColor: [numZones, currentLayerZones],
            getLineColor: [numZones, currentLayerZones],
          },
        }),
      )
    }

    // 3. Polygon preview — highlight cells that would be assigned
    if (tool === 'polygon' && polygonVertices.length >= 2 && cursorPos) {
      const previewPoly = [...polygonVertices, cursorPos]
      const previewCells = cellFeatures.filter(c =>
        currentLayerZones[c.cellIndex] === undefined && pointInPolygon(c.centroid, previewPoly)
      )
      if (previewCells.length > 0) {
        const azc = zoneColor(activeZone - 1, numZones)
        result.push(
          new PolygonLayer<CellFeature>({
            id: 'polygon-preview',
            data: previewCells,
            getPolygon: (d) => d.polygon,
            getFillColor: hslToRgba(azc.border, 80),
            getLineColor: hslToRgba(azc.border, 150),
            getLineWidth: 1,
            lineWidthMinPixels: 0.5,
            pickable: false,
            updateTriggers: {
              getFillColor: [activeZone, numZones],
              getLineColor: [activeZone, numZones],
            },
          }),
        )
      }
    }

    // 4. Rectangle preview — highlight cells inside the rect and show outline
    if (tool === 'rect' && rectStart && cursorPos) {
      const rMinX = Math.min(rectStart[0], cursorPos[0])
      const rMaxX = Math.max(rectStart[0], cursorPos[0])
      const rMinY = Math.min(rectStart[1], cursorPos[1])
      const rMaxY = Math.max(rectStart[1], cursorPos[1])

      const rectCells = cellFeatures.filter(c => {
        if (c.value === null) return false
        const [cx, cy] = c.centroid
        return cx >= rMinX && cx <= rMaxX && cy >= rMinY && cy <= rMaxY
      })
      if (rectCells.length > 0) {
        const azc = zoneColor(activeZone - 1, numZones)
        result.push(
          new PolygonLayer<CellFeature>({
            id: 'rect-preview',
            data: rectCells,
            getPolygon: (d) => d.polygon,
            getFillColor: hslToRgba(azc.border, 80),
            getLineColor: hslToRgba(azc.border, 150),
            getLineWidth: 1,
            lineWidthMinPixels: 0.5,
            pickable: false,
            updateTriggers: {
              getFillColor: [activeZone, numZones],
              getLineColor: [activeZone, numZones],
            },
          }),
        )
      }

      // Rectangle outline
      const rectOutline: [number, number][] = [
        [rMinX, rMinY], [rMaxX, rMinY], [rMaxX, rMaxY], [rMinX, rMaxY], [rMinX, rMinY],
      ]
      result.push(
        new PathLayer({
          id: 'rect-outline',
          data: [{ path: rectOutline }],
          getPath: (d: any) => d.path,
          getColor: [255, 100, 50, 200],
          getWidth: 2,
          widthMinPixels: 2,
          pickable: false,
        }),
      )
    }

    // 5. Drawing polygon path
    if (tool === 'polygon' && polygonVertices.length > 0) {
      const pathData: [number, number][] = [...polygonVertices]
      if (cursorPos) pathData.push(cursorPos)

      result.push(
        new PathLayer({
          id: 'drawing-polygon',
          data: [{ path: pathData }],
          getPath: (d: any) => d.path,
          getColor: [255, 100, 50, 200],
          getWidth: 3,
          widthMinPixels: 2,
          pickable: false,
        }),
      )

      // Vertex dots
      result.push(
        new PolygonLayer({
          id: 'polygon-vertices',
          data: polygonVertices.map(v => ({
            polygon: createCircle(v, hasProjection ? 0.00003 : getVertexRadius(cellFeatures)),
          })),
          getPolygon: (d: any) => d.polygon,
          getFillColor: [255, 100, 50, 255],
          getLineColor: [255, 255, 255, 255],
          getLineWidth: 1,
          lineWidthMinPixels: 1,
          pickable: false,
        }),
      )
    }

    return result
  }, [cellFeatures, currentLayerZones, numZones, activeZone, tool, polygonVertices, cursorPos, hasProjection, rectStart])

  const isLoading = headLoading || (isPolygonGrid ? geomLoading : gridInfoLoading)
  const layerAssigned = Object.keys(currentLayerZones).length

  const toolNames: { key: Tool; label: string; icon: typeof MousePointer2 }[] = [
    { key: 'click', label: 'Click', icon: MousePointer2 },
    { key: 'rect', label: 'Rectangle', icon: Square },
    { key: 'polygon', label: 'Polygon', icon: Pentagon },
  ]

  const toolHint = tool === 'click' ? 'Click cells to toggle zone'
    : tool === 'rect' ? (rectStart ? 'Click to set second corner and select cells' : 'Click to set first corner of rectangle')
    : polygonVertices.length === 0 ? 'Click to place vertices'
    : polygonVertices.length < 3 ? `${polygonVertices.length} vertices — place at least 3`
    : `${polygonVertices.length} vertices — double-click or use Close Polygon button`

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center">
      <div className="bg-white rounded-xl shadow-2xl flex flex-col" style={{ width: '95vw', height: '95vh', maxWidth: '95vw', maxHeight: '95vh' }}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-800">Zone Budget</h2>
          <button onClick={onClose} className="p-1 hover:bg-slate-100 rounded-lg transition-colors">
            <X className="h-5 w-5 text-slate-500" />
          </button>
        </div>

        {/* Toolbar */}
        <div className="flex flex-col gap-1.5 px-4 py-2 border-b border-slate-100 bg-slate-50">
          {/* Row 1: Zone selector + All layers / Clear */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Zone:</span>
            {Array.from({ length: numZones }, (_, idx) => {
              const zoneNum = idx + 1
              const zc = zoneColor(idx, numZones)
              return (
                <button
                  key={zoneNum}
                  onClick={() => setActiveZone(zoneNum)}
                  className={`w-6 h-6 rounded-full text-xs font-bold text-white flex items-center justify-center transition-all ${
                    activeZone === zoneNum ? 'ring-2 ring-offset-1 ring-slate-400 scale-110' : 'opacity-70 hover:opacity-100'
                  }`}
                  style={{ backgroundColor: zc.label }}
                >
                  {zoneNum}
                </button>
              )
            })}
            <button
              onClick={() => setNumZones(n => Math.min(MAX_ZONES, n + 1))}
              disabled={numZones >= MAX_ZONES}
              className="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-slate-400 hover:text-slate-600 hover:border-slate-400 disabled:opacity-30"
              title="Add zone"
            >
              <Plus className="h-3 w-3" />
            </button>
            {numZones > 2 && (
              <button
                onClick={() => setNumZones(n => Math.max(2, n - 1))}
                className="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-slate-400 hover:text-slate-600 hover:border-slate-400"
                title="Remove zone"
              >
                <Minus className="h-3 w-3" />
              </button>
            )}

            <div className="ml-auto flex items-center gap-2">
              {metadata.nlay > 1 && (
                <>
                  <label className="flex items-center gap-1 text-xs text-slate-500 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={applyAllLayers}
                      onChange={(e) => setApplyAllLayers(e.target.checked)}
                      className="h-3 w-3 accent-orange-500"
                    />
                    All layers
                  </label>
                  <button
                    onClick={copyToAllLayers}
                    disabled={totalAssignedAll === 0}
                    className="text-xs text-slate-500 hover:text-slate-700 px-1.5 py-0.5 border border-slate-300 rounded disabled:opacity-30"
                    title="Copy this layer's zones to all layers"
                  >
                    Copy to all
                  </button>
                </>
              )}
              <button
                onClick={() => {
                  onZoneAssignmentsChange({})
                }}
                className="text-xs text-slate-500 hover:text-slate-700 px-2 py-0.5 border border-slate-300 rounded"
              >
                Clear
              </button>

              {/* Save/Load zone definitions */}
              <div className="w-px h-4 bg-slate-300" />
              {showSaveInput ? (
                <div className="flex items-center gap-1">
                  <input
                    type="text"
                    value={saveName}
                    onChange={e => setSaveName(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') handleSaveZones(); if (e.key === 'Escape') setShowSaveInput(false) }}
                    placeholder="Definition name..."
                    className="text-xs border border-slate-300 rounded px-1.5 py-0.5 w-28"
                    autoFocus
                  />
                  <button onClick={handleSaveZones} disabled={!saveName.trim()} className="text-xs text-blue-600 hover:text-blue-800 disabled:opacity-30">Save</button>
                  <button onClick={() => setShowSaveInput(false)} className="text-xs text-slate-400 hover:text-slate-600">Cancel</button>
                </div>
              ) : (
                <button
                  onClick={() => setShowSaveInput(true)}
                  disabled={totalAssignedAll === 0}
                  className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700 px-1.5 py-0.5 border border-slate-300 rounded disabled:opacity-30"
                  title="Save current zone definitions"
                >
                  <Save className="h-3 w-3" />
                  Save
                </button>
              )}
              <div className="relative">
                <button
                  onClick={() => setShowLoadDropdown(!showLoadDropdown)}
                  className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700 px-1.5 py-0.5 border border-slate-300 rounded"
                  title="Load saved zone definitions"
                >
                  <FolderOpen className="h-3 w-3" />
                  Load
                </button>
                {showLoadDropdown && (
                  <div className="absolute top-full left-0 mt-1 bg-white border border-slate-200 rounded-lg shadow-lg z-20 min-w-[160px] py-1">
                    {!savedDefs || savedDefs.length === 0 ? (
                      <div className="px-3 py-2 text-xs text-slate-400">No saved definitions</div>
                    ) : (
                      savedDefs.map(d => (
                        <button
                          key={d.name}
                          onClick={() => handleLoadZones(d.name)}
                          className="w-full text-left px-3 py-1.5 text-xs hover:bg-slate-50 text-slate-700"
                        >
                          {d.name}
                          <span className="ml-2 text-slate-400">{d.zone_count} cells</span>
                        </button>
                      ))
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Row 2: Tools + Layer + Opacity + Compute */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              {toolNames.map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => {
                    setTool(key)
                    if (key !== 'polygon') setPolygonVertices([])
                    if (key !== 'rect') setRectStart(null)
                  }}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md border transition-colors ${
                    tool === key
                      ? 'bg-blue-50 border-blue-300 text-blue-700 font-medium'
                      : 'border-slate-200 text-slate-500 hover:text-slate-700 hover:border-slate-300'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {label}
                </button>
              ))}
            </div>

            {tool === 'polygon' && polygonVertices.length > 0 && (
              <button
                onClick={() => { setPolygonVertices([]); setLastVertexPixel(null) }}
                className="text-xs text-red-500 hover:text-red-700 px-2 py-1 border border-red-200 rounded"
              >
                Cancel Draw
              </button>
            )}

            <div className="w-px h-5 bg-slate-300" />

            {metadata.nlay > 1 && (
              <div className="flex items-center gap-1.5">
                <label className="text-xs text-slate-500">Layer:</label>
                <select
                  value={layer}
                  onChange={(e) => setLayer(Number(e.target.value))}
                  className="text-xs border border-slate-300 rounded px-1.5 py-1"
                >
                  {Array.from({ length: metadata.nlay }, (_, i) => (
                    <option key={i} value={i}>L{i + 1}</option>
                  ))}
                </select>
              </div>
            )}

            <div className="ml-auto flex items-center gap-1.5">
              <button
                onClick={() => handleComputeZoneBudget(true)}
                disabled={computing || totalAssignedAll === 0}
                className="flex items-center gap-1 px-2.5 py-1.5 text-xs rounded border border-orange-300 text-orange-600 hover:bg-orange-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                title="Quick preview (last timestep only)"
              >
                {computing && taskId ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />}
                Quick Preview
              </button>
              <button
                onClick={() => handleComputeZoneBudget(false)}
                disabled={computing || totalAssignedAll === 0}
                className="flex items-center gap-1 px-3 py-1.5 text-xs rounded bg-orange-500 text-white hover:bg-orange-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                title="Full computation (all timesteps)"
              >
                {computing && !taskId ? <Loader2 className="h-3 w-3 animate-spin" /> : <Calculator className="h-3 w-3" />}
                Full Compute
              </button>
            </div>
          </div>
        </div>

        {/* Map area */}
        <div
          className="flex-1 relative bg-slate-300"
          style={{ cursor: tool === 'polygon' || tool === 'rect' ? 'crosshair' : 'default' }}
        >
          {isLoading ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
              <span className="ml-3 text-slate-500">Loading grid data...</span>
            </div>
          ) : headError ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <AlertTriangle className="h-8 w-8 text-amber-500" />
              <p className="text-slate-600 font-medium">Failed to load grid data for layer {layer + 1}</p>
              <p className="text-sm text-slate-400">The server may be busy or temporarily unavailable.</p>
              <button
                onClick={() => refetchHeads()}
                className="mt-1 px-4 py-2 text-sm bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                Retry
              </button>
            </div>
          ) : cellFeatures.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center text-slate-400">
              No grid data available for painting
            </div>
          ) : mapViewState ? (
            <DeckGL
              key={deckKey}
              initialViewState={mapViewState}
              controller={controllerConfig as any}
              layers={layers}
              onClick={handleDeckClick}
              onHover={handleDeckHover}
              getCursor={() => tool === 'polygon' || tool === 'rect' ? 'crosshair' : 'grab'}
              onError={(error) => {
                console.warn('DeckGL error, resetting context:', error)
                setDeckKey(k => k + 1)
              }}
            >
              <Map mapStyle={DEFAULT_BASEMAP.style as any} />
            </DeckGL>
          ) : orthoViewState ? (
            <DeckGL
              key={deckKey}
              views={[new OrthographicView({ id: 'ortho', flipY: false })]}
              initialViewState={{ ortho: orthoViewState }}
              controller={controllerConfig as any}
              layers={layers}
              onClick={handleDeckClick}
              onHover={handleDeckHover}
              getCursor={() => tool === 'polygon' || tool === 'rect' ? 'crosshair' : 'grab'}
              onError={(error) => {
                console.warn('DeckGL error, resetting context:', error)
                setDeckKey(k => k + 1)
              }}
            />
          ) : null}

          {/* Close polygon context button */}
          {tool === 'polygon' && polygonVertices.length >= 3 && lastVertexPixel && (
            <div
              className="absolute z-10"
              style={{
                left: lastVertexPixel[0] + 14,
                top: lastVertexPixel[1] - 14,
              }}
            >
              <button
                onClick={closePolygon}
                className="px-2.5 py-1.5 text-xs font-medium bg-white rounded-lg shadow-lg border border-slate-200 text-orange-600 hover:bg-orange-50 hover:border-orange-300 transition-colors whitespace-nowrap"
              >
                Close Polygon
              </button>
            </div>
          )}
        </div>

        {/* Progress bar */}
        {computing && computeProgress && (
          <div className="px-4 py-2 border-t border-slate-100 bg-blue-50">
            <div className="flex items-center gap-2 text-xs">
              <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500 flex-shrink-0" />
              <span className="text-blue-700">{computeProgress.message || 'Computing...'}</span>
              <span className="ml-auto text-blue-500 font-medium">{computeProgress.progress}%</span>
            </div>
            <div className="mt-1 h-1.5 bg-blue-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-300"
                style={{ width: `${computeProgress.progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Error bar */}
        {computeError && (
          <div className="px-4 py-2 border-t border-red-100 bg-red-50 flex items-center gap-2 text-xs">
            <AlertTriangle className="h-3.5 w-3.5 text-red-500 flex-shrink-0" />
            <span className="text-red-700">{computeError}</span>
            <button
              onClick={() => { setComputeError(null); handleComputeZoneBudget(false) }}
              className="ml-auto px-2 py-0.5 text-red-600 hover:text-red-800 border border-red-300 rounded"
            >
              Retry
            </button>
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center gap-4 px-4 py-2 border-t border-slate-200 bg-slate-50 text-xs text-slate-500">
          <span>{toolHint}</span>
          <span className="ml-auto">Zone {activeZone}</span>
          <span>{layerAssigned} cells (L{layer + 1})</span>
          <span>{totalAssignedAll} total</span>
        </div>
      </div>
    </div>
  )
}

/** Create a small circle polygon for vertex dots */
function createCircle(center: [number, number], radius: number, segments = 12): [number, number][] {
  const pts: [number, number][] = []
  for (let i = 0; i <= segments; i++) {
    const angle = (i / segments) * 2 * Math.PI
    pts.push([
      center[0] + Math.cos(angle) * radius,
      center[1] + Math.sin(angle) * radius,
    ])
  }
  return pts
}

/** Estimate a reasonable vertex dot radius from cell features */
function getVertexRadius(cells: CellFeature[]): number {
  if (cells.length === 0) return 1
  // Use first cell's bounding box size as estimate
  const poly = cells[0].polygon
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const [x, y] of poly) {
    if (x < minX) minX = x; if (x > maxX) maxX = x
    if (y < minY) minY = y; if (y > maxY) maxY = y
  }
  return Math.max(maxX - minX, maxY - minY) * 0.15
}
