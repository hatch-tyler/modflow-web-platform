import { useState, useEffect, useMemo, Suspense } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Box, Layers, Loader2, Eye, EyeOff, Droplets, ArrowDownCircle, Waves, Scissors, X } from 'lucide-react'
import { projectsApi } from '../services/api'
import GridViewer from '../components/viewer3d/GridViewer'
import CrossSectionPanel from '../components/viewer3d/CrossSectionPanel'
import { parseGridMesh, parseArrayData, type GridMeshData, type ArrayData } from '../utils/binaryParser'
import { computeCellMask } from '../utils/crossSection'
import clsx from 'clsx'

// Boundary condition types
export interface BoundaryCell {
  layer: number
  row: number
  col: number
  values: Record<string, number>
}

export interface BoundaryPackage {
  package_type: string
  name: string
  description: string
  stress_period: number
  value_names: string[]
  cell_count: number
  cells: BoundaryCell[]
}

export interface BoundarySummary {
  [key: string]: {
    name: string
    description: string
    cell_count: number
    value_names: string[]
  }
}

// Boundary colors for visualization
export const BOUNDARY_COLORS: Record<string, string> = {
  CHD: '#3b82f6', // blue - constant head
  WEL: '#ef4444', // red - wells
  RIV: '#06b6d4', // cyan - rivers
  DRN: '#f97316', // orange - drains
  GHB: '#8b5cf6', // purple - general head
  RCH: '#22c55e', // green - recharge
  EVT: '#eab308', // yellow - evapotranspiration
}

// API functions for binary data
async function fetchGridMesh(projectId: string): Promise<GridMeshData> {
  const response = await fetch(`/api/v1/projects/${projectId}/grid`)
  if (!response.ok) throw new Error('Failed to load grid')
  const buffer = await response.arrayBuffer()
  return parseGridMesh(buffer)
}

async function fetchArrayData(projectId: string, arrayName: string): Promise<ArrayData> {
  const response = await fetch(`/api/v1/projects/${projectId}/arrays/${arrayName}`)
  if (!response.ok) throw new Error(`Failed to load array: ${arrayName}`)
  const buffer = await response.arrayBuffer()
  return parseArrayData(buffer)
}

async function fetchAvailableArrays(projectId: string): Promise<{ name: string; description: string; min: number; max: number }[]> {
  const response = await fetch(`/api/v1/projects/${projectId}/arrays`)
  if (!response.ok) return []
  const data = await response.json()
  return data.arrays || []
}

async function fetchBoundaries(projectId: string, stressPeriod: number = 0): Promise<{ boundaries: BoundarySummary; nper: number }> {
  const response = await fetch(`/api/v1/projects/${projectId}/boundaries?stress_period=${stressPeriod}`)
  if (!response.ok) return { boundaries: {}, nper: 1 }
  const data = await response.json()
  return { boundaries: data.boundaries || {}, nper: data.nper || 1 }
}

async function fetchBoundaryPackage(projectId: string, packageType: string, stressPeriod: number = 0): Promise<BoundaryPackage | null> {
  const response = await fetch(`/api/v1/projects/${projectId}/boundaries/${packageType}?stress_period=${stressPeriod}`)
  if (!response.ok) return null
  return response.json()
}

// Icon mapping for boundary types
function BoundaryIcon({ type, className, style }: { type: string; className?: string; style?: React.CSSProperties }) {
  switch (type) {
    case 'WEL':
      return <ArrowDownCircle className={className} style={style} />
    case 'RIV':
    case 'DRN':
      return <Waves className={className} style={style} />
    case 'RCH':
    case 'EVT':
      return <Droplets className={className} style={style} />
    default:
      return <Box className={className} style={style} />
  }
}

export default function ViewerPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const [visibleLayers, setVisibleLayers] = useState<boolean[]>([])
  const [selectedProperty, setSelectedProperty] = useState('hk')
  const [colormap, setColormap] = useState<'viridis' | 'plasma' | 'coolwarm' | 'terrain'>('viridis')
  const [opacity, setOpacity] = useState(0.9)
  const [verticalExaggeration, setVerticalExaggeration] = useState(10)
  const [showWireframe, setShowWireframe] = useState(true)
  const [stressPeriod, setStressPeriod] = useState(0)
  const [visibleBoundaries, setVisibleBoundaries] = useState<Record<string, boolean>>({})
  const [loadedBoundaries, setLoadedBoundaries] = useState<Record<string, BoundaryPackage>>({})
  const [showInactive, setShowInactive] = useState(false)
  const [crossSectionLine, setCrossSectionLine] = useState<[number, number][]>([])
  const [crossSectionSide, setCrossSectionSide] = useState<'left' | 'right' | 'both'>('both')
  const [isDrawingCrossSection, setIsDrawingCrossSection] = useState(false)

  // Fetch project info
  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  })

  // Fetch grid mesh
  const { data: gridData, isLoading: gridLoading, error: gridError } = useQuery({
    queryKey: ['grid', projectId],
    queryFn: () => fetchGridMesh(projectId!),
    enabled: !!projectId && !!project?.is_valid,
  })

  // Fetch available arrays
  const { data: availableArrays } = useQuery({
    queryKey: ['arrays', projectId],
    queryFn: () => fetchAvailableArrays(projectId!),
    enabled: !!projectId && !!project?.is_valid,
  })

  // Fetch selected property array
  const { data: arrayData, isLoading: arrayLoading } = useQuery({
    queryKey: ['array', projectId, selectedProperty],
    queryFn: () => fetchArrayData(projectId!, selectedProperty),
    enabled: !!projectId && !!gridData && !!selectedProperty,
  })

  // Always fetch ibound/idomain to identify inactive cells
  const { data: iboundData } = useQuery({
    queryKey: ['array', projectId, 'ibound'],
    queryFn: () => fetchArrayData(projectId!, 'ibound'),
    enabled: !!projectId && !!gridData,
  })

  // Fetch boundary conditions summary
  const { data: boundaryInfo } = useQuery({
    queryKey: ['boundaries', projectId, stressPeriod],
    queryFn: () => fetchBoundaries(projectId!, stressPeriod),
    enabled: !!projectId && !!project?.is_valid,
  })

  // Initialize visible layers when grid loads
  useEffect(() => {
    if (gridData) {
      setVisibleLayers(new Array(gridData.nlay).fill(true))
    }
  }, [gridData])

  // Initialize boundary visibility when boundaries are fetched
  useEffect(() => {
    if (boundaryInfo?.boundaries) {
      const initial: Record<string, boolean> = {}
      Object.keys(boundaryInfo.boundaries).forEach(key => {
        // Only show non-array boundaries by default (CHD, WEL, RIV, DRN, GHB)
        initial[key] = !['RCH', 'EVT'].includes(key)
      })
      setVisibleBoundaries(initial)
    }
  }, [boundaryInfo])

  // Load boundary package data when visibility is toggled on
  useEffect(() => {
    Object.entries(visibleBoundaries).forEach(([type, visible]) => {
      if (visible && !loadedBoundaries[type]) {
        fetchBoundaryPackage(projectId!, type, stressPeriod).then(pkg => {
          if (pkg) {
            setLoadedBoundaries(prev => ({ ...prev, [type]: pkg }))
          }
        })
      }
    })
  }, [visibleBoundaries, projectId, stressPeriod, loadedBoundaries])

  // Toggle layer visibility
  const toggleLayer = (layerIndex: number) => {
    setVisibleLayers(prev => {
      const next = [...prev]
      next[layerIndex] = !next[layerIndex]
      return next
    })
  }

  // Toggle all layers
  const toggleAllLayers = (visible: boolean) => {
    if (gridData) {
      setVisibleLayers(new Array(gridData.nlay).fill(visible))
    }
  }

  // Compute cross-section cell mask
  const cellMask = useMemo(() => {
    if (!gridData || crossSectionLine.length < 2 || crossSectionSide === 'both') return undefined
    const ncells = gridData.nlay * gridData.nrow * gridData.ncol
    return computeCellMask(crossSectionLine, gridData.centers, ncells, crossSectionSide)
  }, [gridData, crossSectionLine, crossSectionSide])

  // Toggle boundary visibility
  const toggleBoundary = (type: string) => {
    setVisibleBoundaries(prev => ({
      ...prev,
      [type]: !prev[type]
    }))
  }

  if (projectLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (!project?.is_valid) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
        <Box className="h-12 w-12 text-yellow-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-yellow-800 mb-2">No Valid Model</h3>
        <p className="text-yellow-600">
          Please upload a valid MODFLOW model first to use the 3D viewer.
        </p>
      </div>
    )
  }

  // Get visible boundary data for the viewer
  const visibleBoundaryData = Object.entries(loadedBoundaries)
    .filter(([type]) => visibleBoundaries[type])
    .reduce((acc, [type, pkg]) => {
      acc[type] = pkg
      return acc
    }, {} as Record<string, BoundaryPackage>)

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-4">
      {/* Control Panel */}
      <div className="w-72 bg-white rounded-lg border border-slate-200 p-4 overflow-y-auto flex-shrink-0">
        <h3 className="font-semibold text-slate-800 mb-4">Model Controls</h3>

        {/* Grid info */}
        <div className="mb-6 p-3 bg-slate-50 rounded-lg text-sm">
          <div className="font-medium text-slate-700 mb-1">Grid Dimensions</div>
          <div className="text-slate-600">
            {project.grid_type === 'vertex'
              ? `${project.nlay} layers × ${project.ncol} cells/layer (DISV)`
              : project.grid_type === 'unstructured'
              ? `${project.ncol} nodes (DISU)`
              : `${project.nlay} layers × ${project.nrow} rows × ${project.ncol} cols`}
          </div>
          <div className="text-slate-500 text-xs mt-1">
            {((project.nlay || 0) * (project.nrow || 1) * (project.ncol || 0)).toLocaleString()} cells
          </div>
          {gridData && (
            <div className="text-slate-500 text-xs mt-1">
              {gridData.gridType === 0 ? 'Structured' : 'Unstructured'} grid
            </div>
          )}
        </div>

        {/* Layer visibility */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-slate-600">Layers</h4>
            <div className="flex gap-1">
              <button
                onClick={() => toggleAllLayers(true)}
                className="p-1 text-slate-400 hover:text-slate-600"
                title="Show all"
              >
                <Eye className="h-4 w-4" />
              </button>
              <button
                onClick={() => toggleAllLayers(false)}
                className="p-1 text-slate-400 hover:text-slate-600"
                title="Hide all"
              >
                <EyeOff className="h-4 w-4" />
              </button>
            </div>
          </div>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {visibleLayers.map((visible, i) => (
              <label
                key={i}
                className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded"
              >
                <input
                  type="checkbox"
                  checked={visible}
                  onChange={() => toggleLayer(i)}
                  className="rounded text-blue-600"
                />
                <Layers className="h-4 w-4 text-slate-400" />
                <span className="text-sm">Layer {i + 1}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Boundary Conditions */}
        {boundaryInfo && Object.keys(boundaryInfo.boundaries).length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-600 mb-2">Boundary Conditions</h4>

            {/* Stress period selector */}
            {(boundaryInfo.nper || 1) > 1 && (
              <div className="mb-2">
                <label className="text-xs text-slate-500">Stress Period</label>
                <select
                  value={stressPeriod}
                  onChange={(e) => {
                    setStressPeriod(parseInt(e.target.value))
                    setLoadedBoundaries({}) // Clear loaded data to refetch
                  }}
                  className="w-full px-2 py-1 border border-slate-300 rounded text-sm"
                >
                  {Array.from({ length: boundaryInfo.nper }, (_, i) => (
                    <option key={i} value={i}>Period {i + 1}</option>
                  ))}
                </select>
              </div>
            )}

            <div className="space-y-1">
              {Object.entries(boundaryInfo.boundaries).map(([type, info]) => (
                <label
                  key={type}
                  className="flex items-center gap-2 cursor-pointer hover:bg-slate-50 p-1 rounded"
                >
                  <input
                    type="checkbox"
                    checked={visibleBoundaries[type] || false}
                    onChange={() => toggleBoundary(type)}
                    className="rounded"
                    style={{ accentColor: BOUNDARY_COLORS[type] || '#666' }}
                  />
                  <BoundaryIcon
                    type={type}
                    className="h-4 w-4"
                    style={{ color: BOUNDARY_COLORS[type] || '#666' }}
                  />
                  <span className="text-sm flex-1">{info.description}</span>
                  <span className="text-xs text-slate-400">{info.cell_count}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Property display */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-slate-600 mb-2">Display Property</h4>
          <select
            value={selectedProperty}
            onChange={(e) => setSelectedProperty(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm"
          >
            {availableArrays && availableArrays.length > 0 ? (
              availableArrays.map((arr) => (
                <option key={arr.name} value={arr.name}>
                  {arr.description || arr.name}
                </option>
              ))
            ) : (
              <option value="hk">Hydraulic Conductivity</option>
            )}
          </select>
          {arrayLoading && (
            <div className="text-xs text-slate-400 mt-1 flex items-center gap-1">
              <Loader2 className="h-3 w-3 animate-spin" />
              Loading...
            </div>
          )}
        </div>

        {/* Colormap */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-slate-600 mb-2">Colormap</h4>
          <div className="grid grid-cols-2 gap-2">
            {(['viridis', 'plasma', 'coolwarm', 'terrain'] as const).map((cm) => (
              <button
                key={cm}
                onClick={() => setColormap(cm)}
                className={clsx(
                  'px-3 py-2 text-xs border rounded-lg capitalize transition-colors',
                  colormap === cm
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-slate-300 hover:bg-slate-50'
                )}
              >
                {cm}
              </button>
            ))}
          </div>
        </div>

        {/* Vertical Exaggeration */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-slate-600 mb-2">
            Vertical Exaggeration: {verticalExaggeration}x
          </h4>
          <input
            type="range"
            min="1"
            max="50"
            step="1"
            value={verticalExaggeration}
            onChange={(e) => setVerticalExaggeration(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>1x</span>
            <span>50x</span>
          </div>
        </div>

        {/* Opacity */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-slate-600 mb-2">
            Opacity: {Math.round(opacity * 100)}%
          </h4>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Wireframe toggle */}
        <div className="mb-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showWireframe}
              onChange={(e) => setShowWireframe(e.target.checked)}
              className="rounded text-blue-600"
            />
            <span className="text-sm text-slate-600">Show cell edges</span>
          </label>
        </div>

        {/* Inactive cells toggle */}
        <div className="mb-6">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showInactive}
              onChange={(e) => setShowInactive(e.target.checked)}
              className="rounded text-blue-600"
            />
            <span className="text-sm text-slate-600">Show inactive cells</span>
          </label>
          <p className="text-xs text-slate-400 mt-1 ml-6">
            Cells with IBOUND/IDOMAIN &le; 0
          </p>
        </div>

        {/* Cross Section */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-1.5">
            <Scissors className="h-4 w-4" />
            Cross Section
          </h4>
          <div className="flex gap-2 mb-2">
            <button
              onClick={() => {
                if (isDrawingCrossSection) {
                  setIsDrawingCrossSection(false)
                } else {
                  setCrossSectionLine([])
                  setCrossSectionSide('both')
                  setIsDrawingCrossSection(true)
                }
              }}
              className={clsx(
                'flex-1 px-3 py-1.5 text-xs rounded-lg border transition-colors',
                isDrawingCrossSection
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-slate-300 hover:bg-slate-50 text-slate-600'
              )}
            >
              {isDrawingCrossSection ? 'Done' : 'Draw'}
            </button>
            <button
              onClick={() => {
                setCrossSectionLine([])
                setCrossSectionSide('both')
                setIsDrawingCrossSection(false)
              }}
              disabled={crossSectionLine.length === 0 && !isDrawingCrossSection}
              className={clsx(
                'px-3 py-1.5 text-xs rounded-lg border transition-colors',
                crossSectionLine.length === 0 && !isDrawingCrossSection
                  ? 'border-slate-200 text-slate-300 cursor-not-allowed'
                  : 'border-slate-300 hover:bg-slate-50 text-slate-600'
              )}
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="flex gap-1 mb-2">
            {(['left', 'right', 'both'] as const).map((s) => (
              <button
                key={s}
                onClick={() => setCrossSectionSide(s)}
                disabled={crossSectionLine.length < 2}
                className={clsx(
                  'flex-1 px-2 py-1.5 text-xs border rounded-lg capitalize transition-colors',
                  crossSectionLine.length < 2
                    ? 'border-slate-200 text-slate-300 cursor-not-allowed'
                    : crossSectionSide === s
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-slate-300 hover:bg-slate-50 text-slate-600'
                )}
              >
                {s}
              </button>
            ))}
          </div>
          <p className="text-xs text-slate-400">
            {isDrawingCrossSection
              ? 'Click on the map above to draw'
              : crossSectionLine.length >= 2
              ? `${crossSectionLine.length} points`
              : 'Draw a line to slice the model'}
          </p>
        </div>

        {/* Legend */}
        {Object.keys(visibleBoundaryData).length > 0 && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-slate-600 mb-2">Legend</h4>
            <div className="space-y-1">
              {Object.entries(visibleBoundaryData).map(([type, pkg]) => (
                <div key={type} className="flex items-center gap-2 text-xs">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: BOUNDARY_COLORS[type] || '#666' }}
                  />
                  <span>{pkg.description}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right panel: drawing panel + 3D viewer */}
      <div className="flex-1 min-h-0 flex flex-col rounded-lg border border-slate-700 overflow-hidden">
        {/* Drawing panel — shown when drawing or when a cross-section line exists */}
        {(isDrawingCrossSection || crossSectionLine.length >= 2) && gridData && (
          <div className="h-[250px] flex-shrink-0 border-b border-slate-700 bg-slate-800">
            <CrossSectionPanel
              gridData={gridData}
              polyline={crossSectionLine}
              isDrawing={isDrawingCrossSection}
              onUpdatePolyline={setCrossSectionLine}
              onFinishDrawing={() => setIsDrawingCrossSection(false)}
            />
          </div>
        )}

        {/* 3D Viewer Canvas — takes remaining height */}
        <div className="flex-1 min-h-0 bg-slate-900 overflow-hidden relative">
          {gridLoading ? (
            <div className="absolute inset-0 flex items-center justify-center text-slate-400">
              <div className="text-center">
                <Loader2 className="h-12 w-12 mx-auto mb-4 animate-spin" />
                <p>Loading 3D grid...</p>
              </div>
            </div>
          ) : gridError ? (
            <div className="absolute inset-0 flex items-center justify-center text-red-400">
              <div className="text-center">
                <Box className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Failed to load grid</p>
                <p className="text-sm opacity-75">{(gridError as Error).message}</p>
              </div>
            </div>
          ) : gridData ? (
            <Suspense fallback={
              <div className="absolute inset-0 flex items-center justify-center text-slate-400">
                <Loader2 className="h-12 w-12 animate-spin" />
              </div>
            }>
              <GridViewer
                gridData={gridData}
                arrayData={arrayData}
                iboundData={iboundData}
                showInactive={showInactive}
                visibleLayers={visibleLayers}
                colormap={colormap}
                opacity={opacity}
                boundaries={visibleBoundaryData}
                verticalExaggeration={verticalExaggeration}
                showWireframe={showWireframe}
                cellMask={cellMask}
              />
            </Suspense>
          ) : null}

          {/* Loading overlay for property data */}
          {arrayLoading && gridData && (
            <div className="absolute top-4 right-4 bg-slate-800/90 text-slate-300 px-4 py-2 rounded-lg flex items-center gap-2 shadow-lg">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">Updating colors...</span>
            </div>
          )}

          {/* Controls hint */}
          <div className="absolute bottom-4 left-4 text-slate-500 text-xs bg-slate-800/80 px-3 py-2 rounded">
            <span className="font-medium">Controls:</span> Drag to rotate • Scroll to zoom • Right-drag to pan
          </div>
        </div>
      </div>
    </div>
  )
}
