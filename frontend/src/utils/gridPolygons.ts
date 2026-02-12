import type proj4 from 'proj4'
import type { GridGeometry, StructuredGridInfo } from '@/types'

export interface CellFeature {
  polygon: [number, number][]
  value: number | null
  cellIndex: number
  row?: number
  col?: number
  centroid: [number, number]
}

export interface MapViewState {
  longitude: number
  latitude: number
  zoom: number
  pitch: number
  bearing: number
}

export interface OrthoViewState {
  target: [number, number, number]
  zoom: number
}

export interface CellFeaturesResult {
  cellFeatures: CellFeature[]
  mapViewState: MapViewState | null
  orthoViewState: OrthoViewState | null
}

/** Check if a head value is no-data (null, dry, or HNOFLO). */
export function isNoData(val: number | null | undefined): boolean {
  return val === null || val === undefined || Math.abs(val!) > 1e20 || val! < -880
}

/** Flatten a possibly-2D head data array to 1D. */
export function flattenHeadData(data: (number | null)[][]): (number | null)[] {
  const values: (number | null)[] = []
  for (const row of data) {
    if (Array.isArray(row)) {
      values.push(...row)
    } else {
      values.push(row as number | null)
    }
  }
  return values
}

interface PolygonGridOpts {
  gridGeometry: GridGeometry
  headData: (number | null)[][]
  layer: number
  xoff: number
  yoff: number
  angrot: number // degrees
  converter: proj4.Converter | null
}

/** Build cell features from a polygon/vertex/unstructured grid geometry. */
export function buildPolygonGridFeatures(opts: PolygonGridOpts): CellFeaturesResult {
  const { gridGeometry, headData, layer, xoff, yoff, angrot: angrotDeg, converter } = opts
  const hasProjection = !!converter
  const angrot = angrotDeg * (Math.PI / 180)

  const layerKey = String(layer)
  const layerData = gridGeometry.layers[layerKey] || gridGeometry.layers['0']
  if (!layerData) return { cellFeatures: [], mapViewState: null, orthoViewState: null }

  const polygons = layerData.polygons
  const flatValues = flattenHeadData(headData)
  const features: CellFeature[] = []
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity

  for (let i = 0; i < polygons.length; i++) {
    const poly = polygons[i]
    if (!poly || poly.length < 3) continue

    const val = i < flatValues.length ? flatValues[i] : null
    const noData = isNoData(val)

    const transformedPoly: [number, number][] = poly.map((pt) => {
      let x = pt[0], y = pt[1]
      if (angrot !== 0) {
        const rx = x * Math.cos(angrot) - y * Math.sin(angrot)
        const ry = x * Math.sin(angrot) + y * Math.cos(angrot)
        x = rx; y = ry
      }
      x += xoff; y += yoff
      if (hasProjection) {
        const [lng, lat] = converter!.forward([x, y])
        return [lng, lat] as [number, number]
      }
      return [x, y] as [number, number]
    })

    for (const [cx, cy] of transformedPoly) {
      if (cx < minX) minX = cx
      if (cx > maxX) maxX = cx
      if (cy < minY) minY = cy
      if (cy > maxY) maxY = cy
    }

    // Centroid (exclude closing vertex if present)
    let sumX = 0, sumY = 0, n = 0
    for (const pt of transformedPoly) { sumX += pt[0]; sumY += pt[1]; n++ }
    if (n > 1 && transformedPoly[0][0] === transformedPoly[n - 1][0] && transformedPoly[0][1] === transformedPoly[n - 1][1]) {
      sumX -= transformedPoly[n - 1][0]; sumY -= transformedPoly[n - 1][1]; n--
    }
    const centroid: [number, number] = [sumX / n, sumY / n]

    features.push({
      polygon: transformedPoly,
      value: noData ? null : val,
      cellIndex: i,
      centroid,
    })
  }

  return finalize(features, minX, maxX, minY, maxY, hasProjection)
}

interface StructuredGridOpts {
  gridInfo: StructuredGridInfo | null
  headData: (number | null)[][]
  xoff: number
  yoff: number
  angrot: number // degrees
  converter: proj4.Converter | null
}

/** Build cell features from a structured grid (delr/delc). Falls back to unit cells when delr/delc unavailable. */
export function buildStructuredGridFeatures(opts: StructuredGridOpts): CellFeaturesResult {
  const { headData, xoff, yoff, angrot: angrotDeg, converter } = opts
  const hasProjection = !!converter
  const angrot = angrotDeg * (Math.PI / 180)

  const nrow = headData.length
  const ncol = headData[0]?.length || 0
  if (nrow === 0 || ncol === 0) return { cellFeatures: [], mapViewState: null, orthoViewState: null }

  const hasGridInfo = opts.gridInfo?.delr && opts.gridInfo?.delc

  // Build edge positions
  let xEdges: number[]
  let yEdges: number[]
  if (hasGridInfo) {
    xEdges = [0]
    for (let c = 0; c < ncol; c++) xEdges.push(xEdges[c] + opts.gridInfo!.delr![c])
    yEdges = [0]
    for (let r = 0; r < nrow; r++) yEdges.push(yEdges[r] + opts.gridInfo!.delc![r])
  } else {
    // Unit cells fallback
    xEdges = Array.from({ length: ncol + 1 }, (_, i) => i)
    yEdges = Array.from({ length: nrow + 1 }, (_, i) => i)
  }
  const totalHeight = yEdges[nrow]

  function transformPoint(localX: number, localY: number): [number, number] {
    let x = localX, y = localY
    if (angrot !== 0) {
      const rx = x * Math.cos(angrot) - y * Math.sin(angrot)
      const ry = x * Math.sin(angrot) + y * Math.cos(angrot)
      x = rx; y = ry
    }
    x += xoff; y += yoff
    if (hasProjection) {
      return converter!.forward([x, y]) as [number, number]
    }
    return [x, y]
  }

  const features: CellFeature[] = []
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity

  for (let r = 0; r < nrow; r++) {
    for (let c = 0; c < ncol; c++) {
      const val = headData[r]?.[c]
      const noData = isNoData(val)
      const cellIdx = r * ncol + c

      const x0 = xEdges[c]
      const x1 = xEdges[c + 1]
      const y0 = totalHeight - yEdges[r + 1]
      const y1 = totalHeight - yEdges[r]

      const corners: [number, number][] = [
        transformPoint(x0, y0),
        transformPoint(x1, y0),
        transformPoint(x1, y1),
        transformPoint(x0, y1),
        transformPoint(x0, y0),
      ]

      for (const [cx, cy] of corners) {
        if (cx < minX) minX = cx
        if (cx > maxX) maxX = cx
        if (cy < minY) minY = cy
        if (cy > maxY) maxY = cy
      }

      const midPt = transformPoint((x0 + x1) / 2, (y0 + y1) / 2)

      features.push({
        polygon: corners,
        value: noData ? null : val,
        cellIndex: cellIdx,
        row: r,
        col: c,
        centroid: midPt,
      })
    }
  }

  return finalize(features, minX, maxX, minY, maxY, hasProjection)
}

function finalize(
  features: CellFeature[],
  minX: number,
  maxX: number,
  minY: number,
  maxY: number,
  hasProjection: boolean,
): CellFeaturesResult {
  if (features.length === 0) return { cellFeatures: [], mapViewState: null, orthoViewState: null }

  if (hasProjection) {
    const centerLng = (minX + maxX) / 2
    const centerLat = (minY + maxY) / 2
    const span = Math.max(maxY - minY, maxX - minX)
    const zoom = span > 0 ? Math.max(1, Math.min(18, Math.log2(360 / span) - 1)) : 12
    return {
      cellFeatures: features,
      mapViewState: { longitude: centerLng, latitude: centerLat, zoom, pitch: 0, bearing: 0 },
      orthoViewState: null,
    }
  } else {
    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    const span = Math.max(maxX - minX, maxY - minY)
    const zoom = span > 0 ? Math.log2(700 / span) : 0
    return {
      cellFeatures: features,
      mapViewState: null,
      orthoViewState: { target: [centerX, centerY, 0], zoom },
    }
  }
}
