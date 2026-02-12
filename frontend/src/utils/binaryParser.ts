/**
 * Binary data parsing utilities for MODFLOW grid data.
 */

export interface GridMeshData {
  gridType: number       // 0=structured, 1=unstructured
  nlay: number
  nrow: number
  ncol: number
  centers: Float32Array  // (ncells * 3) - x, y, z for each cell
  delr?: Float32Array    // column widths (structured only)
  delc?: Float32Array    // row heights (structured only)
  vertices?: Float32Array // cell vertices (USG only, ncells * 8 * 3)
  top: Float32Array      // top elevation (nrow * ncol)
  botm: Float32Array     // bottom elevations (nlay * nrow * ncol)
}

export interface ArrayData {
  shape: number[]
  data: Float32Array
}

/**
 * Parse binary grid mesh data from the API.
 *
 * Structured (gridType=0):
 *   gridType(i4) + nlay(i4) + nrow(i4) + ncol(i4) +
 *   centers(ncells*3 f32) + delr(ncol f32) + delc(nrow f32) +
 *   top(nrow*ncol f32) + botm(nlay*nrow*ncol f32)
 *
 * Unstructured (gridType=1):
 *   gridType(i4) + nlay(i4) + nrow(i4) + ncol(i4) +
 *   centers(ncells*3 f32) + vertices(ncells*8*3 f32) +
 *   top(nrow*ncol f32) + botm(nlay*nrow*ncol f32)
 */
export function parseGridMesh(buffer: ArrayBuffer): GridMeshData {
  const view = new DataView(buffer)
  let offset = 0

  // Read header
  const gridType = view.getInt32(offset, true); offset += 4
  const nlay = view.getInt32(offset, true); offset += 4
  const nrow = view.getInt32(offset, true); offset += 4
  const ncol = view.getInt32(offset, true); offset += 4

  const ncells = nlay * nrow * ncol

  // Read centers (ncells * 3 floats)
  const centersLength = ncells * 3
  const centers = new Float32Array(buffer, offset, centersLength)
  offset += centersLength * 4

  let delr: Float32Array | undefined
  let delc: Float32Array | undefined
  let vertices: Float32Array | undefined

  if (gridType === 0) {
    // Structured: read delr + delc
    delr = new Float32Array(buffer, offset, ncol)
    offset += ncol * 4

    delc = new Float32Array(buffer, offset, nrow)
    offset += nrow * 4
  } else {
    // Unstructured: read vertices (ncells * 8 * 3 floats)
    const verticesLength = ncells * 8 * 3
    vertices = new Float32Array(buffer, offset, verticesLength)
    offset += verticesLength * 4
  }

  // Read top (nrow * ncol floats)
  const topLength = nrow * ncol
  const top = new Float32Array(buffer, offset, topLength)
  offset += topLength * 4

  // Read botm (nlay * nrow * ncol floats)
  const botmLength = nlay * nrow * ncol
  const botm = new Float32Array(buffer, offset, botmLength)

  return { gridType, nlay, nrow, ncol, centers, delr, delc, vertices, top, botm }
}

/**
 * Parse binary array data from the API.
 */
export function parseArrayData(buffer: ArrayBuffer): ArrayData {
  const view = new DataView(buffer)
  let offset = 0

  // Read number of dimensions
  const ndim = view.getInt32(offset, true); offset += 4

  // Read shape
  const shape: number[] = []
  for (let i = 0; i < ndim; i++) {
    shape.push(view.getInt32(offset, true))
    offset += 4
  }

  // Calculate total size
  const totalSize = shape.reduce((a, b) => a * b, 1)

  // Read data
  const data = new Float32Array(buffer, offset, totalSize)

  return { shape, data }
}

/**
 * Get cell index from layer, row, col.
 */
export function getCellIndex(
  layer: number,
  row: number,
  col: number,
  nrow: number,
  ncol: number
): number {
  return layer * nrow * ncol + row * ncol + col
}

/**
 * Get layer, row, col from cell index.
 */
export function getCellCoords(
  index: number,
  nrow: number,
  ncol: number
): { layer: number; row: number; col: number } {
  const layer = Math.floor(index / (nrow * ncol))
  const remainder = index % (nrow * ncol)
  const row = Math.floor(remainder / ncol)
  const col = remainder % ncol
  return { layer, row, col }
}

/**
 * Create a color array from values using a colormap.
 */
export function valuesToColors(
  values: Float32Array,
  colormap: 'viridis' | 'plasma' | 'coolwarm' | 'terrain' = 'viridis'
): Float32Array {
  const colors = new Float32Array(values.length * 3)

  // Find min/max (excluding NaN and very large values)
  let min = Infinity
  let max = -Infinity

  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (isFinite(v) && Math.abs(v) < 1e30) {
      min = Math.min(min, v)
      max = Math.max(max, v)
    }
  }

  // If all values are the same or very close, use a default range
  const range = max - min
  const useRange = range > 1e-10 ? range : 1

  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    let t = 0.5  // default for invalid values

    if (isFinite(v) && Math.abs(v) < 1e30) {
      if (range > 1e-10) {
        t = (v - min) / useRange
      } else {
        // All values are the same - use middle of colormap
        t = 0.5
      }
    }

    const [r, g, b] = getColormapColor(t, colormap)
    colors[i * 3] = r
    colors[i * 3 + 1] = g
    colors[i * 3 + 2] = b
  }

  return colors
}

/**
 * Get RGB color from colormap at position t (0-1).
 */
function getColormapColor(
  t: number,
  colormap: string
): [number, number, number] {
  t = Math.max(0, Math.min(1, t))

  switch (colormap) {
    case 'viridis':
      return viridis(t)
    case 'plasma':
      return plasma(t)
    case 'coolwarm':
      return coolwarm(t)
    case 'terrain':
      return terrain(t)
    default:
      return viridis(t)
  }
}

// Viridis colormap - using lookup table for accuracy
const VIRIDIS_COLORS: [number, number, number][] = [
  [0.267, 0.004, 0.329],  // 0.0 - dark purple
  [0.282, 0.140, 0.458],  // 0.1
  [0.253, 0.265, 0.530],  // 0.2
  [0.191, 0.407, 0.556],  // 0.3
  [0.127, 0.566, 0.551],  // 0.4
  [0.134, 0.658, 0.518],  // 0.5
  [0.267, 0.749, 0.441],  // 0.6
  [0.478, 0.821, 0.318],  // 0.7
  [0.741, 0.873, 0.150],  // 0.8
  [0.993, 0.906, 0.144],  // 0.9
  [0.993, 0.906, 0.144],  // 1.0 - yellow
]

function viridis(t: number): [number, number, number] {
  const idx = t * (VIRIDIS_COLORS.length - 1)
  const i0 = Math.floor(idx)
  const i1 = Math.min(i0 + 1, VIRIDIS_COLORS.length - 1)
  const frac = idx - i0

  const c0 = VIRIDIS_COLORS[i0]
  const c1 = VIRIDIS_COLORS[i1]

  return [
    c0[0] + frac * (c1[0] - c0[0]),
    c0[1] + frac * (c1[1] - c0[1]),
    c0[2] + frac * (c1[2] - c0[2]),
  ]
}

// Plasma colormap - using lookup table
const PLASMA_COLORS: [number, number, number][] = [
  [0.050, 0.030, 0.528],  // 0.0 - dark blue
  [0.294, 0.012, 0.615],  // 0.1
  [0.492, 0.012, 0.658],  // 0.2
  [0.658, 0.135, 0.588],  // 0.3
  [0.798, 0.280, 0.470],  // 0.4
  [0.899, 0.396, 0.358],  // 0.5
  [0.958, 0.518, 0.263],  // 0.6
  [0.988, 0.652, 0.175],  // 0.7
  [0.988, 0.809, 0.145],  // 0.8
  [0.940, 0.975, 0.131],  // 0.9
  [0.940, 0.975, 0.131],  // 1.0 - yellow
]

function plasma(t: number): [number, number, number] {
  const idx = t * (PLASMA_COLORS.length - 1)
  const i0 = Math.floor(idx)
  const i1 = Math.min(i0 + 1, PLASMA_COLORS.length - 1)
  const frac = idx - i0

  const c0 = PLASMA_COLORS[i0]
  const c1 = PLASMA_COLORS[i1]

  return [
    c0[0] + frac * (c1[0] - c0[0]),
    c0[1] + frac * (c1[1] - c0[1]),
    c0[2] + frac * (c1[2] - c0[2]),
  ]
}

// Coolwarm diverging colormap
const COOLWARM_COLORS: [number, number, number][] = [
  [0.230, 0.299, 0.754],  // 0.0 - blue
  [0.411, 0.490, 0.858],  // 0.2
  [0.600, 0.680, 0.920],  // 0.4
  [0.865, 0.865, 0.865],  // 0.5 - white/gray
  [0.950, 0.680, 0.580],  // 0.6
  [0.890, 0.440, 0.330],  // 0.8
  [0.706, 0.016, 0.150],  // 1.0 - red
]

function coolwarm(t: number): [number, number, number] {
  const idx = t * (COOLWARM_COLORS.length - 1)
  const i0 = Math.floor(idx)
  const i1 = Math.min(i0 + 1, COOLWARM_COLORS.length - 1)
  const frac = idx - i0

  const c0 = COOLWARM_COLORS[i0]
  const c1 = COOLWARM_COLORS[i1]

  return [
    c0[0] + frac * (c1[0] - c0[0]),
    c0[1] + frac * (c1[1] - c0[1]),
    c0[2] + frac * (c1[2] - c0[2]),
  ]
}

// Terrain colormap - blue to green to brown to white
const TERRAIN_COLORS: [number, number, number][] = [
  [0.200, 0.400, 0.700],  // 0.0 - water blue
  [0.000, 0.600, 0.500],  // 0.2 - teal
  [0.200, 0.800, 0.400],  // 0.4 - green
  [0.600, 0.700, 0.300],  // 0.5 - yellow-green
  [0.800, 0.600, 0.400],  // 0.7 - tan
  [0.700, 0.500, 0.400],  // 0.8 - brown
  [1.000, 1.000, 1.000],  // 1.0 - white (peaks)
]

function terrain(t: number): [number, number, number] {
  const idx = t * (TERRAIN_COLORS.length - 1)
  const i0 = Math.floor(idx)
  const i1 = Math.min(i0 + 1, TERRAIN_COLORS.length - 1)
  const frac = idx - i0

  const c0 = TERRAIN_COLORS[i0]
  const c1 = TERRAIN_COLORS[i1]

  return [
    c0[0] + frac * (c1[0] - c0[0]),
    c0[1] + frac * (c1[1] - c0[1]),
    c0[2] + frac * (c1[2] - c0[2]),
  ]
}
