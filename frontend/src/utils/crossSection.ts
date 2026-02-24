/**
 * Cross-section geometry utilities for determining which side of a polyline
 * each cell center falls on. Used to create cutaway views in the 3D viewer.
 */

/**
 * Determine which side of a polyline a 2D point falls on.
 * Returns +1 (left of line direction) or -1 (right).
 *
 * For the first segment, the line extends backward to -infinity.
 * For the last segment, the line extends forward to +infinity.
 * This ensures the polyline divides the entire model domain.
 */
export function pointSideOfPolyline(
  px: number,
  py: number,
  polyline: [number, number][]
): number {
  let minDist = Infinity
  let bestCross = 0

  for (let s = 0; s < polyline.length - 1; s++) {
    const ax = polyline[s][0]
    const ay = polyline[s][1]
    const bx = polyline[s + 1][0]
    const by = polyline[s + 1][1]

    const dx = bx - ax
    const dy = by - ay
    const lenSq = dx * dx + dy * dy
    if (lenSq === 0) continue

    // Parametric projection of point onto segment
    let t = ((px - ax) * dx + (py - ay) * dy) / lenSq

    // Extend first segment backward, last segment forward
    const isFirst = s === 0
    const isLast = s === polyline.length - 2
    if (!isFirst && t < 0) t = 0
    if (!isLast && t > 1) t = 1

    // Closest point on (extended) segment
    const cx = ax + t * dx
    const cy = ay + t * dy
    const distSq = (px - cx) * (px - cx) + (py - cy) * (py - cy)

    if (distSq < minDist) {
      minDist = distSq
      // Cross product: positive = left of line direction, negative = right
      bestCross = dx * (py - ay) - dy * (px - ax)
    }
  }

  return bestCross >= 0 ? 1 : -1
}

/**
 * Build a visibility mask for all cells based on which side of the polyline
 * their centers fall on.
 *
 * @param polyline - Array of [x, y] points defining the cross-section line
 * @param centers - Flat Float32Array from gridData.centers (ncells * 3: x, y, z per cell)
 * @param ncells - Total number of cells
 * @param side - Which side to show: 'left', 'right', or 'both'
 * @returns Uint8Array of length ncells (1=visible, 0=hidden)
 */
export function computeCellMask(
  polyline: [number, number][],
  centers: Float32Array,
  ncells: number,
  side: 'left' | 'right' | 'both'
): Uint8Array {
  const mask = new Uint8Array(ncells)

  if (side === 'both' || polyline.length < 2) {
    mask.fill(1)
    return mask
  }

  const targetSide = side === 'left' ? 1 : -1

  for (let i = 0; i < ncells; i++) {
    const x = centers[i * 3]
    const y = centers[i * 3 + 1]
    const cellSide = pointSideOfPolyline(x, y, polyline)
    mask[i] = cellSide === targetSide ? 1 : 0
  }

  return mask
}
