// Shared color scale utilities for deck.gl and Plotly head visualizations

// --- Viridis for deck.gl (8-stop, position + RGB) ---

export const VIRIDIS_STOPS: [number, [number, number, number]][] = [
  [0.0, [68, 1, 84]],
  [0.14, [72, 36, 117]],
  [0.29, [56, 88, 140]],
  [0.43, [39, 127, 142]],
  [0.57, [31, 161, 135]],
  [0.71, [74, 194, 109]],
  [0.86, [159, 218, 58]],
  [1.0, [253, 231, 37]],
]

/** Map a value in [vmin, vmax] to an RGBA tuple for deck.gl PolygonLayer. */
export function valueToColor(
  value: number,
  vmin: number,
  vmax: number,
  opacity: number = 200,
): [number, number, number, number] {
  const range = vmax - vmin
  if (range === 0) return [39, 127, 142, opacity]
  const t = Math.max(0, Math.min(1, (value - vmin) / range))

  for (let i = 0; i < VIRIDIS_STOPS.length - 1; i++) {
    const [t0, c0] = VIRIDIS_STOPS[i]
    const [t1, c1] = VIRIDIS_STOPS[i + 1]
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0)
      return [
        Math.round(c0[0] + f * (c1[0] - c0[0])),
        Math.round(c0[1] + f * (c1[1] - c0[1])),
        Math.round(c0[2] + f * (c1[2] - c0[2])),
        opacity,
      ]
    }
  }
  return [253, 231, 37, opacity]
}

// --- Viridis for Plotly (11-stop, evenly spaced RGB) ---

const VIRIDIS_PLOTLY_STOPS: [number, number, number][] = [
  [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
  [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
  [121, 209, 81], [189, 222, 38], [253, 231, 37],
]

/** Interpolate a 0-1 parameter into a CSS rgb() string on the Viridis ramp. */
export function viridisColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t))
  const scaled = clamped * (VIRIDIS_PLOTLY_STOPS.length - 1)
  const i = Math.floor(scaled)
  const f = scaled - i
  const a = VIRIDIS_PLOTLY_STOPS[Math.min(i, VIRIDIS_PLOTLY_STOPS.length - 1)]
  const b = VIRIDIS_PLOTLY_STOPS[Math.min(i + 1, VIRIDIS_PLOTLY_STOPS.length - 1)]
  const r = Math.round(a[0] + (b[0] - a[0]) * f)
  const g = Math.round(a[1] + (b[1] - a[1]) * f)
  const bl = Math.round(a[2] + (b[2] - a[2]) * f)
  return `rgb(${r},${g},${bl})`
}

// --- RdBu diverging scale for difference maps ---

export const DIFF_COLORSCALE: [number, string][] = [
  [0, '#2166ac'],
  [0.25, '#67a9cf'],
  [0.5, '#f7f7f7'],
  [0.75, '#ef8a62'],
  [1, '#b2182b'],
]

/** Map a difference value to an RGBA tuple for deck.gl, centered on 0. */
export function diffValueToColor(
  value: number,
  absMax: number,
  opacity: number = 200,
): [number, number, number, number] {
  if (absMax === 0) return [247, 247, 247, opacity]
  const t = Math.max(0, Math.min(1, (value + absMax) / (2 * absMax)))

  for (let i = 0; i < DIFF_COLORSCALE.length - 1; i++) {
    const [t0, c0] = DIFF_COLORSCALE[i]
    const [t1, c1] = DIFF_COLORSCALE[i + 1]
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0)
      const r0 = parseInt(c0.slice(1, 3), 16)
      const g0 = parseInt(c0.slice(3, 5), 16)
      const b0 = parseInt(c0.slice(5, 7), 16)
      const r1 = parseInt(c1.slice(1, 3), 16)
      const g1 = parseInt(c1.slice(3, 5), 16)
      const b1 = parseInt(c1.slice(5, 7), 16)
      return [
        Math.round(r0 + (r1 - r0) * f),
        Math.round(g0 + (g1 - g0) * f),
        Math.round(b0 + (b1 - b0) * f),
        opacity,
      ]
    }
  }
  return [178, 24, 43, opacity]
}

/** CSS gradient for Viridis color legend bars. */
export const VIRIDIS_GRADIENT_CSS =
  'linear-gradient(to right, rgb(68,1,84), rgb(56,88,140), rgb(31,161,135), rgb(159,218,58), rgb(253,231,37))'

/** CSS gradient for RdBu difference legend bars. */
export const DIFF_GRADIENT_CSS =
  'linear-gradient(to right, #2166ac, #67a9cf, #f7f7f7, #ef8a62, #b2182b)'
