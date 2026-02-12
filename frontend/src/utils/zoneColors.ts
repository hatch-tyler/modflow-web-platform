export const MAX_ZONES = 20

export function zoneColor(index: number, total: number) {
  const hue = (index * 360 / Math.max(total, 1)) % 360
  return {
    bg: `hsla(${hue}, 70%, 50%, 0.4)`,
    border: `hsl(${hue}, 70%, 50%)`,
    label: `hsl(${hue}, 70%, 45%)`,
  }
}

// Per-layer zone assignments: layer -> { cellIdx -> zoneNum }
export type LayerZoneAssignments = Record<number, Record<number, number>>
