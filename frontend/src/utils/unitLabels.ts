/** Abbreviations for MODFLOW length units */
const LENGTH_ABBREV: Record<string, string> = {
  meters: 'm',
  feet: 'ft',
  centimeters: 'cm',
}

/** Abbreviations for MODFLOW time units */
const TIME_ABBREV: Record<string, string> = {
  seconds: 's',
  minutes: 'min',
  hours: 'hr',
  days: 'd',
  years: 'yr',
}

/** Return abbreviated length unit, e.g. "meters" → "m". Empty string if undefined/unknown. */
export function lengthAbbrev(unit?: string | null): string {
  if (!unit || unit === 'undefined') return ''
  return LENGTH_ABBREV[unit.toLowerCase()] ?? unit
}

/** Return abbreviated time unit, e.g. "days" → "d". Empty string if undefined/unknown. */
export function timeAbbrev(unit?: string | null): string {
  if (!unit || unit === 'undefined') return ''
  return TIME_ABBREV[unit.toLowerCase()] ?? unit
}

/** Build an axis / colorbar label with optional unit, e.g. "Head (m)" or just "Head". */
export function labelWithUnit(label: string, unit: string): string {
  return unit ? `${label} (${unit})` : label
}

/** Build a volumetric rate label, e.g. "Flow Rate (m³/d)" */
export function flowRateLabel(lengthUnit?: string | null, timeUnit?: string | null): string {
  const l = lengthAbbrev(lengthUnit)
  const t = timeAbbrev(timeUnit)
  if (l && t) return `Flow Rate (${l}\u00B3/${t})`
  if (l) return `Flow Rate (${l}\u00B3)`
  return 'Flow Rate'
}

/** Build a cumulative volume label, e.g. "Cumulative Volume (m³)" */
export function volumeLabel(lengthUnit?: string | null): string {
  const l = lengthAbbrev(lengthUnit)
  if (l) return `Cumulative Volume (${l}\u00B3)`
  return 'Cumulative Volume'
}
