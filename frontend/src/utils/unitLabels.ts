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

// ─── Flow Unit Conversion ────────────────────────────────────────────────────

export interface FlowConversion {
  label: string
  factor: number  // multiply native value by this
}

/** Seconds per day */
const SEC_PER_DAY = 86400
/** Seconds per year */
const SEC_PER_YEAR = 365.25 * SEC_PER_DAY
/** Square feet per acre */
const SQFT_PER_ACRE = 43560
/** Cubic meters per acre-foot */
const M3_PER_ACREFT = 1233.48184

/**
 * Get the number of seconds per one native time unit.
 * Returns 1 for unknown units (identity).
 */
function secondsPerTimeUnit(timeUnit?: string): number {
  switch ((timeUnit || '').toLowerCase()) {
    case 'seconds': return 1
    case 'minutes': return 60
    case 'hours': return 3600
    case 'days': return SEC_PER_DAY
    case 'years': return SEC_PER_YEAR
    default: return SEC_PER_DAY  // assume days if unknown
  }
}

/**
 * Get available flow rate conversions based on model's length and time units.
 * First entry is always "Native" (factor = 1).
 */
export function getFlowConversions(lengthUnit?: string, timeUnit?: string): FlowConversion[] {
  const l = (lengthUnit || '').toLowerCase()
  const nativeLabel = flowRateLabel(lengthUnit, timeUnit) || 'Native'
  const conversions: FlowConversion[] = [
    { label: nativeLabel, factor: 1 },
  ]

  // Seconds in one native time unit
  const secPerNative = secondsPerTimeUnit(timeUnit)

  if (l === 'feet') {
    // ft³/native → acre-ft/d
    const ft3PerSec = 1 / secPerNative  // ft³/s per 1 ft³/native
    const acreftPerSec = ft3PerSec / SQFT_PER_ACRE  // acre-ft/s
    const acreftPerDay = acreftPerSec * SEC_PER_DAY
    conversions.push({ label: 'acre-ft/d', factor: acreftPerDay })

    // ft³/native → acre-ft/yr
    const acreftPerYear = acreftPerSec * SEC_PER_YEAR
    conversions.push({ label: 'acre-ft/yr', factor: acreftPerYear })
  } else if (l === 'meters') {
    // m³/native → acre-ft/d
    const m3PerSec = 1 / secPerNative
    const acreftPerSec = m3PerSec / M3_PER_ACREFT
    const acreftPerDay = acreftPerSec * SEC_PER_DAY
    conversions.push({ label: 'acre-ft/d', factor: acreftPerDay })

    // m³/native → acre-ft/yr
    const acreftPerYear = acreftPerSec * SEC_PER_YEAR
    conversions.push({ label: 'acre-ft/yr', factor: acreftPerYear })
  } else if (l === 'centimeters') {
    // cm³/native → acre-ft/d  (1 cm³ = 1e-6 m³)
    const m3PerSec = 1e-6 / secPerNative
    const acreftPerSec = m3PerSec / M3_PER_ACREFT
    conversions.push({ label: 'acre-ft/d', factor: acreftPerSec * SEC_PER_DAY })
    conversions.push({ label: 'acre-ft/yr', factor: acreftPerSec * SEC_PER_YEAR })
  }

  return conversions
}
