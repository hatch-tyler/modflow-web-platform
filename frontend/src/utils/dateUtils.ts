/**
 * Compute cumulative end-time for each stress period by summing perlen values.
 *
 * @param stressPeriodData - Array of {perlen, nstp, tsmult} per stress period
 * @returns Array of cumulative end-times (one per stress period)
 */
export function kperToEndTimes(
  stressPeriodData: { perlen: number; nstp: number; tsmult: number }[],
): number[] {
  const endTimes: number[] = []
  let cumulative = 0
  for (const sp of stressPeriodData) {
    cumulative += sp.perlen
    endTimes.push(cumulative)
  }
  return endTimes
}

/**
 * Convert cumulative simulation times to ISO date strings.
 *
 * @param startDate - ISO date string (e.g., "2000-01-01")
 * @param times - Array of cumulative simulation times
 * @param timeUnit - Time unit: "seconds", "minutes", "hours", "days", "years"
 * @returns Array of ISO date strings, or null if startDate is not provided
 */
export function timesToDates(
  startDate: string | null | undefined,
  times: number[],
  timeUnit: string | null | undefined,
): string[] | null {
  if (!startDate || !times || times.length === 0) return null

  const base = new Date(startDate + 'T00:00:00Z')
  if (isNaN(base.getTime())) return null

  // Map time unit to a multiplier that converts to days
  const unitToDays: Record<string, number> = {
    seconds: 1 / 86400,
    minutes: 1 / 1440,
    hours: 1 / 24,
    days: 1,
    years: 365.25,
  }

  const dayMultiplier = unitToDays[(timeUnit || 'days').toLowerCase()] ?? 1

  return times.map((t) => {
    const daysOffset = t * dayMultiplier
    const ms = base.getTime() + daysOffset * 86400000
    return new Date(ms).toISOString().split('T')[0]
  })
}
