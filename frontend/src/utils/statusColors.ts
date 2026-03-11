/**
 * Status color utilities for run status badges across the application.
 *
 * Two variants:
 * - `getStatusBadgeClass`: compact badge (bg + text) for inline use
 * - `getStatusCardClass`: card variant (bg + text + border) for panels
 */

/** Compact badge classes (background + text). Used in sidebar badges, run lists. */
export function getStatusBadgeClass(status: string): string {
  switch (status) {
    case 'running':
      return 'bg-blue-100 text-blue-800'
    case 'completed':
      return 'bg-green-100 text-green-800'
    case 'failed':
      return 'bg-red-100 text-red-800'
    case 'cancelled':
      return 'bg-yellow-100 text-yellow-800'
    case 'pending':
    case 'queued':
      return 'bg-slate-100 text-slate-800'
    default:
      return 'bg-slate-100 text-slate-600'
  }
}

/** Card classes (border + background + text). Used in status panels, dashboard cards. */
export function getStatusCardClass(status: string): string {
  switch (status) {
    case 'running':
      return 'text-blue-600 bg-blue-50 border-blue-200'
    case 'completed':
      return 'text-green-600 bg-green-50 border-green-200'
    case 'failed':
      return 'text-red-600 bg-red-50 border-red-200'
    case 'cancelled':
      return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    case 'pending':
    case 'queued':
      return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    default:
      return 'text-slate-600 bg-slate-50 border-slate-200'
  }
}
