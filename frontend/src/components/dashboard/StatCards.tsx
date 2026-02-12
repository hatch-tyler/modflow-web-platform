import { Droplets, Layers, Clock, CheckCircle2, XCircle, Loader2, AlertTriangle } from 'lucide-react'
import type { ResultsSummary, PostProcessProgress } from '../../types'

interface StatCardsProps {
  summary?: ResultsSummary
  compareSummary?: ResultsSummary
  awaitingProgress?: PostProcessProgress
}

export default function StatCards({ summary, compareSummary, awaitingProgress }: StatCardsProps) {
  // Show skeleton cards when awaiting data
  if (awaitingProgress || !summary) {
    const skeletonCards = [
      { label: 'Mass Balance Error', color: 'indigo' as const },
      { label: 'Head Range', color: 'blue' as const },
      { label: 'Stress Periods', color: 'purple' as const },
      { label: 'Convergence', color: 'slate' as const },
    ]

    const colorClasses: Record<string, string> = {
      indigo: 'bg-indigo-50 text-indigo-300',
      blue: 'bg-blue-50 text-blue-300',
      purple: 'bg-purple-50 text-purple-300',
      slate: 'bg-slate-100 text-slate-300',
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {skeletonCards.map((card) => (
          <div key={card.label} className="bg-white rounded-lg border border-slate-200 p-4">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${colorClasses[card.color]}`}>
                <Loader2 className="h-5 w-5 animate-spin" />
              </div>
              <div className="flex-1">
                <div className="text-sm text-slate-500">{card.label}</div>
                <div className="h-6 bg-slate-100 rounded animate-pulse mt-1 w-20" />
                <div className="h-3 bg-slate-50 rounded animate-pulse mt-2 w-16" />
              </div>
            </div>
          </div>
        ))}
      </div>
    )
  }

  const { convergence, heads_summary, metadata } = summary
  const massBalanceError = convergence.mass_balance_error_pct

  const headRange = (heads_summary.max_head != null && heads_summary.min_head != null)
    ? `${heads_summary.min_head.toFixed(2)} - ${heads_summary.max_head.toFixed(2)}`
    : 'N/A'

  // Compare values
  const cmpMbe = compareSummary?.convergence.mass_balance_error_pct
  const cmpHeadRange = (compareSummary?.heads_summary.max_head != null && compareSummary?.heads_summary.min_head != null)
    ? `${compareSummary.heads_summary.min_head.toFixed(2)} - ${compareSummary.heads_summary.max_head.toFixed(2)}`
    : null
  const cmpNper = compareSummary?.metadata.nper
  const cmpConverged = compareSummary?.convergence.converged

  const hasWarnings = convergence.warnings && convergence.warnings.length > 0

  // Determine convergence display
  let convergenceIcon = convergence.converged ? CheckCircle2 : XCircle
  let convergenceValue = convergence.converged ? 'Converged' : 'Failed'
  let convergenceSubtext = convergence.converged ? 'All periods converged' : 'Check listing file'
  let convergenceColor: 'green' | 'yellow' | 'red' = convergence.converged ? 'green' : 'red'

  if (convergence.converged && hasWarnings) {
    convergenceIcon = AlertTriangle
    convergenceValue = 'Converged'
    convergenceSubtext = `${convergence.warnings!.length} warning(s)`
    convergenceColor = 'yellow'
  }

  const cards = [
    {
      icon: Droplets,
      label: 'Mass Balance Error',
      value: massBalanceError != null ? `${massBalanceError.toFixed(4)}%` : 'N/A',
      compareValue: cmpMbe != null ? `${cmpMbe.toFixed(4)}%` : null,
      subtext: 'Cumulative discrepancy',
      color: 'indigo' as const,
    },
    {
      icon: Layers,
      label: 'Head Range',
      value: headRange,
      compareValue: cmpHeadRange,
      subtext: `${metadata.nlay} layer(s)`,
      color: 'blue' as const,
    },
    {
      icon: Clock,
      label: 'Stress Periods',
      value: String(metadata.nper),
      compareValue: cmpNper != null ? String(cmpNper) : null,
      subtext: `${heads_summary.nstp_total} total time step(s)`,
      color: 'purple' as const,
    },
    {
      icon: convergenceIcon,
      label: 'Convergence',
      value: convergenceValue,
      compareValue: cmpConverged != null ? (cmpConverged ? 'Converged' : 'Failed') : null,
      subtext: convergenceSubtext,
      color: convergenceColor,
      warnings: hasWarnings ? convergence.warnings : undefined,
    },
  ]

  const colorClasses: Record<string, string> = {
    green: 'bg-green-50 text-green-600',
    yellow: 'bg-yellow-50 text-yellow-600',
    blue: 'bg-blue-50 text-blue-600',
    indigo: 'bg-indigo-50 text-indigo-600',
    purple: 'bg-purple-50 text-purple-600',
    red: 'bg-red-50 text-red-600',
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => {
        const Icon = card.icon
        return (
          <div key={card.label} className="bg-white rounded-lg border border-slate-200 p-4">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-lg ${colorClasses[card.color]}`}>
                <Icon className="h-5 w-5" />
              </div>
              <div className="flex-1">
                <div className="text-sm text-slate-500">{card.label}</div>
                <div className="text-xl font-bold text-slate-800">{card.value}</div>
                {card.compareValue != null && (
                  <div className="text-sm text-orange-500 font-medium">vs {card.compareValue}</div>
                )}
                <div className="text-xs text-slate-400">{card.subtext}</div>
                {card.warnings && card.warnings.length > 0 && (
                  <div className="mt-1 space-y-0.5">
                    {card.warnings.map((warning, i) => (
                      <div key={i} className="text-xs text-yellow-600 truncate" title={warning}>
                        {warning}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
