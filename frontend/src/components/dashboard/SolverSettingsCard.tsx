import { Settings } from 'lucide-react'
import type { SolverSettings } from '../../types'

interface SolverSettingsCardProps {
  solverSettings: SolverSettings
}

function formatValue(value: unknown): string {
  if (value === undefined || value === null) return '-'
  if (typeof value === 'number') {
    if (Math.abs(value) < 0.001 || Math.abs(value) > 1e6) {
      return value.toExponential(2)
    }
    return value.toString()
  }
  return String(value)
}

export default function SolverSettingsCard({ solverSettings }: SolverSettingsCardProps) {
  const isMf6 = solverSettings.solver_type === 'IMS'

  const sections = isMf6
    ? [
        {
          title: 'General',
          items: [
            { label: 'Solver Type', value: solverSettings.solver_type },
            { label: 'Complexity', value: solverSettings.complexity },
            { label: 'Linear Acceleration', value: solverSettings.linear_acceleration },
          ],
        },
        {
          title: 'Nonlinear (Outer) Settings',
          items: [
            { label: 'OUTER_DVCLOSE', value: solverSettings.outer_dvclose, desc: 'Head change criterion' },
            { label: 'OUTER_MAXIMUM', value: solverSettings.outer_maximum, desc: 'Max outer iterations' },
            { label: 'Under-Relaxation', value: solverSettings.under_relaxation },
            { label: 'Backtracking #', value: solverSettings.backtracking_number },
          ],
        },
        {
          title: 'Linear (Inner) Settings',
          items: [
            { label: 'INNER_DVCLOSE', value: solverSettings.inner_dvclose, desc: 'Inner head change criterion' },
            { label: 'INNER_RCLOSE', value: solverSettings.inner_rclose, desc: 'Residual criterion' },
            { label: 'INNER_MAXIMUM', value: solverSettings.inner_maximum, desc: 'Max inner iterations' },
          ],
        },
      ]
    : [
        {
          title: 'Solver Configuration',
          items: [
            { label: 'Solver Type', value: solverSettings.solver_type },
            { label: 'Max Outer Iterations', value: solverSettings.outer_maximum },
            { label: 'Max Inner Iterations', value: solverSettings.inner_maximum },
            { label: 'HCLOSE', value: solverSettings.hclose, desc: 'Head closure criterion' },
            { label: 'RCLOSE', value: solverSettings.rclose, desc: 'Residual closure criterion' },
          ],
        },
      ]

  return (
    <div className="max-w-2xl">
      <div className="flex items-center gap-2 mb-4">
        <Settings className="h-5 w-5 text-slate-500" />
        <h3 className="text-lg font-medium text-slate-800">
          {solverSettings.solver_type || 'Solver'} Configuration
        </h3>
      </div>

      <div className="space-y-6">
        {sections.map((section) => (
          <div key={section.title}>
            <h4 className="text-sm font-medium text-slate-500 mb-2">{section.title}</h4>
            <div className="bg-slate-50 rounded-lg divide-y divide-slate-200">
              {section.items
                .filter((item) => item.value !== undefined && item.value !== null)
                .map((item) => (
                  <div key={item.label} className="flex items-center justify-between px-4 py-2.5">
                    <div>
                      <span className="text-sm text-slate-700">{item.label}</span>
                      {'desc' in item && item.desc && (
                        <span className="ml-2 text-xs text-slate-400">{item.desc}</span>
                      )}
                    </div>
                    <span className="font-mono text-sm font-medium text-slate-800">
                      {formatValue(item.value)}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
