import { useState, useMemo } from 'react'
import Plot from 'react-plotly.js'
import { Loader2 } from 'lucide-react'
import type { ConvergenceDetail, StressSummary } from '../../types'

interface StressCorrelationChartProps {
  convergenceDetail: ConvergenceDetail
  stressData: StressSummary | null
  height: number
  loading: boolean
}

const PACKAGE_COLORS: Record<string, string> = {
  WEL: '#3b82f6',
  MAW: '#8b5cf6',
  RCH: '#22c55e',
  EVT: '#ef4444',
  SFR: '#06b6d4',
  GHB: '#f97316',
  CHD: '#ec4899',
  DRN: '#a855f7',
  RIV: '#14b8a6',
  LAK: '#6366f1',
  UZF: '#84cc16',
}

export default function StressCorrelationChart({
  convergenceDetail,
  stressData,
  height,
  loading,
}: StressCorrelationChartProps) {
  const packages = stressData?.packages || []
  const [enabledPackages, setEnabledPackages] = useState<Set<string>>(new Set(packages))

  // Update enabled packages when stress data loads
  useMemo(() => {
    if (packages.length > 0 && enabledPackages.size === 0) {
      setEnabledPackages(new Set(packages))
    }
  }, [packages])

  const traces = useMemo(() => {
    const spSummary = convergenceDetail.stress_period_summary
    const kpers = spSummary.map(s => s.kper + 1)
    const result: Plotly.Data[] = []

    // Iteration line (primary y-axis)
    result.push({
      x: kpers,
      y: spSummary.map(s => s.max_iterations),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Max Iterations',
      line: { color: '#1e293b', width: 2 },
      marker: {
        size: 6,
        color: spSummary.map(s => s.difficulty === 'failed' ? '#ef4444' : '#1e293b'),
      },
      yaxis: 'y',
    })

    // Average iterations line
    result.push({
      x: kpers,
      y: spSummary.map(s => s.avg_iterations),
      type: 'scatter',
      mode: 'lines',
      name: 'Avg Iterations',
      line: { color: '#94a3b8', width: 1, dash: 'dash' },
      yaxis: 'y',
    })

    // Package stress bars (secondary y-axis)
    if (stressData) {
      for (const pkg of packages) {
        if (!enabledPackages.has(pkg)) continue

        const rates = stressData.periods.map(p => {
          const pkgData = p[pkg]
          if (typeof pkgData === 'object' && pkgData !== null && 'total_rate' in pkgData) {
            return Math.abs(pkgData.total_rate)
          }
          return 0
        })

        result.push({
          x: stressData.periods.map(p => p.kper + 1),
          y: rates,
          type: 'bar',
          name: pkg,
          marker: { color: PACKAGE_COLORS[pkg] || '#94a3b8', opacity: 0.6 },
          yaxis: 'y2',
        })
      }
    }

    return result
  }, [convergenceDetail, stressData, enabledPackages])

  if (loading) {
    return (
      <div className="flex items-center justify-center" style={{ height }}>
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <span className="ml-3 text-slate-500">Loading stress data...</span>
      </div>
    )
  }

  const togglePackage = (pkg: string) => {
    setEnabledPackages(prev => {
      const next = new Set(prev)
      if (next.has(pkg)) {
        next.delete(pkg)
      } else {
        next.add(pkg)
      }
      return next
    })
  }

  return (
    <div>
      {/* Package toggles */}
      {packages.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {packages.map(pkg => (
            <button
              key={pkg}
              onClick={() => togglePackage(pkg)}
              className={`px-2.5 py-1 text-xs font-medium rounded-full border transition-colors ${
                enabledPackages.has(pkg)
                  ? 'border-current text-white'
                  : 'border-slate-300 text-slate-400 bg-white'
              }`}
              style={enabledPackages.has(pkg) ? {
                backgroundColor: PACKAGE_COLORS[pkg] || '#94a3b8',
                borderColor: PACKAGE_COLORS[pkg] || '#94a3b8',
              } : {}}
            >
              {pkg}
            </button>
          ))}
        </div>
      )}

      <Plot
        data={traces}
        layout={{
          height: height - 40,
          margin: { t: 30, b: 50, l: 60, r: 60 },
          font: { family: 'Inter, system-ui, sans-serif', size: 12 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          barmode: 'group',
          showlegend: true,
          legend: { orientation: 'h', y: -0.15 },
          xaxis: { title: { text: 'Stress Period' } },
          yaxis: {
            title: { text: 'Outer Iterations' },
            side: 'left',
            showgrid: true,
            gridcolor: '#e2e8f0',
          },
          yaxis2: {
            title: { text: 'Abs. Rate (model units)' },
            side: 'right',
            overlaying: 'y',
            showgrid: false,
          },
        }}
        config={{ responsive: true, displayModeBar: true, displaylogo: false }}
        style={{ width: '100%' }}
      />
    </div>
  )
}
