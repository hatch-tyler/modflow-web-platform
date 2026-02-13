import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import type { ConvergenceDetail } from '../../types'

interface IterationHeatmapProps {
  convergenceDetail: ConvergenceDetail
  height: number
}

export default function IterationHeatmap({ convergenceDetail, height }: IterationHeatmapProps) {
  const plotData = useMemo(() => {
    const spSummary = convergenceDetail.stress_period_summary
    if (!spSummary.length) return null

    // Bar chart of max iterations per stress period with color by difficulty
    const kpers = spSummary.map(s => s.kper + 1) // 1-indexed for display
    const maxIters = spSummary.map(s => s.max_iterations)
    const avgIters = spSummary.map(s => s.avg_iterations)
    const colors = spSummary.map(s => {
      switch (s.difficulty) {
        case 'failed': return '#ef4444'
        case 'high': return '#f97316'
        case 'moderate': return '#eab308'
        case 'low': return '#22c55e'
        default: return '#94a3b8'
      }
    })

    // If we have per-timestep data, build a heatmap
    const timesteps = convergenceDetail.timesteps
    if (timesteps.length > 0 && timesteps.length <= 5000) {
      // Group timesteps by stress period
      const spGroups = new Map<number, typeof timesteps>()
      for (const ts of timesteps) {
        const group = spGroups.get(ts.kper) || []
        group.push(ts)
        spGroups.set(ts.kper, group)
      }

      const maxTsPerSp = Math.max(...Array.from(spGroups.values()).map(g => g.length), 1)

      // Build heatmap arrays
      const zData: (number | null)[][] = []
      const hoverText: string[][] = []
      const xLabels: string[] = []
      const yLabels: string[] = []

      for (let ts = 0; ts < maxTsPerSp; ts++) {
        yLabels.push(`TS ${ts + 1}`)
      }

      const sortedSps = Array.from(spGroups.keys()).sort((a, b) => a - b)
      for (const sp of sortedSps) {
        xLabels.push(`SP ${sp + 1}`)
        const group = spGroups.get(sp) || []
        const col: (number | null)[] = []
        const hoverCol: string[] = []

        for (let ts = 0; ts < maxTsPerSp; ts++) {
          if (ts < group.length) {
            const t = group[ts]
            col.push(t.outer_iterations)
            hoverCol.push(
              `SP ${sp + 1}, TS ${ts + 1}<br>` +
              `Iterations: ${t.outer_iterations}<br>` +
              `Converged: ${t.converged ? 'Yes' : 'NO'}<br>` +
              `Max dv: ${t.max_dvmax.toExponential(2)} (${t.max_dvmax_cell})<br>` +
              `Max res: ${t.max_rclose.toExponential(2)} (${t.max_rclose_cell})`
            )
          } else {
            col.push(null)
            hoverCol.push('')
          }
        }
        zData.push(col)
        hoverText.push(hoverCol)
      }

      // Transpose for Plotly (z[y][x])
      const zTransposed: (number | null)[][] = []
      const hoverTransposed: string[][] = []
      for (let y = 0; y < maxTsPerSp; y++) {
        const row: (number | null)[] = []
        const hrow: string[] = []
        for (let x = 0; x < sortedSps.length; x++) {
          row.push(zData[x]?.[y] ?? null)
          hrow.push(hoverText[x]?.[y] ?? '')
        }
        zTransposed.push(row)
        hoverTransposed.push(hrow)
      }

      // Mark failed timesteps
      const failedX: number[] = []
      const failedY: number[] = []
      for (const ts of timesteps) {
        if (!ts.converged) {
          const xIdx = sortedSps.indexOf(ts.kper)
          const group = spGroups.get(ts.kper) || []
          const yIdx = group.indexOf(ts)
          if (xIdx >= 0 && yIdx >= 0) {
            failedX.push(xIdx)
            failedY.push(yIdx)
          }
        }
      }

      return {
        type: 'heatmap' as const,
        traces: [
          {
            z: zTransposed,
            x: xLabels,
            y: yLabels,
            type: 'heatmap' as const,
            colorscale: [
              [0, '#22c55e'],
              [0.3, '#eab308'],
              [0.6, '#f97316'],
              [1, '#ef4444'],
            ],
            hovertext: hoverTransposed,
            hoverinfo: 'text' as const,
            colorbar: { title: 'Iterations', thickness: 15 },
          },
          ...(failedX.length > 0 ? [{
            x: failedX.map(i => xLabels[i]),
            y: failedY.map(i => yLabels[i]),
            mode: 'markers' as const,
            type: 'scatter' as const,
            marker: { symbol: 'x', size: 10, color: '#dc2626', line: { width: 2 } },
            name: 'Failed',
            hoverinfo: 'name' as const,
          }] : []),
        ],
        layout: {
          title: { text: 'Outer Iterations per Timestep' },
          xaxis: { title: { text: 'Stress Period' } },
          yaxis: { title: { text: 'Time Step' }, autorange: 'reversed' as const },
        },
      }
    }

    // Fallback: bar chart of stress period summary
    return {
      type: 'bar' as const,
      traces: [
        {
          x: kpers,
          y: maxIters,
          type: 'bar' as const,
          name: 'Max Iterations',
          marker: { color: colors },
          hovertemplate: 'SP %{x}<br>Max: %{y} iters<extra></extra>',
        },
        {
          x: kpers,
          y: avgIters,
          type: 'scatter' as const,
          mode: 'lines' as const,
          name: 'Avg Iterations',
          line: { color: '#3b82f6', width: 2 },
        },
      ],
      layout: {
        title: { text: 'Solver Iterations per Stress Period' },
        xaxis: { title: { text: 'Stress Period' } },
        yaxis: { title: { text: 'Outer Iterations' } },
      },
    }
  }, [convergenceDetail])

  if (!plotData) {
    return (
      <div className="flex items-center justify-center text-slate-400" style={{ height }}>
        No convergence data available
      </div>
    )
  }

  return (
    <Plot
      data={plotData.traces as Plotly.Data[]}
      layout={{
        ...plotData.layout,
        height,
        margin: { t: 40, b: 50, l: 60, r: 30 },
        font: { family: 'Inter, system-ui, sans-serif', size: 12 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        showlegend: plotData.type === 'bar',
      }}
      config={{ responsive: true, displayModeBar: true, displaylogo: false }}
      style={{ width: '100%' }}
    />
  )
}
