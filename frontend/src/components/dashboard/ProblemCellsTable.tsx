import type { ProblemCell } from '../../types'

interface ProblemCellsTableProps {
  problemCells: ProblemCell[]
  height: number
}

export default function ProblemCellsTable({ problemCells, height }: ProblemCellsTableProps) {
  if (problemCells.length === 0) {
    return (
      <div className="flex items-center justify-center text-slate-400" style={{ height }}>
        No problem cells identified. All cells converged normally.
      </div>
    )
  }

  return (
    <div className="overflow-auto" style={{ maxHeight: height }}>
      <table className="w-full text-sm">
        <thead className="bg-slate-50 sticky top-0">
          <tr>
            <th className="px-4 py-2 text-left font-medium text-slate-600">Rank</th>
            <th className="px-4 py-2 text-left font-medium text-slate-600">Cell ID</th>
            <th className="px-4 py-2 text-right font-medium text-slate-600">Occurrences</th>
            <th className="px-4 py-2 text-left font-medium text-slate-600">Type</th>
            <th className="px-4 py-2 text-left font-medium text-slate-600">Affected Stress Periods</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {problemCells.map((cell, idx) => (
            <tr key={cell.cell_id} className="hover:bg-slate-50">
              <td className="px-4 py-2 text-slate-400">{idx + 1}</td>
              <td className="px-4 py-2 font-mono text-slate-800">{cell.cell_id}</td>
              <td className="px-4 py-2 text-right">
                <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                  cell.occurrences > 50
                    ? 'bg-red-100 text-red-700'
                    : cell.occurrences > 20
                    ? 'bg-amber-100 text-amber-700'
                    : 'bg-slate-100 text-slate-600'
                }`}>
                  {cell.occurrences}
                </span>
              </td>
              <td className="px-4 py-2 text-slate-600 capitalize">{cell.type.replace(/_/g, ' ')}</td>
              <td className="px-4 py-2">
                <div className="flex flex-wrap gap-1">
                  {cell.affected_sps.slice(0, 10).map(sp => (
                    <span key={sp} className="px-1.5 py-0.5 bg-blue-50 text-blue-600 rounded text-xs">
                      SP {sp + 1}
                    </span>
                  ))}
                  {cell.affected_sps.length > 10 && (
                    <span className="px-1.5 py-0.5 text-slate-400 text-xs">
                      +{cell.affected_sps.length - 10} more
                    </span>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
