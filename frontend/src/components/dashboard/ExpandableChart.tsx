import { useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Maximize2, X } from 'lucide-react'

interface ExpandableChartProps {
  title: string
  children: (expanded: boolean) => ReactNode
}

export default function ExpandableChart({ title, children }: ExpandableChartProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <>
      <div className="bg-white rounded-lg border border-slate-200 p-4 relative">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-slate-800">{title}</h3>
          <button
            onClick={() => setExpanded(true)}
            className="p-1.5 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600"
            title="Expand to fullscreen"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
        </div>
        {children(false)}
      </div>

      {expanded &&
        createPortal(
          <div className="fixed inset-0 z-50 flex items-center justify-center">
            <div
              className="absolute inset-0 bg-black/50"
              onClick={() => setExpanded(false)}
            />
            <div className="relative m-4 w-full h-full bg-white rounded-lg shadow-2xl flex flex-col overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200">
                <h3 className="font-semibold text-slate-800">{title}</h3>
                <button
                  onClick={() => setExpanded(false)}
                  className="p-1.5 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600"
                  title="Close"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <div className="flex-1 p-4 overflow-auto">
                {children(true)}
              </div>
            </div>
          </div>,
          document.body
        )}
    </>
  )
}
