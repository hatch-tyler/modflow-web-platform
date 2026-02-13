import { useState } from 'react'
import { Save, Undo2, ChevronDown, Loader2, FileText } from 'lucide-react'
import type { BackupInfo } from '../../types'

interface EditorToolbarProps {
  filePath: string
  isDirty: boolean
  saving: boolean
  backups: BackupInfo[]
  onSave: () => void
  onRevert: (timestamp: string) => void
}

export default function EditorToolbar({
  filePath,
  isDirty,
  saving,
  backups,
  onSave,
  onRevert,
}: EditorToolbarProps) {
  const [showBackups, setShowBackups] = useState(false)

  // File path breadcrumbs
  const pathParts = filePath.split('/')

  return (
    <div className="flex items-center justify-between h-10 px-3 bg-white border-b border-slate-200">
      {/* Left: file path breadcrumb + dirty indicator */}
      <div className="flex items-center gap-2 min-w-0">
        <FileText className="h-4 w-4 text-slate-400 flex-shrink-0" />
        <div className="flex items-center text-sm text-slate-500 truncate">
          {pathParts.map((part, i) => (
            <span key={i} className="flex items-center">
              {i > 0 && <span className="mx-1 text-slate-300">/</span>}
              <span className={i === pathParts.length - 1 ? 'text-slate-800 font-medium' : ''}>
                {part}
              </span>
            </span>
          ))}
        </div>
        {isDirty && (
          <span className="flex-shrink-0 w-2 h-2 rounded-full bg-orange-400" title="Unsaved changes" />
        )}
      </div>

      {/* Right: actions */}
      <div className="flex items-center gap-2">
        {/* Revert dropdown */}
        {backups.length > 0 && (
          <div className="relative">
            <button
              onClick={() => setShowBackups(!showBackups)}
              className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded transition-colors"
            >
              <Undo2 className="h-3.5 w-3.5" />
              Revert
              <ChevronDown className="h-3 w-3" />
            </button>

            {showBackups && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setShowBackups(false)} />
                <div className="absolute right-0 top-full mt-1 z-20 w-56 bg-white border border-slate-200 rounded-lg shadow-lg py-1">
                  <div className="px-3 py-1.5 text-xs font-medium text-slate-400">
                    Previous versions
                  </div>
                  {backups.map((backup) => (
                    <button
                      key={backup.timestamp}
                      onClick={() => {
                        onRevert(backup.timestamp)
                        setShowBackups(false)
                      }}
                      className="w-full px-3 py-1.5 text-left text-sm text-slate-600 hover:bg-slate-50"
                    >
                      <span className="font-mono text-xs">{backup.timestamp}</span>
                      <span className="ml-2 text-slate-400">
                        ({(backup.size / 1024).toFixed(1)} KB)
                      </span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* Save button */}
        <button
          onClick={onSave}
          disabled={!isDirty || saving}
          className="flex items-center gap-1 px-3 py-1 text-xs font-medium text-white bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {saving ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Save className="h-3.5 w-3.5" />
          )}
          Save
        </button>
      </div>
    </div>
  )
}
