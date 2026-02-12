import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ChevronRight,
  ChevronDown,
  File,
  Folder,
  Trash2,
  FileSpreadsheet,
  Settings,
  AlertTriangle,
  CheckCircle,
  Search,
  X,
  Loader2,
} from 'lucide-react'
import clsx from 'clsx'
import { filesApi } from '../../services/api'
import type { FileCategory, FileInfo, CategorizedFiles } from '../../types'

interface FileBrowserProps {
  projectId: string
  onMarkAsObservation?: (filePath: string) => void
}

// Category metadata for display
const CATEGORY_META: Record<FileCategory, { label: string; icon: React.ElementType; color: string; badge?: string }> = {
  model_core: {
    label: 'Core Model Files',
    icon: CheckCircle,
    color: 'text-green-600',
    badge: 'Core',
  },
  model_input: {
    label: 'Model Input Files',
    icon: File,
    color: 'text-blue-600',
    badge: 'Input',
  },
  model_output: {
    label: 'Model Output Files',
    icon: File,
    color: 'text-slate-500',
  },
  pest: {
    label: 'PEST/pyEMU Files',
    icon: Settings,
    color: 'text-purple-600',
  },
  observation: {
    label: 'Observation Data',
    icon: FileSpreadsheet,
    color: 'text-orange-500',
  },
  blocked: {
    label: 'Blocked Files',
    icon: AlertTriangle,
    color: 'text-red-600',
  },
  other: {
    label: 'Other Files',
    icon: File,
    color: 'text-slate-400',
  },
}

const CATEGORY_ORDER: FileCategory[] = [
  'model_core',
  'model_input',
  'model_output',
  'pest',
  'observation',
  'other',
  'blocked',
]

function formatFileSize(bytes: number): string {
  if (bytes === 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

interface FileRowProps {
  file: FileInfo
  category: FileCategory
  onDelete: (path: string) => void
  onMarkAsObs?: (path: string) => void
  isDeleting: boolean
}

function FileRow({ file, category, onDelete, onMarkAsObs, isDeleting }: FileRowProps) {
  const [showConfirm, setShowConfirm] = useState(false)
  const isCore = category === 'model_core'
  const isInput = category === 'model_input'
  const isObservation = category === 'observation'
  const isCsv = file.extension === '.csv'

  return (
    <div className="flex items-center gap-2 py-1.5 px-2 hover:bg-slate-50 rounded group">
      <File className="h-4 w-4 text-slate-400 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm truncate" title={file.path}>
            {file.name}
          </span>
          {isCore && (
            <span className="text-[10px] px-1.5 py-0.5 bg-green-100 text-green-700 rounded font-medium">
              Core
            </span>
          )}
          {isInput && (
            <span className="text-[10px] px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded font-medium">
              Input
            </span>
          )}
        </div>
        {file.description && (
          <div className="text-xs text-slate-400 truncate">{file.description}</div>
        )}
      </div>
      <span className="text-xs text-slate-400 flex-shrink-0">
        {formatFileSize(file.size)}
      </span>

      {/* Actions */}
      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        {isCsv && onMarkAsObs && (
          <button
            onClick={() => onMarkAsObs(file.path)}
            className="p-1 text-orange-400 hover:text-orange-600 hover:bg-orange-50 rounded"
            title={isObservation ? "Configure observation data" : "Mark as observation data"}
          >
            <FileSpreadsheet className="h-3.5 w-3.5" />
          </button>
        )}

        {showConfirm ? (
          <div className="flex items-center gap-1">
            <button
              onClick={() => {
                onDelete(file.path)
                setShowConfirm(false)
              }}
              disabled={isDeleting}
              className="px-2 py-0.5 text-xs bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50"
            >
              {isDeleting ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Delete'}
            </button>
            <button
              onClick={() => setShowConfirm(false)}
              className="px-2 py-0.5 text-xs border border-slate-300 rounded hover:bg-slate-100"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setShowConfirm(true)}
            className="p-1 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded"
            title="Delete file"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  )
}

interface CategorySectionProps {
  category: FileCategory
  files: FileInfo[]
  onDelete: (path: string) => void
  onMarkAsObs?: (path: string) => void
  isDeleting: boolean
  deletingPath: string | null
}

function CategorySection({
  category,
  files,
  onDelete,
  onMarkAsObs,
  isDeleting,
  deletingPath,
}: CategorySectionProps) {
  const [isExpanded, setIsExpanded] = useState(
    category === 'model_core' || category === 'model_input' || category === 'observation'
  )
  const meta = CATEGORY_META[category]
  const Icon = meta.icon

  if (files.length === 0) return null

  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-slate-50 hover:bg-slate-100 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-4 w-4 text-slate-400" />
        ) : (
          <ChevronRight className="h-4 w-4 text-slate-400" />
        )}
        <Icon className={clsx('h-4 w-4', meta.color)} />
        <span className="font-medium text-sm text-slate-700">{meta.label}</span>
        <span className="text-xs text-slate-400">({files.length})</span>
      </button>

      {isExpanded && (
        <div className="divide-y divide-slate-100">
          {files.map((file) => (
            <FileRow
              key={file.path}
              file={file}
              category={category}
              onDelete={onDelete}
              onMarkAsObs={onMarkAsObs}
              isDeleting={isDeleting && deletingPath === file.path}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default function FileBrowser({ projectId, onMarkAsObservation }: FileBrowserProps) {
  const queryClient = useQueryClient()
  const [searchTerm, setSearchTerm] = useState('')
  const [deletingPath, setDeletingPath] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['categorized-files', projectId],
    queryFn: () => filesApi.getCategorized(projectId),
    enabled: !!projectId,
  })

  const deleteMutation = useMutation({
    mutationFn: (path: string) => {
      setDeletingPath(path)
      return filesApi.deleteFile(projectId, path)
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['categorized-files', projectId] })
      if (result.warning) {
        // Could show a toast here
        console.warn(result.warning)
      }
    },
    onSettled: () => {
      setDeletingPath(null)
    },
  })

  // Filter files by search term
  const filteredData = useMemo(() => {
    if (!data || !searchTerm) return data

    const term = searchTerm.toLowerCase()
    const filtered: CategorizedFiles = {
      ...data,
      categories: {} as Record<FileCategory, FileInfo[]>,
    }

    for (const [cat, files] of Object.entries(data.categories)) {
      filtered.categories[cat as FileCategory] = files.filter(
        (f) =>
          f.name.toLowerCase().includes(term) ||
          f.path.toLowerCase().includes(term) ||
          f.description.toLowerCase().includes(term)
      )
    }

    return filtered
  }, [data, searchTerm])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
        <span className="ml-2 text-slate-500">Loading files...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
        <AlertTriangle className="h-5 w-5 inline mr-2" />
        Failed to load files
      </div>
    )
  }

  if (!data || data.total_files === 0) {
    return (
      <div className="bg-slate-50 rounded-lg p-6 text-center text-slate-400">
        <Folder className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p>No files uploaded yet</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header with stats */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-600">
          <span className="font-medium">{data.total_files}</span> files
          {data.total_size_mb > 0 && (
            <span className="ml-2 text-slate-400">
              ({data.total_size_mb.toFixed(2)} MB)
            </span>
          )}
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search files..."
            className="pl-8 pr-8 py-1.5 text-sm border border-slate-300 rounded-lg w-48 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Blocked files warning */}
      {data.blocked_rejected.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="flex items-center gap-2 text-red-700 font-medium text-sm">
            <AlertTriangle className="h-4 w-4" />
            {data.blocked_rejected.length} blocked files were rejected
          </div>
          <div className="text-xs text-red-600 mt-1">
            {data.blocked_rejected.slice(0, 5).join(', ')}
            {data.blocked_rejected.length > 5 && ` and ${data.blocked_rejected.length - 5} more`}
          </div>
        </div>
      )}

      {/* Category sections */}
      <div className="space-y-3">
        {CATEGORY_ORDER.map((category) => {
          const files = filteredData?.categories[category] || []
          return (
            <CategorySection
              key={category}
              category={category}
              files={files}
              onDelete={(path) => deleteMutation.mutate(path)}
              onMarkAsObs={onMarkAsObservation}
              isDeleting={deleteMutation.isPending}
              deletingPath={deletingPath}
            />
          )
        })}
      </div>
    </div>
  )
}
