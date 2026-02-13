import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  ChevronRight, ChevronDown, FileText, Search,
  Loader2, FolderOpen, Database, Settings, FileCode,
} from 'lucide-react'
import { filesApi } from '../../services/api'
import type { FileCategory, FileInfo } from '../../types'

interface FileTreeSidebarProps {
  projectId: string
  onFileSelect: (filePath: string) => void
  selectedFile: string | null
  width: number
}

const CATEGORY_LABELS: Record<FileCategory, string> = {
  model_core: 'Core Files',
  model_input: 'Input Packages',
  model_output: 'Output Files',
  pest: 'PEST Files',
  observation: 'Observations',
  blocked: 'Blocked',
  other: 'Other',
}

const CATEGORY_ICONS: Record<FileCategory, React.ElementType> = {
  model_core: Database,
  model_input: FileCode,
  model_output: FolderOpen,
  pest: Settings,
  observation: FileText,
  blocked: FileText,
  other: FileText,
}

function getFileIcon(ext: string): string {
  const extMap: Record<string, string> = {
    '.nam': 'N', '.dis': 'D', '.tdis': 'T', '.ims': 'S',
    '.npf': 'K', '.sto': 'S', '.ic': 'I', '.oc': 'O',
    '.wel': 'W', '.rch': 'R', '.evt': 'E', '.chd': 'C',
    '.ghb': 'G', '.drn': 'D', '.riv': 'R', '.sfr': 'F',
    '.maw': 'M', '.lak': 'L', '.uzf': 'U', '.hfb': 'H',
  }
  return extMap[ext.toLowerCase()] || ext.replace('.', '').toUpperCase().slice(0, 2)
}

export default function FileTreeSidebar({
  projectId,
  onFileSelect,
  selectedFile,
  width,
}: FileTreeSidebarProps) {
  const [searchFilter, setSearchFilter] = useState('')
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['model_core', 'model_input'])
  )

  const { data: categorizedFiles, isLoading } = useQuery({
    queryKey: ['files-categorized', projectId],
    queryFn: () => filesApi.getCategorized(projectId),
    enabled: !!projectId,
  })

  const filteredCategories = useMemo(() => {
    if (!categorizedFiles) return []

    const categories = Object.entries(categorizedFiles.categories) as [FileCategory, FileInfo[]][]

    return categories
      .map(([category, files]) => {
        const filtered = searchFilter
          ? files.filter(f =>
              f.name.toLowerCase().includes(searchFilter.toLowerCase()) ||
              f.path.toLowerCase().includes(searchFilter.toLowerCase())
            )
          : files

        return { category, files: filtered }
      })
      .filter(({ files }) => files.length > 0)
  }, [categorizedFiles, searchFilter])

  const toggleCategory = (cat: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev)
      if (next.has(cat)) next.delete(cat)
      else next.add(cat)
      return next
    })
  }

  return (
    <div className="flex flex-col h-full bg-slate-50 border-r border-slate-200" style={{ width }}>
      {/* Search */}
      <div className="p-2 border-b border-slate-200">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400" />
          <input
            type="text"
            value={searchFilter}
            onChange={e => setSearchFilter(e.target.value)}
            placeholder="Filter files..."
            className="w-full pl-8 pr-3 py-1.5 text-sm bg-white border border-slate-200 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-400"
          />
        </div>
      </div>

      {/* Tree */}
      <div className="flex-1 overflow-auto py-1">
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
          </div>
        ) : (
          filteredCategories.map(({ category, files }) => {
            const Icon = CATEGORY_ICONS[category]
            const isExpanded = expandedCategories.has(category)

            return (
              <div key={category}>
                <button
                  onClick={() => toggleCategory(category)}
                  className="flex items-center w-full px-2 py-1.5 text-sm hover:bg-slate-100 transition-colors"
                >
                  {isExpanded ? (
                    <ChevronDown className="h-3.5 w-3.5 text-slate-400 mr-1" />
                  ) : (
                    <ChevronRight className="h-3.5 w-3.5 text-slate-400 mr-1" />
                  )}
                  <Icon className="h-3.5 w-3.5 text-slate-500 mr-1.5" />
                  <span className="font-medium text-slate-600">{CATEGORY_LABELS[category]}</span>
                  <span className="ml-auto text-xs text-slate-400">{files.length}</span>
                </button>

                {isExpanded && (
                  <div className="ml-4">
                    {files.map(file => {
                      const isSelected = selectedFile === file.path
                      return (
                        <button
                          key={file.path}
                          onClick={() => onFileSelect(file.path)}
                          className={`flex items-center w-full px-2 py-1 text-sm rounded-sm transition-colors ${
                            isSelected
                              ? 'bg-blue-100 text-blue-700'
                              : 'text-slate-600 hover:bg-slate-100'
                          }`}
                          title={file.description || file.path}
                        >
                          <span className={`flex-shrink-0 w-5 h-5 rounded text-[10px] font-bold flex items-center justify-center mr-1.5 ${
                            isSelected ? 'bg-blue-200' : 'bg-slate-200 text-slate-500'
                          }`}>
                            {getFileIcon(file.extension)}
                          </span>
                          <span className="truncate">{file.name}</span>
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>

      {/* Stats */}
      {categorizedFiles && (
        <div className="px-3 py-2 border-t border-slate-200 text-xs text-slate-400">
          {categorizedFiles.total_files} files ({categorizedFiles.total_size_mb.toFixed(1)} MB)
        </div>
      )}
    </div>
  )
}
