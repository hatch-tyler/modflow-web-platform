import { useState, useEffect, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { FileCode } from 'lucide-react'
import { projectsApi } from '../services/api'
import { useFileEditorStore } from '../store/fileEditorStore'
import FileTreeSidebar from '../components/files/FileTreeSidebar'
import FileEditorPane from '../components/files/FileEditorPane'

const SIDEBAR_WIDTH = 280

export default function FilesPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const { isDirty, reset } = useFileEditorStore()

  const { data: project } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  })

  // Unsaved changes guard
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault()
        e.returnValue = ''
      }
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [isDirty])

  // Clean up on unmount
  useEffect(() => {
    return () => reset()
  }, [reset])

  const handleFileSelect = useCallback((path: string) => {
    if (isDirty) {
      const confirmed = window.confirm('You have unsaved changes. Discard and open a new file?')
      if (!confirmed) return
    }
    setSelectedFile(path)
  }, [isDirty])

  if (!project?.is_valid) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <FileCode className="h-12 w-12 text-slate-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-slate-700">No Model Files</h3>
          <p className="text-slate-500 mt-1">Upload and validate a model to view and edit files.</p>
        </div>
      </div>
    )
  }

  const modelType = project?.model_type || 'mf6'

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      <FileTreeSidebar
        projectId={projectId!}
        onFileSelect={handleFileSelect}
        selectedFile={selectedFile}
        width={SIDEBAR_WIDTH}
      />
      <div className="flex-1 min-w-0">
        <FileEditorPane
          projectId={projectId!}
          filePath={selectedFile}
          modelType={modelType}
        />
      </div>
    </div>
  )
}
