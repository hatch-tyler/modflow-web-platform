import { useCallback, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Loader2, FileText } from 'lucide-react'
import { fileEditorApi } from '../../services/api'
import { useFileEditorStore } from '../../store/fileEditorStore'
import EditorToolbar from './EditorToolbar'
import MonacoWrapper from './MonacoWrapper'
import ModflowTooltipProvider from './ModflowTooltipProvider'

interface FileEditorPaneProps {
  projectId: string
  filePath: string | null
  modelType: string
}

export default function FileEditorPane({
  projectId,
  filePath,
  modelType,
}: FileEditorPaneProps) {
  const queryClient = useQueryClient()
  const [saveError, setSaveError] = useState<string | null>(null)
  const {
    openFilePath,
    editedContent,
    isDirty,
    backups,
    openFile,
    updateContent,
    markSaved,
    setBackups,
  } = useFileEditorStore()

  // Load file content when path changes
  const { isLoading: contentLoading } = useQuery({
    queryKey: ['file-content', projectId, filePath],
    queryFn: async () => {
      if (!filePath) return null
      const result = await fileEditorApi.getContent(projectId, filePath)
      openFile(filePath, result.content)
      return result
    },
    enabled: !!filePath && filePath !== openFilePath,
  })

  // Load backups
  useQuery({
    queryKey: ['file-backups', projectId, filePath],
    queryFn: async () => {
      if (!filePath) return { backups: [] }
      const result = await fileEditorApi.getBackups(projectId, filePath)
      setBackups(result.backups)
      return result
    },
    enabled: !!filePath,
  })

  // Save mutation
  const saveMutation = useMutation({
    mutationFn: () => {
      if (!filePath || !editedContent) throw new Error('No content to save')
      return fileEditorApi.saveContent(projectId, filePath, editedContent)
    },
    onSuccess: () => {
      setSaveError(null)
      if (editedContent) {
        markSaved(editedContent)
      }
      // Invalidate cached file content and project data so other views see updates
      queryClient.invalidateQueries({ queryKey: ['file-content', projectId] })
      queryClient.invalidateQueries({ queryKey: ['project', projectId] })
      // Refresh backups
      if (filePath) {
        fileEditorApi.getBackups(projectId, filePath).then(r => setBackups(r.backups))
      }
    },
    onError: (error: Error) => {
      setSaveError(error.message || 'Failed to save file')
    },
  })

  // Revert mutation
  const revertMutation = useMutation({
    mutationFn: (timestamp: string) => {
      if (!filePath) throw new Error('No file to revert')
      return fileEditorApi.revert(projectId, filePath, timestamp)
    },
    onSuccess: (result) => {
      openFile(filePath!, result.content)
    },
  })

  const handleContentChange = useCallback((value: string) => {
    updateContent(value)
  }, [updateContent])

  // No file selected
  if (!filePath) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-white text-slate-400">
        <FileText className="h-12 w-12 mb-3" />
        <p className="text-lg font-medium">Select a file to edit</p>
        <p className="text-sm mt-1">Choose a file from the tree on the left</p>
      </div>
    )
  }

  // Loading
  if (contentLoading) {
    return (
      <div className="flex items-center justify-center h-full bg-white">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <span className="ml-3 text-slate-500">Loading file...</span>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-white">
      <EditorToolbar
        filePath={filePath}
        isDirty={isDirty}
        saving={saveMutation.isPending}
        backups={backups}
        onSave={() => saveMutation.mutate()}
        onRevert={(ts) => revertMutation.mutate(ts)}
      />

      <div className="flex-1 relative">
        {editedContent !== null && (
          <MonacoWrapper
            content={editedContent}
            onChange={handleContentChange}
            filePath={filePath}
          />
        )}
        <ModflowTooltipProvider
          modelType={modelType}
          filePath={filePath}
        />
      </div>

      {/* Save error banner */}
      {saveError && (
        <div className="flex items-center justify-between px-3 py-1 bg-red-50 border-t border-red-200 text-xs text-red-600">
          <span>Save failed: {saveError}</span>
          <button
            onClick={() => setSaveError(null)}
            className="ml-2 underline hover:no-underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Status bar */}
      <div className="flex items-center justify-between h-6 px-3 bg-slate-50 border-t border-slate-200 text-xs text-slate-400">
        <span>
          {saveMutation.isPending ? 'Saving...' : saveError ? 'Save failed' : saveMutation.isSuccess ? 'Saved' : isDirty ? 'Modified' : 'Ready'}
        </span>
        <span>
          {editedContent ? `${editedContent.split('\n').length} lines` : ''}
        </span>
      </div>
    </div>
  )
}
