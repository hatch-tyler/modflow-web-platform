import { create } from 'zustand'
import type { BackupInfo } from '../types'

interface FileEditorState {
  openFilePath: string | null
  originalContent: string | null
  editedContent: string | null
  isDirty: boolean
  backups: BackupInfo[]

  openFile: (path: string, content: string) => void
  updateContent: (content: string) => void
  markSaved: (content: string) => void
  setBackups: (backups: BackupInfo[]) => void
  reset: () => void
}

export const useFileEditorStore = create<FileEditorState>((set) => ({
  openFilePath: null,
  originalContent: null,
  editedContent: null,
  isDirty: false,
  backups: [],

  openFile: (path, content) =>
    set({
      openFilePath: path,
      originalContent: content,
      editedContent: content,
      isDirty: false,
      backups: [],
    }),

  updateContent: (content) =>
    set((state) => ({
      editedContent: content,
      isDirty: content !== state.originalContent,
    })),

  markSaved: (content) =>
    set({
      originalContent: content,
      editedContent: content,
      isDirty: false,
    }),

  setBackups: (backups) => set({ backups }),

  reset: () =>
    set({
      openFilePath: null,
      originalContent: null,
      editedContent: null,
      isDirty: false,
      backups: [],
    }),
}))
