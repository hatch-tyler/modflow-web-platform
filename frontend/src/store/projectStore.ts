import { create } from 'zustand'
import type { Project, Run } from '../types'

interface ProjectState {
  // Current project context
  currentProject: Project | null
  setCurrentProject: (project: Project | null) => void

  // Current run context
  currentRun: Run | null
  setCurrentRun: (run: Run | null) => void

  // UI state
  sidebarOpen: boolean
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void

  // Upload state
  uploadProgress: number
  setUploadProgress: (progress: number) => void
  isUploading: boolean
  setIsUploading: (uploading: boolean) => void
}

export const useProjectStore = create<ProjectState>((set) => ({
  // Project
  currentProject: null,
  setCurrentProject: (project) => set({ currentProject: project }),

  // Run
  currentRun: null,
  setCurrentRun: (run) => set({ currentRun: run }),

  // Sidebar
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),

  // Upload
  uploadProgress: 0,
  setUploadProgress: (progress) => set({ uploadProgress: progress }),
  isUploading: false,
  setIsUploading: (uploading) => set({ isUploading: uploading }),
}))
