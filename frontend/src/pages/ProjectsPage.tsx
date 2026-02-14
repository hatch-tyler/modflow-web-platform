import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Plus, Trash2, FolderOpen, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import { projectsApi } from '../services/api'
import { useProjectStore } from '../store/projectStore'
import type { Project, ProjectCreate } from '../types'

export default function ProjectsPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { setCurrentProject } = useProjectStore()
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newProject, setNewProject] = useState<ProjectCreate>({ name: '', description: '' })

  const { data: projects, isLoading, error } = useQuery({
    queryKey: ['projects'],
    queryFn: () => projectsApi.list(),
  })

  const createMutation = useMutation({
    mutationFn: projectsApi.create,
    onSuccess: (project) => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setShowCreateModal(false)
      setNewProject({ name: '', description: '' })
      setCurrentProject(project)
      navigate(`/projects/${project.id}/upload`)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: projectsApi.delete,
    onMutate: async (projectId) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: ['projects'] })

      // Snapshot the previous value
      const previousProjects = queryClient.getQueryData<Project[]>(['projects'])

      // Optimistically remove the project from the list
      queryClient.setQueryData<Project[]>(['projects'], (old) =>
        old?.filter((p) => p.id !== projectId) ?? []
      )

      // Clear store references if the deleted project is currently selected
      const store = useProjectStore.getState()
      if (store.currentProject?.id === projectId) {
        store.setCurrentProject(null)
        store.setCurrentRun(null)
      }

      // Return context with the snapshot
      return { previousProjects }
    },
    onError: (_err, _projectId, context) => {
      // Rollback on error
      if (context?.previousProjects) {
        queryClient.setQueryData(['projects'], context.previousProjects)
      }
    },
    onSettled: () => {
      // Refetch after mutation settles
      queryClient.invalidateQueries({ queryKey: ['projects'] })
    },
  })

  const handleSelectProject = (project: Project) => {
    setCurrentProject(project)
    if (project.is_valid) {
      navigate(`/projects/${project.id}/viewer`)
    } else {
      navigate(`/projects/${project.id}/upload`)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
        Failed to load projects. Is the backend running?
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-slate-800">Projects</h2>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          <Plus className="h-5 w-5" />
          New Project
        </button>
      </div>

      {projects?.length === 0 ? (
        <div className="bg-white rounded-lg border border-slate-200 p-12 text-center">
          <FolderOpen className="h-12 w-12 text-slate-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-slate-800 mb-2">No projects yet</h3>
          <p className="text-slate-500 mb-4">Create a new project to get started</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Create Project
          </button>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {projects?.map((project) => (
            <div
              key={project.id}
              className="bg-white rounded-lg border border-slate-200 p-4 hover:shadow-md transition-shadow cursor-pointer"
              onClick={() => handleSelectProject(project)}
            >
              <div className="flex items-start justify-between mb-3">
                <h3 className="font-semibold text-slate-800 truncate flex-1">
                  {project.name}
                </h3>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    if (confirm('Delete this project?')) {
                      deleteMutation.mutate(project.id)
                    }
                  }}
                  className="p-1 text-slate-400 hover:text-red-500 transition-colors"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>

              {project.description && (
                <p className="text-sm text-slate-500 mb-3 line-clamp-2">
                  {project.description}
                </p>
              )}

              <div className="flex items-center gap-2 text-sm">
                {project.is_valid ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    <span className="text-green-600">Valid</span>
                    <span className="text-slate-400">
                      {project.model_type?.toUpperCase()} |{' '}
                      {project.grid_type === 'vertex'
                        ? `${project.nlay}L × ${project.ncol} cells (DISV)`
                        : project.grid_type === 'unstructured'
                        ? `${project.ncol} nodes (DISU)`
                        : `${project.nlay}L × ${project.nrow}R × ${project.ncol}C`}
                    </span>
                  </>
                ) : (
                  <>
                    <XCircle className="h-4 w-4 text-slate-400" />
                    <span className="text-slate-500">No model uploaded</span>
                  </>
                )}
              </div>

              <div className="mt-3 text-xs text-slate-400">
                Updated {new Date(project.updated_at).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Project Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Create New Project</h3>
            <form
              onSubmit={(e) => {
                e.preventDefault()
                createMutation.mutate(newProject)
              }}
            >
              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Project Name
                </label>
                <input
                  type="text"
                  value={newProject.name}
                  onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="My MODFLOW Model"
                  required
                />
              </div>
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Description (optional)
                </label>
                <textarea
                  value={newProject.description || ''}
                  onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  placeholder="Brief description of the model..."
                />
              </div>
              <div className="flex justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="px-4 py-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={createMutation.isPending || !newProject.name}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors flex items-center gap-2"
                >
                  {createMutation.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
                  Create
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
