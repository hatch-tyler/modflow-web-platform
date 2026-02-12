import { useCallback, useEffect, useState, useRef, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Upload, FileArchive, CheckCircle, XCircle, AlertTriangle, Loader2, Package, Database, Cpu, HardDrive, FolderOpen, FileSpreadsheet, ChevronDown, ChevronUp } from 'lucide-react'
import { projectsApi, uploadApi, observationsApi, filesApi } from '../services/api'
import { useProjectStore } from '../store/projectStore'
import FileBrowser from '../components/upload/FileBrowser'
import { ObservationSetList } from '../components/upload/ObservationSetCard'
import ColumnMappingModal from '../components/upload/ColumnMappingModal'
import SpatialReferenceEditor from '../components/upload/SpatialReferenceEditor'
import type { UploadStatus, UploadStage, ObservationSet } from '../types'
import clsx from 'clsx'

// Local storage key for persisting upload job across navigation
const getUploadJobKey = (projectId: string) => `upload_job_${projectId}`

// Stage configuration for UI
const STAGES: { key: UploadStage; label: string; icon: typeof Upload }[] = [
  { key: 'receiving', label: 'Uploading', icon: Upload },
  { key: 'extracting', label: 'Extracting', icon: Package },
  { key: 'validating', label: 'Validating', icon: Cpu },
  { key: 'storing', label: 'Storing', icon: Database },
  { key: 'caching', label: 'Caching', icon: HardDrive },
]

function getStageIndex(stage: UploadStage): number {
  const idx = STAGES.findIndex(s => s.key === stage)
  return idx >= 0 ? idx : 0
}

function StageProgress({ status }: { status: UploadStatus }) {
  const currentIndex = getStageIndex(status.stage)
  const isComplete = status.stage === 'complete'
  const isFailed = status.stage === 'failed'

  return (
    <div className="space-y-4">
      {/* Stage indicators */}
      <div className="flex items-center justify-between">
        {STAGES.map((stage, index) => {
          const Icon = stage.icon
          const isActive = index === currentIndex && !isComplete && !isFailed
          const isDone = index < currentIndex || isComplete
          const isPending = index > currentIndex && !isFailed

          return (
            <div key={stage.key} className="flex flex-col items-center flex-1">
              <div
                className={clsx(
                  'w-10 h-10 rounded-full flex items-center justify-center transition-colors',
                  isDone && 'bg-green-500 text-white',
                  isActive && 'bg-blue-500 text-white',
                  isPending && 'bg-slate-200 text-slate-400',
                  isFailed && index === currentIndex && 'bg-red-500 text-white'
                )}
              >
                {isDone ? (
                  <CheckCircle className="h-5 w-5" />
                ) : isActive ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Icon className="h-5 w-5" />
                )}
              </div>
              <span
                className={clsx(
                  'text-xs mt-1 font-medium',
                  isDone && 'text-green-600',
                  isActive && 'text-blue-600',
                  isPending && 'text-slate-400',
                  isFailed && index === currentIndex && 'text-red-600'
                )}
              >
                {stage.label}
              </span>
              {/* Connector line */}
              {index < STAGES.length - 1 && (
                <div
                  className={clsx(
                    'absolute h-0.5 w-full -z-10',
                    isDone ? 'bg-green-500' : 'bg-slate-200'
                  )}
                  style={{
                    left: '50%',
                    top: '20px',
                    width: 'calc(100% - 40px)',
                    transform: 'translateX(20px)',
                  }}
                />
              )}
            </div>
          )
        })}
      </div>

      {/* Progress bar for current stage */}
      {!isComplete && !isFailed && (
        <div className="mt-4">
          <div className="flex justify-between text-sm text-slate-600 mb-1">
            <span>{status.message}</span>
            <span>{status.progress}%</span>
          </div>
          <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-300"
              style={{ width: `${status.progress}%` }}
            />
          </div>
          {status.files_processed !== undefined && status.file_count !== undefined && (
            <div className="text-xs text-slate-500 mt-1">
              {status.files_processed} of {status.file_count} files
            </div>
          )}
        </div>
      )}

      {/* Complete message */}
      {isComplete && (
        <div className="mt-4 text-center">
          <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-2" />
          <div className="text-lg font-medium text-green-700">Upload Complete!</div>
          <div className="text-sm text-slate-600">{status.message}</div>
        </div>
      )}

      {/* Error message */}
      {isFailed && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-700 font-medium">
            <XCircle className="h-5 w-5" />
            Upload Failed
          </div>
          <div className="text-sm text-red-600 mt-1">{status.error || status.message}</div>
        </div>
      )}
    </div>
  )
}

export default function UploadPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { uploadProgress, setUploadProgress, isUploading, setIsUploading, setCurrentProject } = useProjectStore()
  const [activeJobId, setActiveJobId] = useState<string | null>(null)

  // Panel states
  const [showFileBrowser, setShowFileBrowser] = useState(true)
  const [showObservations, setShowObservations] = useState(true)

  // Modal states
  const [mappingModalFile, setMappingModalFile] = useState<string | null>(null)
  const obsFileInputRef = useRef<HTMLInputElement>(null)

  // Check for existing upload job on mount
  useEffect(() => {
    if (projectId) {
      const storedJobId = localStorage.getItem(getUploadJobKey(projectId))
      if (storedJobId) {
        setActiveJobId(storedJobId)
        setIsUploading(true)
      }
    }
  }, [projectId, setIsUploading])

  // Poll for upload status when there's an active job
  const { data: uploadStatus } = useQuery({
    queryKey: ['uploadStatus', activeJobId],
    queryFn: () => uploadApi.getStatus(activeJobId!),
    enabled: !!activeJobId,
    refetchInterval: (query) => {
      const data = query.state.data
      // Stop polling when complete or failed
      if (data?.stage === 'complete' || data?.stage === 'failed') {
        return false
      }
      return 1000 // Poll every second
    },
  })

  // Handle upload completion
  useEffect(() => {
    if (uploadStatus?.stage === 'complete' || uploadStatus?.stage === 'failed') {
      // Clear stored job
      if (projectId) {
        localStorage.removeItem(getUploadJobKey(projectId))
      }
      setActiveJobId(null)
      setIsUploading(false)
      setUploadProgress(0)

      // Refresh project data
      queryClient.invalidateQueries({ queryKey: ['project', projectId] })

      if (uploadStatus.stage === 'complete' && uploadStatus.is_valid) {
        // Update current project and navigate to viewer
        projectsApi.get(projectId!).then((updatedProject) => {
          setCurrentProject(updatedProject)
        })
      }
    }
  }, [uploadStatus, projectId, queryClient, setIsUploading, setUploadProgress, setCurrentProject])

  const { data: project, isLoading } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId,
  })

  // Query for observation sets
  const { data: observationSets = [] } = useQuery({
    queryKey: ['observation-sets', projectId],
    queryFn: () => observationsApi.listSets(projectId!),
    enabled: !!projectId && !!project?.is_valid,
    retry: false,
  })

  // Query for categorized files (includes detected observations)
  const { data: categorizedFiles } = useQuery({
    queryKey: ['categorized-files', projectId],
    queryFn: () => filesApi.getCategorized(projectId!),
    enabled: !!projectId && !!project?.is_valid,
    retry: false,
  })

  // Combine observation sets with detected observations from upload
  // Detected observations that haven't been configured yet need to be shown
  const allObservationSets = useMemo(() => {
    const sets: ObservationSet[] = [...observationSets]

    // Add detected observations that aren't already in the sets list
    if (categorizedFiles?.detected_observations) {
      for (const detected of categorizedFiles.detected_observations) {
        // Check if this detected observation is already in the sets (by ID or name)
        const alreadyExists = sets.some(
          s => s.id === detected.id || s.name === detected.name
        )
        if (!alreadyExists) {
          // Convert to ObservationSet format, preserving file_path
          sets.push({
            id: detected.id,
            name: detected.name,
            source: 'zip_detected',
            format: detected.format,
            wells: detected.wells || [],
            n_observations: detected.n_observations,
            created_at: detected.created_at,
            column_mapping: null,
            file_path: detected.file_path,
          })
        }
      }
    }

    return sets
  }, [observationSets, categorizedFiles])

  // Mutation for uploading new observation CSV
  const uploadObsMutation = useMutation({
    mutationFn: (file: File) => observationsApi.createSet(projectId!, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['observation-sets', projectId] })
      queryClient.invalidateQueries({ queryKey: ['observations', projectId] })
    },
  })

  const handleObsUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      uploadObsMutation.mutate(file)
    }
    if (obsFileInputRef.current) {
      obsFileInputRef.current.value = ''
    }
  }

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      setIsUploading(true)

      // Create a custom upload that tracks both network and server-side progress
      const formData = new FormData()
      formData.append('file', file)

      // Use fetch with XMLHttpRequest for upload progress
      return new Promise<{ job_id?: string }>((resolve, reject) => {
        const xhr = new XMLHttpRequest()

        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            const progress = Math.round((e.loaded * 100) / e.total)
            setUploadProgress(progress)
          }
        })

        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const response = JSON.parse(xhr.responseText)
              resolve(response)
            } catch {
              resolve({})
            }
          } else {
            reject(new Error(xhr.statusText || 'Upload failed'))
          }
        })

        xhr.addEventListener('error', () => reject(new Error('Network error')))
        xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')))

        xhr.open('POST', `/api/v1/projects/${projectId}/upload`)
        xhr.send(formData)
      })
    },
    onSuccess: (data) => {
      // If the server returned a job_id, store it and start polling
      // Otherwise the upload completed synchronously
      if (data.job_id) {
        setActiveJobId(data.job_id)
        localStorage.setItem(getUploadJobKey(projectId!), data.job_id)
      } else {
        // Synchronous completion
        queryClient.invalidateQueries({ queryKey: ['project', projectId] })
        setIsUploading(false)
        setUploadProgress(0)

        projectsApi.get(projectId!).then((updatedProject) => {
          setCurrentProject(updatedProject)
          if (updatedProject.is_valid) {
            navigate(`/projects/${projectId}/viewer`)
          }
        })
      }
    },
    onError: () => {
      setIsUploading(false)
      setUploadProgress(0)
      if (projectId) {
        localStorage.removeItem(getUploadJobKey(projectId))
      }
    },
  })

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file && file.name.endsWith('.zip')) {
        uploadMutation.mutate(file)
      }
    },
    [uploadMutation]
  )

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        uploadMutation.mutate(file)
      }
    },
    [uploadMutation]
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    )
  }

  const showUploadProgress = isUploading || activeJobId

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-2xl font-bold text-slate-800 mb-2">Upload Model</h2>
      <p className="text-slate-500 mb-6">
        Upload a ZIP file containing your MODFLOW model input files (NAM, DIS, BAS6, etc.)
      </p>

      {/* Upload progress with stages */}
      {showUploadProgress && (
        <div className="mb-8 bg-white border border-slate-200 rounded-xl p-6">
          {uploadStatus ? (
            <StageProgress status={uploadStatus} />
          ) : (
            <div className="text-center">
              <Loader2 className="h-12 w-12 text-blue-500 mx-auto mb-4 animate-spin" />
              <div className="text-lg font-medium text-slate-700 mb-2">
                Uploading file...
              </div>
              <div className="w-64 h-2 bg-slate-200 rounded-full mx-auto overflow-hidden">
                <div
                  className="h-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <div className="text-sm text-slate-500 mt-2">{uploadProgress}%</div>
            </div>
          )}
        </div>
      )}

      {/* Drop zone - hidden during upload */}
      {!showUploadProgress && (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="border-2 border-dashed rounded-xl p-12 text-center transition-colors border-slate-300 hover:border-blue-400 hover:bg-slate-50"
        >
          <FileArchive className="h-12 w-12 text-slate-400 mx-auto mb-4" />
          <div className="text-lg font-medium text-slate-700 mb-2">
            Drag and drop your model ZIP file here
          </div>
          <div className="text-slate-500 mb-4">or</div>
          <label className="inline-block px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors">
            <Upload className="h-5 w-5 inline mr-2" />
            Browse Files
            <input
              type="file"
              accept=".zip"
              onChange={handleFileSelect}
              className="hidden"
            />
          </label>
          <div className="text-sm text-slate-400 mt-4">
            Supports MODFLOW 2005, MODFLOW-NWT, MODFLOW-USG, and MODFLOW 6
          </div>
        </div>
      )}

      {/* Validation results - show when not uploading and project has validation status */}
      {!showUploadProgress && project?.is_valid !== undefined && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Validation Results</h3>

          {project.is_valid ? (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center gap-2 text-green-700 font-medium mb-2">
                <CheckCircle className="h-5 w-5" />
                Model is valid and ready for visualization
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm text-green-600">
                <div>
                  <span className="font-medium">Type:</span> {project.model_type?.toUpperCase()}
                </div>
                <div>
                  <span className="font-medium">Grid:</span> {project.grid_type === 'vertex'
                    ? `${project.nlay} layers × ${project.ncol} cells/layer (DISV)`
                    : project.grid_type === 'unstructured'
                    ? `${project.ncol} nodes (DISU)`
                    : `${project.nlay} layers × ${project.nrow} rows × ${project.ncol} cols`}
                </div>
                <div>
                  <span className="font-medium">Stress Periods:</span> {project.nper}
                </div>
                <div>
                  <span className="font-medium">Packages:</span>{' '}
                  {project.packages ? Object.keys(project.packages).length : 0}
                </div>
              </div>
              <button
                onClick={() => navigate(`/projects/${projectId}/viewer`)}
                className="mt-4 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                Open 3D Viewer
              </button>
            </div>
          ) : project.validation_errors ? (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center gap-2 text-red-700 font-medium mb-2">
                <XCircle className="h-5 w-5" />
                Validation failed
              </div>
              <ul className="text-sm text-red-600 list-disc list-inside">
                {Object.entries(project.validation_errors).map(([key, value]) => (
                  <li key={key}>{String(value)}</li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="bg-slate-100 border border-slate-200 rounded-lg p-4">
              <div className="flex items-center gap-2 text-slate-600">
                <AlertTriangle className="h-5 w-5" />
                No model has been uploaded yet
              </div>
            </div>
          )}
        </div>
      )}

      {/* Spatial Reference Editor - show when valid model is uploaded */}
      {!showUploadProgress && project?.is_valid && (
        <SpatialReferenceEditor project={project} />
      )}

      {/* Upload error */}
      {uploadMutation.isError && !showUploadProgress && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-700">
            <XCircle className="h-5 w-5" />
            Upload failed: {(uploadMutation.error as Error)?.message || 'Unknown error'}
          </div>
        </div>
      )}

      {/* Info box about navigation */}
      {showUploadProgress && (
        <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="text-sm text-blue-700">
            <strong>Tip:</strong> You can navigate away from this page. Your upload will continue in the background and you can return to check its progress.
          </div>
        </div>
      )}

      {/* File Browser Panel - show when project has files */}
      {!showUploadProgress && project?.is_valid && (
        <div className="mt-8">
          <button
            onClick={() => setShowFileBrowser(!showFileBrowser)}
            className="flex items-center gap-2 text-lg font-semibold text-slate-800 mb-4 hover:text-slate-600 transition-colors"
          >
            <FolderOpen className="h-5 w-5" />
            Model Files
            {showFileBrowser ? (
              <ChevronUp className="h-4 w-4 text-slate-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-slate-400" />
            )}
          </button>

          {showFileBrowser && (
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <FileBrowser
                projectId={projectId!}
                onMarkAsObservation={(filePath) => setMappingModalFile(filePath)}
              />
            </div>
          )}
        </div>
      )}

      {/* Observation Sets Panel - show when project has files */}
      {!showUploadProgress && project?.is_valid && (
        <div className="mt-8">
          <button
            onClick={() => setShowObservations(!showObservations)}
            className="flex items-center gap-2 text-lg font-semibold text-slate-800 mb-4 hover:text-slate-600 transition-colors"
          >
            <FileSpreadsheet className="h-5 w-5 text-orange-500" />
            Observation Sets
            {observationSets.length > 0 && (
              <span className="text-sm font-normal text-slate-400">
                ({observationSets.length})
              </span>
            )}
            {showObservations ? (
              <ChevronUp className="h-4 w-4 text-slate-400" />
            ) : (
              <ChevronDown className="h-4 w-4 text-slate-400" />
            )}
          </button>

          {showObservations && (
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              {/* Hidden file input for observation upload */}
              <input
                ref={obsFileInputRef}
                type="file"
                accept=".csv"
                onChange={handleObsUpload}
                className="hidden"
              />

              {uploadObsMutation.isPending && (
                <div className="flex items-center gap-2 text-sm text-blue-600 mb-4">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Uploading observation data...
                </div>
              )}

              {uploadObsMutation.isError && (
                <div className="flex items-center gap-2 text-sm text-red-600 mb-4">
                  <XCircle className="h-4 w-4" />
                  {(uploadObsMutation.error as Error)?.message || 'Upload failed'}
                </div>
              )}

              <ObservationSetList
                sets={allObservationSets}
                projectId={projectId!}
                onUpload={() => obsFileInputRef.current?.click()}
                onConfigure={(set) => {
                  // Use file_path directly if available, otherwise try to find it
                  if (set.file_path) {
                    setMappingModalFile(set.file_path)
                  } else {
                    // Fallback: try to find the file in categorized observation files
                    const obsFile = categorizedFiles?.categories?.observation?.find(
                      f => f.name.replace(/\.[^/.]+$/, '') === set.name || f.path.includes(set.name)
                    )
                    if (obsFile) {
                      setMappingModalFile(obsFile.path)
                    }
                  }
                }}
              />
            </div>
          )}
        </div>
      )}

      {/* Column Mapping Modal */}
      {mappingModalFile && (
        <ColumnMappingModal
          projectId={projectId!}
          filePath={mappingModalFile}
          onClose={() => setMappingModalFile(null)}
          onSuccess={() => {
            setMappingModalFile(null)
            queryClient.invalidateQueries({ queryKey: ['observation-sets', projectId] })
          }}
        />
      )}
    </div>
  )
}
