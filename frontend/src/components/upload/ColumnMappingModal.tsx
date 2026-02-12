import { useState, useEffect, useMemo } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { X, AlertCircle, Loader2, CheckCircle } from 'lucide-react'
import { observationsApi, filesApi } from '../../services/api'
import type { ColumnMapping } from '../../types'

interface ColumnMappingModalProps {
  projectId: string
  filePath: string
  onClose: () => void
  onSuccess?: () => void
}

const FIELD_LABELS: Record<string, { label: string; description: string }> = {
  well_name: { label: 'Well Name', description: 'Column containing well identifier' },
  layer: { label: 'Layer', description: 'Layer number (1-based) or fixed value' },
  row: { label: 'Row', description: 'Row number (1-based) or fixed value' },
  col: { label: 'Column', description: 'Column number (1-based) or fixed value' },
  node: { label: 'Node', description: 'Node number for unstructured grids' },
  time: { label: 'Time', description: 'Simulation time column' },
  value: { label: 'Value', description: 'Observation value (e.g., head)' },
}

export default function ColumnMappingModal({
  projectId,
  filePath,
  onClose,
  onSuccess,
}: ColumnMappingModalProps) {
  const queryClient = useQueryClient()
  const [setName, setSetName] = useState('')
  const [csvPreview, setCsvPreview] = useState<string[][]>([])
  const [headers, setHeaders] = useState<string[]>([])
  const [mapping, setMapping] = useState<Record<string, string | number | null>>({
    well_name: null,
    layer: null,
    row: null,
    col: null,
    node: null,
    time: null,
    value: null,
  })
  const [useFixedValue, setUseFixedValue] = useState<Record<string, boolean>>({})
  const [fixedValues, setFixedValues] = useState<Record<string, number>>({})
  const [isLoadingPreview, setIsLoadingPreview] = useState(true)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [useNode, setUseNode] = useState(false)

  // Load CSV preview
  useEffect(() => {
    async function loadPreview() {
      setIsLoadingPreview(true)
      setPreviewError(null)

      try {
        // Fetch actual CSV preview from backend
        const preview = await filesApi.previewCsv(projectId, filePath, 5)

        if (!preview.headers || preview.headers.length === 0) {
          throw new Error('CSV file appears to be empty')
        }

        // Set the name from filename
        const fileName = filePath.split('/').pop() || ''
        setSetName(fileName.replace(/\.csv$/i, '').replace(/[_-]/g, ' '))

        // Set headers and preview data
        setHeaders(preview.headers)
        setCsvPreview([preview.headers, ...preview.rows])

        // Auto-detect column mappings based on actual headers
        const lowerHeaders = preview.headers.map((h) => h.toLowerCase())
        const newMapping: Record<string, string | number | null> = { ...mapping }

        // Try to auto-map columns
        const autoMappings: Record<string, string[]> = {
          well_name: ['wellname', 'well_name', 'well', 'site', 'id', 'name', 'well_id', 'wellid', 'obs_name', 'obsname'],
          layer: ['layer', 'lay', 'k', 'lyr'],
          row: ['row', 'r', 'i', 'row_num', 'rownum'],
          col: ['col', 'column', 'c', 'j', 'col_num', 'colnum'],
          node: ['node', 'nodeid', 'n', 'node_num', 'nodenum'],
          time: ['time', 't', 'simtime', 'stress_period', 'sp', 'totim', 'datetime', 'date'],
          value: ['head', 'value', 'obs', 'observed', 'waterlevel', 'level', 'simulated', 'measured', 'obs_value'],
        }

        for (const [field, aliases] of Object.entries(autoMappings)) {
          for (const alias of aliases) {
            const idx = lowerHeaders.indexOf(alias)
            if (idx >= 0) {
              newMapping[field] = preview.headers[idx]
              break
            }
          }
        }

        setMapping(newMapping)
      } catch (err) {
        setPreviewError((err as Error).message || 'Failed to load file preview')
      } finally {
        setIsLoadingPreview(false)
      }
    }

    loadPreview()
  }, [projectId, filePath])

  const markMutation = useMutation({
    mutationFn: () => {
      // Build column mapping
      const columnMapping: ColumnMapping = {
        well_name: mapping.well_name as string,
        layer: useFixedValue.layer ? fixedValues.layer || 1 : (mapping.layer as string),
        row: useNode ? null : (useFixedValue.row ? fixedValues.row : mapping.row as string | null),
        col: useNode ? null : (useFixedValue.col ? fixedValues.col : mapping.col as string | null),
        node: useNode ? (useFixedValue.node ? fixedValues.node : mapping.node as string | null) : null,
        time: mapping.time as string,
        value: mapping.value as string,
      }

      return observationsApi.markFileAsObs(projectId, {
        file_path: filePath,
        name: setName || filePath.split('/').pop() || 'Observations',
        column_mapping: columnMapping,
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['observation-sets', projectId] })
      queryClient.invalidateQueries({ queryKey: ['observations', projectId] })
      queryClient.invalidateQueries({ queryKey: ['categorized-files', projectId] })
      onSuccess?.()
      onClose()
    },
  })

  const isValid = useMemo(() => {
    // Check required fields
    if (!mapping.well_name || !mapping.time || !mapping.value) return false

    // Check location fields
    if (useNode) {
      if (!useFixedValue.node && !mapping.node) return false
    } else {
      if (!useFixedValue.row && !mapping.row) return false
      if (!useFixedValue.col && !mapping.col) return false
    }

    // Layer can be fixed or column
    if (!useFixedValue.layer && !mapping.layer) return false

    return true
  }, [mapping, useFixedValue, useNode])

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div>
            <h2 className="text-lg font-semibold text-slate-800">
              Map CSV Columns
            </h2>
            <p className="text-sm text-slate-500">{filePath}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-slate-400" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {isLoadingPreview ? (
            <div className="flex items-center justify-center h-48">
              <Loader2 className="h-6 w-6 animate-spin text-blue-500" />
              <span className="ml-2 text-slate-500">Loading preview...</span>
            </div>
          ) : previewError ? (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
              <AlertCircle className="h-5 w-5 inline mr-2" />
              {previewError}
            </div>
          ) : (
            <>
              {/* Set name */}
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Observation Set Name
                </label>
                <input
                  type="text"
                  value={setName}
                  onChange={(e) => setSetName(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Field Campaign 2024"
                />
              </div>

              {/* CSV Preview */}
              <div>
                <h3 className="text-sm font-medium text-slate-700 mb-2">
                  CSV Preview
                </h3>
                <div className="border border-slate-200 rounded-lg overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50">
                      <tr>
                        {headers.map((h, i) => (
                          <th
                            key={i}
                            className="px-3 py-2 text-left text-slate-600 font-medium border-b"
                          >
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {csvPreview.slice(1, 5).map((row, i) => (
                        <tr key={i} className="border-b last:border-0">
                          {row.map((cell, j) => (
                            <td key={j} className="px-3 py-1.5 text-slate-600">
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Grid type toggle */}
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium text-slate-700">
                  Grid Type:
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    checked={!useNode}
                    onChange={() => setUseNode(false)}
                    className="text-blue-600"
                  />
                  <span className="text-sm">Structured (Row/Col)</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    checked={useNode}
                    onChange={() => setUseNode(true)}
                    className="text-blue-600"
                  />
                  <span className="text-sm">Unstructured (Node)</span>
                </label>
              </div>

              {/* Column mappings */}
              <div>
                <h3 className="text-sm font-medium text-slate-700 mb-3">
                  Column Mappings
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  {/* Well Name */}
                  <ColumnMappingField
                    label={FIELD_LABELS.well_name.label}
                    description={FIELD_LABELS.well_name.description}
                    headers={headers}
                    value={mapping.well_name as string}
                    onChange={(v) => setMapping({ ...mapping, well_name: v })}
                    required
                  />

                  {/* Time */}
                  <ColumnMappingField
                    label={FIELD_LABELS.time.label}
                    description={FIELD_LABELS.time.description}
                    headers={headers}
                    value={mapping.time as string}
                    onChange={(v) => setMapping({ ...mapping, time: v })}
                    required
                  />

                  {/* Value */}
                  <ColumnMappingField
                    label={FIELD_LABELS.value.label}
                    description={FIELD_LABELS.value.description}
                    headers={headers}
                    value={mapping.value as string}
                    onChange={(v) => setMapping({ ...mapping, value: v })}
                    required
                  />

                  {/* Layer */}
                  <ColumnMappingFieldWithFixed
                    label={FIELD_LABELS.layer.label}
                    description={FIELD_LABELS.layer.description}
                    headers={headers}
                    value={mapping.layer as string}
                    onChange={(v) => setMapping({ ...mapping, layer: v })}
                    useFixed={useFixedValue.layer || false}
                    onUseFixedChange={(v) => setUseFixedValue({ ...useFixedValue, layer: v })}
                    fixedValue={fixedValues.layer || 1}
                    onFixedValueChange={(v) => setFixedValues({ ...fixedValues, layer: v })}
                  />

                  {/* Row/Col or Node */}
                  {useNode ? (
                    <ColumnMappingFieldWithFixed
                      label={FIELD_LABELS.node.label}
                      description={FIELD_LABELS.node.description}
                      headers={headers}
                      value={mapping.node as string}
                      onChange={(v) => setMapping({ ...mapping, node: v })}
                      useFixed={useFixedValue.node || false}
                      onUseFixedChange={(v) => setUseFixedValue({ ...useFixedValue, node: v })}
                      fixedValue={fixedValues.node || 1}
                      onFixedValueChange={(v) => setFixedValues({ ...fixedValues, node: v })}
                    />
                  ) : (
                    <>
                      <ColumnMappingFieldWithFixed
                        label={FIELD_LABELS.row.label}
                        description={FIELD_LABELS.row.description}
                        headers={headers}
                        value={mapping.row as string}
                        onChange={(v) => setMapping({ ...mapping, row: v })}
                        useFixed={useFixedValue.row || false}
                        onUseFixedChange={(v) => setUseFixedValue({ ...useFixedValue, row: v })}
                        fixedValue={fixedValues.row || 1}
                        onFixedValueChange={(v) => setFixedValues({ ...fixedValues, row: v })}
                      />
                      <ColumnMappingFieldWithFixed
                        label={FIELD_LABELS.col.label}
                        description={FIELD_LABELS.col.description}
                        headers={headers}
                        value={mapping.col as string}
                        onChange={(v) => setMapping({ ...mapping, col: v })}
                        useFixed={useFixedValue.col || false}
                        onUseFixedChange={(v) => setUseFixedValue({ ...useFixedValue, col: v })}
                        fixedValue={fixedValues.col || 1}
                        onFixedValueChange={(v) => setFixedValues({ ...fixedValues, col: v })}
                      />
                    </>
                  )}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t bg-slate-50">
          {markMutation.isError && (
            <div className="flex items-center gap-2 text-red-600 text-sm">
              <AlertCircle className="h-4 w-4" />
              {(markMutation.error as Error)?.message || 'Failed to create observation set'}
            </div>
          )}
          <div className="ml-auto flex items-center gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-100 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => markMutation.mutate()}
              disabled={!isValid || markMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {markMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <CheckCircle className="h-4 w-4" />
              )}
              Create Observation Set
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

interface ColumnMappingFieldProps {
  label: string
  description: string
  headers: string[]
  value: string | null
  onChange: (value: string) => void
  required?: boolean
}

function ColumnMappingField({
  label,
  description,
  headers,
  value,
  onChange,
  required,
}: ColumnMappingFieldProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      <select
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="">Select column...</option>
        {headers.map((h) => (
          <option key={h} value={h}>
            {h}
          </option>
        ))}
      </select>
      <p className="text-xs text-slate-400 mt-1">{description}</p>
    </div>
  )
}

interface ColumnMappingFieldWithFixedProps extends Omit<ColumnMappingFieldProps, 'required'> {
  useFixed: boolean
  onUseFixedChange: (value: boolean) => void
  fixedValue: number
  onFixedValueChange: (value: number) => void
}

function ColumnMappingFieldWithFixed({
  label,
  description,
  headers,
  value,
  onChange,
  useFixed,
  onUseFixedChange,
  fixedValue,
  onFixedValueChange,
}: ColumnMappingFieldWithFixedProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        <label className="flex items-center gap-1 text-xs text-slate-500">
          <input
            type="checkbox"
            checked={useFixed}
            onChange={(e) => onUseFixedChange(e.target.checked)}
            className="rounded text-blue-600"
          />
          Fixed value
        </label>
      </div>

      {useFixed ? (
        <input
          type="number"
          value={fixedValue}
          onChange={(e) => onFixedValueChange(Number(e.target.value))}
          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          min={1}
        />
      ) : (
        <select
          value={value || ''}
          onChange={(e) => onChange(e.target.value)}
          className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select column...</option>
          {headers.map((h) => (
            <option key={h} value={h}>
              {h}
            </option>
          ))}
        </select>
      )}
      <p className="text-xs text-slate-400 mt-1">{description}</p>
    </div>
  )
}
