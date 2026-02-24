import { useRef, useEffect, useState, useCallback, useMemo } from 'react'
import type { GridMeshData } from '../../utils/binaryParser'

interface CrossSectionPanelProps {
  gridData: GridMeshData
  polyline: [number, number][]
  isDrawing: boolean
  onUpdatePolyline: (pts: [number, number][]) => void
  onFinishDrawing: () => void
}

export default function CrossSectionPanel({
  gridData,
  polyline,
  isDrawing,
  onUpdatePolyline,
  onFinishDrawing,
}: CrossSectionPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null)
  const clickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Compute grid extent from gridData
  const extent = useMemo(() => {
    const isStructured = gridData.gridType === 0

    if (isStructured && gridData.delr && gridData.delc) {
      let xMax = 0
      for (let j = 0; j < gridData.ncol; j++) xMax += gridData.delr[j]
      let yMax = 0
      for (let i = 0; i < gridData.nrow; i++) yMax += gridData.delc[i]
      return { xMin: 0, xMax, yMin: 0, yMax }
    }

    // Unstructured: scan centers for extent
    const ncells = gridData.nlay * gridData.nrow * gridData.ncol
    let xMin = Infinity, xMax = -Infinity
    let yMin = Infinity, yMax = -Infinity
    for (let i = 0; i < ncells; i++) {
      const x = gridData.centers[i * 3]
      const y = gridData.centers[i * 3 + 1]
      if (x < xMin) xMin = x
      if (x > xMax) xMax = x
      if (y < yMin) yMin = y
      if (y > yMax) yMax = y
    }
    const padX = (xMax - xMin) * 0.05
    const padY = (yMax - yMin) * 0.05
    return {
      xMin: xMin - padX,
      xMax: xMax + padX,
      yMin: yMin - padY,
      yMax: yMax + padY,
    }
  }, [gridData])

  // Transform functions between model coords and canvas pixels
  const getTransforms = useCallback((canvasW: number, canvasH: number) => {
    const modelW = extent.xMax - extent.xMin
    const modelH = extent.yMax - extent.yMin
    if (modelW === 0 || modelH === 0) return null

    const padding = 16
    const availW = canvasW - padding * 2
    const availH = canvasH - padding * 2
    const scale = Math.min(availW / modelW, availH / modelH)
    const offsetX = padding + (availW - modelW * scale) / 2
    const offsetY = padding + (availH - modelH * scale) / 2

    const modelToCanvas = (mx: number, my: number): [number, number] => {
      const cx = offsetX + (mx - extent.xMin) * scale
      const cy = offsetY + (extent.yMax - my) * scale // flip Y
      return [cx, cy]
    }

    const canvasToModel = (cx: number, cy: number): [number, number] => {
      const mx = (cx - offsetX) / scale + extent.xMin
      const my = extent.yMax - (cy - offsetY) / scale // flip Y
      return [mx, my]
    }

    return { modelToCanvas, canvasToModel, scale }
  }, [extent])

  // Draw the canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height
    const transforms = getTransforms(w, h)
    if (!transforms) return

    const { modelToCanvas } = transforms
    const isStructured = gridData.gridType === 0

    // Clear
    ctx.clearRect(0, 0, w, h)

    // Draw grid
    if (isStructured && gridData.delr && gridData.delc) {
      // Structured: draw rectangle with grid lines
      const [x0, y0] = modelToCanvas(extent.xMin, extent.yMax)
      const [x1, y1] = modelToCanvas(extent.xMax, extent.yMin)
      const rectW = x1 - x0
      const rectH = y1 - y0

      // Fill background
      ctx.fillStyle = 'rgba(100, 116, 139, 0.15)'
      ctx.fillRect(x0, y0, rectW, rectH)

      // Grid lines - columns
      ctx.strokeStyle = 'rgba(150, 150, 150, 0.15)'
      ctx.lineWidth = 0.5
      let xPos = 0
      for (let j = 0; j < gridData.ncol; j++) {
        xPos += gridData.delr[j]
        const [cx] = modelToCanvas(xPos, 0)
        ctx.beginPath()
        ctx.moveTo(cx, y0)
        ctx.lineTo(cx, y1)
        ctx.stroke()
      }

      // Grid lines - rows
      let yPos = 0
      for (let i = 0; i < gridData.nrow; i++) {
        yPos += gridData.delc[i]
        const [, cy] = modelToCanvas(0, extent.yMax - yPos)
        ctx.beginPath()
        ctx.moveTo(x0, cy)
        ctx.lineTo(x1, cy)
        ctx.stroke()
      }

      // Border
      ctx.strokeStyle = 'rgba(150, 150, 150, 0.4)'
      ctx.lineWidth = 1
      ctx.strokeRect(x0, y0, rectW, rectH)
    } else {
      // Unstructured: draw cell centers as dots (layer 0 only for speed)
      ctx.fillStyle = 'rgba(150, 150, 150, 0.4)'
      const ncpl = gridData.nrow * gridData.ncol
      for (let i = 0; i < ncpl; i++) {
        const x = gridData.centers[i * 3]
        const y = gridData.centers[i * 3 + 1]
        const [cx, cy] = modelToCanvas(x, y)
        ctx.beginPath()
        ctx.arc(cx, cy, 1.5, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Draw polyline
    if (polyline.length > 0) {
      ctx.strokeStyle = '#3b82f6'
      ctx.lineWidth = 2
      ctx.beginPath()
      const [sx, sy] = modelToCanvas(polyline[0][0], polyline[0][1])
      ctx.moveTo(sx, sy)
      for (let i = 1; i < polyline.length; i++) {
        const [px, py] = modelToCanvas(polyline[i][0], polyline[i][1])
        ctx.lineTo(px, py)
      }
      ctx.stroke()

      // Vertex markers
      ctx.fillStyle = '#3b82f6'
      for (const pt of polyline) {
        const [px, py] = modelToCanvas(pt[0], pt[1])
        ctx.beginPath()
        ctx.arc(px, py, 4, 0, Math.PI * 2)
        ctx.fill()
      }

      // Preview line from last vertex to cursor while drawing
      if (isDrawing && cursorPos && polyline.length > 0) {
        const lastPt = polyline[polyline.length - 1]
        const [lx, ly] = modelToCanvas(lastPt[0], lastPt[1])
        ctx.strokeStyle = '#3b82f6'
        ctx.lineWidth = 1.5
        ctx.setLineDash([6, 4])
        ctx.beginPath()
        ctx.moveTo(lx, ly)
        ctx.lineTo(cursorPos.x, cursorPos.y)
        ctx.stroke()
        ctx.setLineDash([])
      }
    }
  }, [gridData, extent, polyline, isDrawing, cursorPos, getTransforms])

  // Redraw when state changes
  useEffect(() => {
    draw()
  }, [draw])

  // ResizeObserver for canvas sizing
  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        const dpr = window.devicePixelRatio || 1
        canvas.width = width * dpr
        canvas.height = height * dpr
        canvas.style.width = `${width}px`
        canvas.style.height = `${height}px`
        const ctx = canvas.getContext('2d')
        if (ctx) ctx.scale(dpr, dpr)
        draw()
      }
    })

    observer.observe(container)
    return () => observer.disconnect()
  }, [draw])

  // Handle click - add vertex to polyline
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return

    // Suppress click from double-click via timer
    if (clickTimerRef.current) {
      clearTimeout(clickTimerRef.current)
      clickTimerRef.current = null
      return
    }

    clickTimerRef.current = setTimeout(() => {
      clickTimerRef.current = null
      const canvas = canvasRef.current
      if (!canvas) return
      const rect = canvas.getBoundingClientRect()
      const dpr = window.devicePixelRatio || 1
      const cx = (e.clientX - rect.left) * dpr
      const cy = (e.clientY - rect.top) * dpr
      const transforms = getTransforms(canvas.width, canvas.height)
      if (!transforms) return

      // Use pre-DPR canvas coords for transform since we scaled the context
      const [mx, my] = transforms.canvasToModel(cx / dpr, cy / dpr)
      onUpdatePolyline([...polyline, [mx, my]])
    }, 200)
  }, [isDrawing, polyline, onUpdatePolyline, getTransforms])

  // Handle double-click - finish drawing
  const handleDoubleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    if (!isDrawing) return

    // Cancel pending single-click
    if (clickTimerRef.current) {
      clearTimeout(clickTimerRef.current)
      clickTimerRef.current = null
    }

    if (polyline.length >= 2) {
      onFinishDrawing()
    }
  }, [isDrawing, polyline, onFinishDrawing])

  // Handle mouse move - update cursor for preview line
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    setCursorPos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    })
  }, [isDrawing])

  const handleMouseLeave = useCallback(() => {
    setCursorPos(null)
  }, [])

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <canvas
        ref={canvasRef}
        className={`w-full h-full ${isDrawing ? 'cursor-crosshair' : 'cursor-default'}`}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {isDrawing && polyline.length === 0 && (
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-xs text-slate-400 bg-slate-900/70 px-2 py-1 rounded pointer-events-none">
          Click to place points, double-click to finish
        </div>
      )}
    </div>
  )
}
