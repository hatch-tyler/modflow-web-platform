import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import type { GridMeshData, ArrayData } from '../../utils/binaryParser'
import { valuesToColors } from '../../utils/binaryParser'
import { BOUNDARY_COLORS, type BoundaryPackage } from '../../pages/ViewerPage'

interface GridViewerProps {
  gridData: GridMeshData
  arrayData?: ArrayData
  iboundData?: ArrayData
  showInactive?: boolean
  visibleLayers: boolean[]
  colormap?: 'viridis' | 'plasma' | 'coolwarm' | 'terrain'
  opacity?: number
  boundaries?: Record<string, BoundaryPackage>
  verticalExaggeration?: number
  showWireframe?: boolean
  cellMask?: Uint8Array
}

export default function GridViewer({
  gridData,
  arrayData,
  iboundData,
  showInactive = false,
  visibleLayers,
  colormap = 'viridis',
  opacity = 0.9,
  boundaries = {},
  verticalExaggeration = 10,
  showWireframe = true,
  cellMask,
}: GridViewerProps) {
  const { nlay, nrow, ncol } = gridData

  // Floating origin: offset coords to median centroid (robust to outliers)
  const sceneOrigin = useMemo(() => {
    const { centers } = gridData
    const ncells = centers.length / 3

    // Use median instead of mean so a single outlier cell
    // doesn't pull the origin away from the main grid body
    const xs = new Float32Array(ncells)
    const ys = new Float32Array(ncells)
    const zs = new Float32Array(ncells)
    for (let i = 0; i < ncells; i++) {
      xs[i] = centers[i * 3]
      ys[i] = centers[i * 3 + 1]
      zs[i] = centers[i * 3 + 2]
    }
    xs.sort()
    ys.sort()
    zs.sort()

    const mid = Math.floor(ncells / 2)
    return {
      x: xs[mid],
      y: ys[mid],
      z: zs[mid],
    }
  }, [gridData])

  // Build a map of boundary cells for highlighting
  const boundaryCellMap = useMemo(() => {
    const map = new Map<string, { type: string; color: string }>()
    Object.entries(boundaries).forEach(([type, pkg]) => {
      const color = BOUNDARY_COLORS[type] || '#ffffff'
      pkg.cells.forEach(cell => {
        const key = `${cell.layer}-${cell.row}-${cell.col}`
        map.set(key, { type, color })
      })
    })
    return map
  }, [boundaries])

  // Create color array from property data
  const cellColors = useMemo(() => {
    const totalCells = nlay * nrow * ncol
    if (arrayData && arrayData.data.length === totalCells) {
      return valuesToColors(arrayData.data, colormap as any)
    }
    // Default color: teal/aqua
    const colors = new Float32Array(totalCells * 3)
    for (let i = 0; i < totalCells; i++) {
      colors[i * 3] = 0.2
      colors[i * 3 + 1] = 0.7
      colors[i * 3 + 2] = 0.9
    }
    return colors
  }, [arrayData, nlay, nrow, ncol, colormap])

  // Compute grid bounding box from gridData directly (independent of visibleLayers)
  // This ensures camera is positioned correctly from the very first render.
  const gridBbox = useMemo(() => {
    const box = new THREE.Box3()
    const isStructured = gridData.gridType === 0
    const { centers, top, botm } = gridData

    if (isStructured && gridData.delr && gridData.delc) {
      // Compute X/Y extent from delr/delc
      let totalX = 0
      for (let j = 0; j < ncol; j++) totalX += gridData.delr[j]
      let totalYVal = 0
      for (let i = 0; i < nrow; i++) totalYVal += gridData.delc[i]
      const x0 = -sceneOrigin.x
      const x1 = totalX - sceneOrigin.x
      const y0 = -sceneOrigin.y
      const y1 = totalYVal - sceneOrigin.y

      // Z extent from top/botm with VE
      let zMin = Infinity, zMax = -Infinity
      for (let i = 0; i < top.length; i++) {
        zMax = Math.max(zMax, top[i])
      }
      for (let i = 0; i < botm.length; i++) {
        zMin = Math.min(zMin, botm[i])
      }
      const z0 = zMin * verticalExaggeration - sceneOrigin.z * verticalExaggeration
      const z1 = zMax * verticalExaggeration - sceneOrigin.z * verticalExaggeration

      box.set(
        new THREE.Vector3(Math.min(x0, x1), Math.min(y0, y1), Math.min(z0, z1)),
        new THREE.Vector3(Math.max(x0, x1), Math.max(y0, y1), Math.max(z0, z1)),
      )
    } else {
      // USG or fallback: compute from centers with IQR-based outlier filtering.
      // This prevents a single far-away cell from making the bbox enormous.
      const ncellsTotal = nlay * nrow * ncol
      const cxs = new Float32Array(ncellsTotal)
      const cys = new Float32Array(ncellsTotal)
      for (let i = 0; i < ncellsTotal; i++) {
        cxs[i] = centers[i * 3]
        cys[i] = centers[i * 3 + 1]
      }
      cxs.sort()
      cys.sort()

      const q1x = cxs[Math.floor(ncellsTotal * 0.25)]
      const q3x = cxs[Math.floor(ncellsTotal * 0.75)]
      const iqrX = q3x - q1x
      const q1y = cys[Math.floor(ncellsTotal * 0.25)]
      const q3y = cys[Math.floor(ncellsTotal * 0.75)]
      const iqrY = q3y - q1y

      // 3x IQR fence (generous, includes ~99.7% of normal data).
      // If IQR=0 (all values identical on one axis), skip filtering on that axis.
      const lowerX = iqrX > 0 ? q1x - 3 * iqrX : -Infinity
      const upperX = iqrX > 0 ? q3x + 3 * iqrX : Infinity
      const lowerY = iqrY > 0 ? q1y - 3 * iqrY : -Infinity
      const upperY = iqrY > 0 ? q3y + 3 * iqrY : Infinity

      for (let i = 0; i < ncellsTotal; i++) {
        const cx = centers[i * 3]
        const cy = centers[i * 3 + 1]
        // Skip outlier cells for bbox/camera computation
        if (cx < lowerX || cx > upperX || cy < lowerY || cy > upperY) continue
        const x = cx - sceneOrigin.x
        const y = cy - sceneOrigin.y
        const z = centers[i * 3 + 2] * verticalExaggeration - sceneOrigin.z * verticalExaggeration
        box.expandByPoint(new THREE.Vector3(x, y, z))
      }
      // Expand slightly to account for cell size around centers
      if (!box.isEmpty()) {
        const size = new THREE.Vector3()
        box.getSize(size)
        const padding = Math.max(size.x, size.y) * 0.02
        box.expandByScalar(padding)
      }
    }
    return box
  }, [gridData, nlay, nrow, ncol, sceneOrigin, verticalExaggeration])

  // Unified geometry builder: produces merged BufferGeometry for ALL grid types
  const { geometry, wireframeGeometry, boundaryGeometry } = useMemo(() => {
    const isStructured = gridData.gridType === 0
    const { centers, top, botm } = gridData

    // Pre-compute structured grid edges if needed
    let xEdges: number[] | null = null
    let yEdges: number[] | null = null
    let totalY = 0
    if (isStructured && gridData.delr && gridData.delc) {
      xEdges = [0]
      for (let j = 0; j < ncol; j++) {
        xEdges.push(xEdges[j] + gridData.delr[j])
      }
      yEdges = [0]
      for (let i = 0; i < nrow; i++) {
        yEdges.push(yEdges[i] + gridData.delc[i])
      }
      totalY = yEdges[yEdges.length - 1]
    }

    // Count visible cells first
    let visibleCount = 0
    for (let k = 0; k < nlay; k++) {
      if (!visibleLayers[k]) continue
      visibleCount += nrow * ncol
    }

    if (visibleCount === 0) {
      const emptyGeo = new THREE.BufferGeometry()
      return {
        geometry: emptyGeo,
        wireframeGeometry: emptyGeo,
        boundaryGeometry: null,
      }
    }

    // Allocate typed arrays
    const positions = new Float32Array(visibleCount * 8 * 3)
    const colors = new Float32Array(visibleCount * 8 * 3)
    const indices = new Uint32Array(visibleCount * 36)
    const wireframeIndices = new Uint32Array(visibleCount * 24)

    // Boundary tracking
    const boundaryPositions: number[] = []
    const boundaryColors: number[] = []
    const boundaryWireIndices: number[] = []
    let boundaryVertCount = 0

    let visIdx = 0

    // Face triangle indices template (6 faces * 2 triangles * 3 indices = 36)
    const faceTemplate = [
      0, 1, 2, 0, 2, 3,  // bottom
      5, 4, 7, 5, 7, 6,  // top
      3, 2, 6, 3, 6, 7,  // front
      4, 5, 1, 4, 1, 0,  // back
      1, 5, 6, 1, 6, 2,  // right
      4, 0, 3, 4, 3, 7,  // left
    ]

    // Edge line indices template (12 edges * 2 = 24)
    const edgeTemplate = [
      0, 1, 1, 2, 2, 3, 3, 0,
      4, 5, 5, 6, 6, 7, 7, 4,
      0, 4, 1, 5, 2, 6, 3, 7,
    ]

    // Structured grids: slight gap for visual cell separation
    // USG grids: no gap - cells share edges and should tile exactly
    const gap = isStructured ? 0.98 : 1.0

    for (let k = 0; k < nlay; k++) {
      if (!visibleLayers[k]) continue

      for (let i = 0; i < nrow; i++) {
        for (let j = 0; j < ncol; j++) {
          const cellIdx = k * nrow * ncol + i * ncol + j
          const baseIdx = i * ncol + j  // index within a layer

          // Skip inactive cells if showInactive is false
          if (!showInactive && iboundData && iboundData.data[cellIdx] <= 0) {
            continue
          }

          // Skip cells hidden by cross-section mask
          if (cellMask && !cellMask[cellIdx]) continue

          let x0: number, x1: number, y0: number, y1: number

          if (isStructured && xEdges && yEdges) {
            // Reconstruct exact vertices from delr/delc
            x0 = xEdges[j]
            x1 = xEdges[j + 1]
            y0 = totalY - yEdges[i + 1]
            y1 = totalY - yEdges[i]
          } else if (gridData.vertices) {
            // USG: use actual vertices from backend
            const vBase = cellIdx * 8 * 3
            x0 = gridData.vertices[vBase]      // vertex 0 x
            x1 = gridData.vertices[vBase + 3]  // vertex 1 x
            y0 = gridData.vertices[vBase + 1]   // vertex 0 y
            y1 = gridData.vertices[vBase + 7]   // vertex 2 y
          } else {
            // Fallback: shouldn't happen but be safe
            const cx = centers[cellIdx * 3]
            const cy = centers[cellIdx * 3 + 1]
            x0 = cx - 50
            x1 = cx + 50
            y0 = cy - 50
            y1 = cy + 50
          }

          const zTop = k === 0 ? top[baseIdx] : botm[(k - 1) * nrow * ncol + baseIdx]
          const zBot = botm[k * nrow * ncol + baseIdx]

          // Skip zero-thickness cells (inactive cells with top == bot)
          if (Math.abs(zTop - zBot) < 1e-6) continue

          // Apply gap for visual separation
          const midX = (x0 + x1) / 2
          const midY = (y0 + y1) / 2
          const gx0 = midX + (x0 - midX) * gap
          const gx1 = midX + (x1 - midX) * gap
          const gy0 = midY + (y0 - midY) * gap
          const gy1 = midY + (y1 - midY) * gap

          // Apply vertical exaggeration and floating origin offset
          const zTopVE = zTop * verticalExaggeration
          const zBotVE = zBot * verticalExaggeration
          const midZ = (zTopVE + zBotVE) / 2
          const gzBot = midZ + (zBotVE - midZ) * gap
          const gzTop = midZ + (zTopVE - midZ) * gap

          // Subtract scene origin (floating origin)
          const ox = -sceneOrigin.x
          const oy = -sceneOrigin.y
          const oz = -sceneOrigin.z * verticalExaggeration

          // 8 vertices: bottom face (0-3), top face (4-7)
          const verts = [
            gx0 + ox, gy0 + oy, gzBot + oz,
            gx1 + ox, gy0 + oy, gzBot + oz,
            gx1 + ox, gy1 + oy, gzBot + oz,
            gx0 + ox, gy1 + oy, gzBot + oz,
            gx0 + ox, gy0 + oy, gzTop + oz,
            gx1 + ox, gy0 + oy, gzTop + oz,
            gx1 + ox, gy1 + oy, gzTop + oz,
            gx0 + ox, gy1 + oy, gzTop + oz,
          ]

          const baseVertex = visIdx * 8
          const posOff = baseVertex * 3

          for (let v = 0; v < 8; v++) {
            positions[posOff + v * 3] = verts[v * 3]
            positions[posOff + v * 3 + 1] = verts[v * 3 + 1]
            positions[posOff + v * 3 + 2] = verts[v * 3 + 2]
          }

          // Per-cell color from cellColors
          const r = cellColors[cellIdx * 3]
          const g = cellColors[cellIdx * 3 + 1]
          const b = cellColors[cellIdx * 3 + 2]
          for (let v = 0; v < 8; v++) {
            colors[posOff + v * 3] = r
            colors[posOff + v * 3 + 1] = g
            colors[posOff + v * 3 + 2] = b
          }

          // Triangle indices
          const triOff = visIdx * 36
          for (let t = 0; t < 36; t++) {
            indices[triOff + t] = baseVertex + faceTemplate[t]
          }

          // Wireframe indices
          const wireOff = visIdx * 24
          for (let e = 0; e < 24; e++) {
            wireframeIndices[wireOff + e] = baseVertex + edgeTemplate[e]
          }

          // Boundary cell check
          const boundaryKey = `${k}-${i}-${j}`
          if (boundaryCellMap.has(boundaryKey)) {
            const bInfo = boundaryCellMap.get(boundaryKey)!
            const bColor = new THREE.Color(bInfo.color)
            const bBase = boundaryVertCount

            for (let v = 0; v < 8; v++) {
              boundaryPositions.push(verts[v * 3], verts[v * 3 + 1], verts[v * 3 + 2])
              boundaryColors.push(bColor.r, bColor.g, bColor.b)
            }

            for (let e = 0; e < 24; e++) {
              boundaryWireIndices.push(bBase + edgeTemplate[e])
            }
            boundaryVertCount += 8
          }

          visIdx++
        }
      }
    }

    // Trim arrays to actual cell count (cells may have been skipped)
    const actualVerts = visIdx * 8 * 3
    const actualTris = visIdx * 36
    const actualEdges = visIdx * 24
    const usedPositions = visIdx < visibleCount ? positions.subarray(0, actualVerts) : positions
    const usedColors = visIdx < visibleCount ? colors.subarray(0, actualVerts) : colors
    const usedIndices = visIdx < visibleCount ? indices.subarray(0, actualTris) : indices
    const usedWireIndices = visIdx < visibleCount ? wireframeIndices.subarray(0, actualEdges) : wireframeIndices

    // Main geometry
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(usedPositions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(usedColors, 3))
    geo.setIndex(new THREE.BufferAttribute(usedIndices, 1))
    geo.computeVertexNormals()
    geo.computeBoundingBox()

    // Wireframe geometry (shares positions)
    const wireGeo = new THREE.BufferGeometry()
    wireGeo.setAttribute('position', new THREE.BufferAttribute(usedPositions, 3))
    wireGeo.setIndex(new THREE.BufferAttribute(usedWireIndices, 1))

    // Boundary geometry
    let bGeo: THREE.BufferGeometry | null = null
    if (boundaryVertCount > 0) {
      bGeo = new THREE.BufferGeometry()
      bGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(boundaryPositions), 3))
      bGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(boundaryColors), 3))
      bGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(boundaryWireIndices), 1))
    }

    return { geometry: geo, wireframeGeometry: wireGeo, boundaryGeometry: bGeo }
  }, [gridData, nlay, nrow, ncol, visibleLayers, cellColors, verticalExaggeration, boundaryCellMap, sceneOrigin, iboundData, showInactive, cellMask])

  // Camera auto-fit from bounding box (uses gridBbox which is independent of visibleLayers)
  const cameraSetup = useMemo(() => {
    if (gridBbox.isEmpty()) {
      return {
        position: [0, 0, 100] as [number, number, number],
        target: [0, 0, 0] as [number, number, number],
        near: 0.1,
        far: 10000,
        minDistance: 1,
        maxDistance: 10000,
      }
    }

    const size = new THREE.Vector3()
    const center = new THREE.Vector3()
    gridBbox.getSize(size)
    gridBbox.getCenter(center)

    const maxDim = Math.max(size.x, size.y, size.z)
    const fov = 45
    const distance = (maxDim / 2) / Math.tan((fov / 2) * Math.PI / 180)

    // Oblique camera angle
    const camDist = distance * 1.3
    const position: [number, number, number] = [
      center.x + camDist * 0.2,
      center.y - camDist * 0.15,
      center.z + camDist * 0.9,
    ]

    return {
      position,
      target: [center.x, center.y, center.z] as [number, number, number],
      near: Math.max(0.1, maxDim * 0.001),
      far: maxDim * 20,
      minDistance: Math.max(0.5, maxDim * 0.005),
      maxDistance: maxDim * 10,
    }
  }, [gridBbox])

  // Scene helpers proportional to geometry
  const helperSize = useMemo(() => {
    if (gridBbox.isEmpty()) return { axes: 10, grid: 100 }
    const size = new THREE.Vector3()
    gridBbox.getSize(size)
    const maxDim = Math.max(size.x, size.y)
    return {
      axes: maxDim * 0.15,
      grid: maxDim * 0.8,
    }
  }, [gridBbox])

  const gridHelperZ = useMemo(() => {
    if (gridBbox.isEmpty()) return 0
    const size = new THREE.Vector3()
    gridBbox.getSize(size)
    return gridBbox.min.z - size.z * 0.2
  }, [gridBbox])

  return (
    <Canvas
      gl={{
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance',
        failIfMajorPerformanceCaveat: false,
      }}
      style={{ background: 'linear-gradient(180deg, #1e3a5f 0%, #0f172a 100%)' }}
      camera={{ up: [0, 0, 1] }}
      onCreated={({ camera, gl }) => {
        camera.up.set(0, 0, 1)
        gl.domElement.addEventListener('webglcontextlost', (e) => {
          e.preventDefault()
          console.warn('WebGL context lost - will attempt recovery')
        })
        gl.domElement.addEventListener('webglcontextrestored', () => {
          console.log('WebGL context restored')
        })
      }}
    >
      <PerspectiveCamera
        makeDefault
        position={cameraSetup.position}
        fov={45}
        near={cameraSetup.near}
        far={cameraSetup.far}
        up={[0, 0, 1]}
      />

      <ambientLight intensity={0.7} />
      <directionalLight position={[1, 1, 2]} intensity={1.0} color="#ffffff" />
      <directionalLight position={[-1, -1, 1]} intensity={0.5} color="#ffffff" />
      <hemisphereLight args={['#ffffff', '#444444', 0.4]} />

      <OrbitControls
        target={cameraSetup.target}
        enableDamping
        dampingFactor={0.1}
        zoomSpeed={2}
        minDistance={cameraSetup.minDistance}
        maxDistance={cameraSetup.maxDistance}
        makeDefault
      />

      {/* Main filled mesh */}
      <mesh geometry={geometry} frustumCulled={false}>
        <meshBasicMaterial
          vertexColors
          transparent
          opacity={opacity}
          side={THREE.DoubleSide}
          polygonOffset
          polygonOffsetFactor={1}
          polygonOffsetUnits={1}
        />
      </mesh>

      {/* Wireframe overlay */}
      {showWireframe && (
        <lineSegments geometry={wireframeGeometry} frustumCulled={false}>
          <lineBasicMaterial color="#ffffff" transparent opacity={0.15} />
        </lineSegments>
      )}

      {/* Boundary highlights */}
      {boundaryGeometry && (
        <lineSegments geometry={boundaryGeometry} frustumCulled={false}>
          <lineBasicMaterial vertexColors transparent opacity={1} linewidth={2} />
        </lineSegments>
      )}

      {/* Axes helper - sized to geometry */}
      <axesHelper args={[helperSize.axes]} />

      {/* Ground plane - sized to geometry */}
      <gridHelper
        args={[helperSize.grid, 20, '#4a6fa5', '#2d4a6f']}
        position={[cameraSetup.target[0], cameraSetup.target[1], gridHelperZ]}
        rotation={[Math.PI / 2, 0, 0]}
      />
    </Canvas>
  )
}
