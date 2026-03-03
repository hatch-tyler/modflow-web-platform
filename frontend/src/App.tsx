import { lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import Layout from './components/layout/Layout'
import ErrorBoundary from './components/layout/ErrorBoundary'
import ProjectsPage from './pages/ProjectsPage'

// Lazy-load heavy pages so the initial bundle (ProjectsPage) stays small.
// ViewerPage pulls in Three.js (~600KB), DashboardPage/PestPage pull in
// Plotly.js (~3.5MB), FilesPage pulls in Monaco (~2MB).
const UploadPage = lazy(() => import('./pages/UploadPage'))
const ViewerPage = lazy(() => import('./pages/ViewerPage'))
const ConsolePage = lazy(() => import('./pages/ConsolePage'))
const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const FilesPage = lazy(() => import('./pages/FilesPage'))
const PestPage = lazy(() => import('./pages/PestPage'))

function PageFallback() {
  return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
    </div>
  )
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/projects" replace />} />
        <Route path="projects" element={<ErrorBoundary><ProjectsPage /></ErrorBoundary>} />
        <Route path="projects/:projectId">
          <Route path="upload" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><UploadPage /></Suspense></ErrorBoundary>} />
          <Route path="viewer" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><ViewerPage /></Suspense></ErrorBoundary>} />
          <Route path="console" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><ConsolePage /></Suspense></ErrorBoundary>} />
          <Route path="dashboard" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><DashboardPage /></Suspense></ErrorBoundary>} />
          <Route path="files" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><FilesPage /></Suspense></ErrorBoundary>} />
          <Route path="pest" element={<ErrorBoundary><Suspense fallback={<PageFallback />}><PestPage /></Suspense></ErrorBoundary>} />
        </Route>
      </Route>
    </Routes>
  )
}

export default App
