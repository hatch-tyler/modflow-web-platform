import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/layout/Layout'
import ErrorBoundary from './components/layout/ErrorBoundary'
import ProjectsPage from './pages/ProjectsPage'
import UploadPage from './pages/UploadPage'
import ViewerPage from './pages/ViewerPage'
import ConsolePage from './pages/ConsolePage'
import DashboardPage from './pages/DashboardPage'
import FilesPage from './pages/FilesPage'
import PestPage from './pages/PestPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/projects" replace />} />
        <Route path="projects" element={<ErrorBoundary><ProjectsPage /></ErrorBoundary>} />
        <Route path="projects/:projectId">
          <Route path="upload" element={<ErrorBoundary><UploadPage /></ErrorBoundary>} />
          <Route path="viewer" element={<ErrorBoundary><ViewerPage /></ErrorBoundary>} />
          <Route path="console" element={<ErrorBoundary><ConsolePage /></ErrorBoundary>} />
          <Route path="dashboard" element={<ErrorBoundary><DashboardPage /></ErrorBoundary>} />
          <Route path="files" element={<ErrorBoundary><FilesPage /></ErrorBoundary>} />
          <Route path="pest" element={<ErrorBoundary><PestPage /></ErrorBoundary>} />
        </Route>
      </Route>
    </Routes>
  )
}

export default App
