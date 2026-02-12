import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/layout/Layout'
import ProjectsPage from './pages/ProjectsPage'
import UploadPage from './pages/UploadPage'
import ViewerPage from './pages/ViewerPage'
import ConsolePage from './pages/ConsolePage'
import DashboardPage from './pages/DashboardPage'
import PestPage from './pages/PestPage'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/projects" replace />} />
        <Route path="projects" element={<ProjectsPage />} />
        <Route path="projects/:projectId">
          <Route path="upload" element={<UploadPage />} />
          <Route path="viewer" element={<ViewerPage />} />
          <Route path="console" element={<ConsolePage />} />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="pest" element={<PestPage />} />
        </Route>
      </Route>
    </Routes>
  )
}

export default App
