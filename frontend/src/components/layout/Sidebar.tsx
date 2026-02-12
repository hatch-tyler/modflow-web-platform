import { useEffect } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  FolderOpen,
  Upload,
  Box,
  Terminal,
  LayoutDashboard,
  Settings,
  ChevronLeft,
  ChevronRight,
  Layers,
} from 'lucide-react'
import { useProjectStore } from '../../store/projectStore'
import { projectsApi } from '../../services/api'
import clsx from 'clsx'

interface NavItem {
  name: string
  href: string
  icon: React.ElementType
  requiresProject?: boolean
}

const navigation: NavItem[] = [
  { name: 'Projects', href: '/projects', icon: FolderOpen },
  { name: 'Upload Model', href: '/upload', icon: Upload, requiresProject: true },
  { name: '3D Viewer', href: '/viewer', icon: Box, requiresProject: true },
  { name: 'Console', href: '/console', icon: Terminal, requiresProject: true },
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard, requiresProject: true },
  { name: 'PEST / UQ', href: '/pest', icon: Settings, requiresProject: true },
]

export default function Sidebar() {
  const { projectId } = useParams()
  const location = useLocation()
  const { sidebarOpen, toggleSidebar, currentProject, setCurrentProject } = useProjectStore()

  // Fetch project if we have a projectId but no currentProject (or different project)
  const { data: fetchedProject } = useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectsApi.get(projectId!),
    enabled: !!projectId && (!currentProject || currentProject.id !== projectId),
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  })

  // Update store when project is fetched
  useEffect(() => {
    if (fetchedProject && (!currentProject || currentProject.id !== fetchedProject.id)) {
      setCurrentProject(fetchedProject)
    }
  }, [fetchedProject, currentProject, setCurrentProject])

  // Use fetched project if currentProject doesn't match projectId
  const displayProject = currentProject?.id === projectId ? currentProject : fetchedProject

  const getHref = (item: NavItem) => {
    if (item.requiresProject && projectId) {
      return `/projects/${projectId}${item.href}`
    }
    return item.href
  }

  const isActive = (item: NavItem) => {
    const href = getHref(item)
    return location.pathname === href || location.pathname.startsWith(href + '/')
  }

  return (
    <aside
      className={clsx(
        'fixed inset-y-0 left-0 z-50 flex flex-col bg-slate-900 text-white transition-all duration-300',
        sidebarOpen ? 'w-64' : 'w-16'
      )}
    >
      {/* Logo */}
      <div className="flex items-center h-16 px-4 border-b border-slate-700">
        <Layers className="h-8 w-8 text-blue-400 flex-shrink-0" />
        {sidebarOpen && (
          <span className="ml-3 text-lg font-semibold">MODFLOW Web</span>
        )}
      </div>

      {/* Current project indicator */}
      {displayProject && sidebarOpen && (
        <div className="px-4 py-3 border-b border-slate-700 bg-slate-800">
          <div className="text-xs text-slate-400">Current Project</div>
          <div className="text-sm font-medium truncate">{displayProject.name}</div>
          {displayProject.is_valid && (
            <div className="text-xs text-green-400 mt-1">
              {displayProject.grid_type === 'vertex'
                ? `${displayProject.nlay}L × ${displayProject.ncol} cells (DISV)`
                : displayProject.grid_type === 'unstructured'
                ? `${displayProject.ncol} nodes (DISU)`
                : `${displayProject.nlay}L × ${displayProject.nrow}R × ${displayProject.ncol}C`}
            </div>
          )}
        </div>
      )}
      {/* Show project indicator when sidebar is collapsed */}
      {displayProject && !sidebarOpen && (
        <div className="px-2 py-3 border-b border-slate-700 bg-slate-800" title={displayProject.name}>
          <div className="w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm">
            {displayProject.name.substring(0, 2).toUpperCase()}
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 py-4 overflow-y-auto">
        <ul className="space-y-1 px-2">
          {navigation.map((item) => {
            const disabled = item.requiresProject && !projectId
            const active = isActive(item)

            return (
              <li key={item.name}>
                {disabled ? (
                  <div
                    className={clsx(
                      'flex items-center px-3 py-2 rounded-lg text-slate-500 cursor-not-allowed',
                      !sidebarOpen && 'justify-center'
                    )}
                    title={sidebarOpen ? undefined : item.name}
                  >
                    <item.icon className="h-5 w-5 flex-shrink-0" />
                    {sidebarOpen && <span className="ml-3">{item.name}</span>}
                  </div>
                ) : (
                  <Link
                    to={getHref(item)}
                    className={clsx(
                      'flex items-center px-3 py-2 rounded-lg transition-colors',
                      active
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-300 hover:bg-slate-800 hover:text-white',
                      !sidebarOpen && 'justify-center'
                    )}
                    title={sidebarOpen ? undefined : item.name}
                  >
                    <item.icon className="h-5 w-5 flex-shrink-0" />
                    {sidebarOpen && <span className="ml-3">{item.name}</span>}
                  </Link>
                )}
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Toggle button */}
      <button
        onClick={toggleSidebar}
        className="flex items-center justify-center h-12 border-t border-slate-700 hover:bg-slate-800 transition-colors"
      >
        {sidebarOpen ? (
          <ChevronLeft className="h-5 w-5" />
        ) : (
          <ChevronRight className="h-5 w-5" />
        )}
      </button>
    </aside>
  )
}
