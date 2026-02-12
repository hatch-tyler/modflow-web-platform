import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import Header from './Header'
import { useProjectStore } from '../../store/projectStore'
import clsx from 'clsx'

export default function Layout() {
  const { sidebarOpen } = useProjectStore()

  return (
    <div className="min-h-screen bg-slate-50 flex">
      <Sidebar />
      <div
        className={clsx(
          'flex-1 flex flex-col transition-all duration-300',
          sidebarOpen ? 'ml-64' : 'ml-16'
        )}
      >
        <Header />
        <main className="flex-1 p-6 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
