import { useQuery } from '@tanstack/react-query'
import { Activity, AlertCircle, CheckCircle } from 'lucide-react'
import { healthApi } from '../../services/api'
import ActiveRunsIndicator from './ActiveRunsIndicator'
import clsx from 'clsx'

export default function Header() {
  const { data: health, isLoading, isError } = useQuery({
    queryKey: ['health'],
    queryFn: healthApi.check,
    refetchInterval: 30000, // Check every 30 seconds
    retry: false,
  })

  return (
    <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <h1 className="text-xl font-semibold text-slate-800">
          MODFLOW Web Platform
        </h1>
      </div>

      <div className="flex items-center gap-4">
        {/* Active runs indicator */}
        <ActiveRunsIndicator />

        {/* Health indicator */}
        <div className="flex items-center gap-2 text-sm">
          {isLoading ? (
            <Activity className="h-4 w-4 text-slate-400 animate-pulse" />
          ) : isError ? (
            <>
              <AlertCircle className="h-4 w-4 text-red-500" />
              <span className="text-red-600">Disconnected</span>
            </>
          ) : (
            <>
              <CheckCircle
                className={clsx(
                  'h-4 w-4',
                  health?.status === 'healthy' ? 'text-green-500' : 'text-yellow-500'
                )}
              />
              <span
                className={clsx(
                  health?.status === 'healthy' ? 'text-green-600' : 'text-yellow-600'
                )}
              >
                {health?.status === 'healthy' ? 'All Systems Operational' : 'Degraded'}
              </span>
              <span className="text-slate-400 text-xs">v{health?.version}</span>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
