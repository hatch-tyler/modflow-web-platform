import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo)
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[400px] p-8">
          <AlertTriangle className="h-12 w-12 text-red-400 mb-4" />
          <h2 className="text-lg font-semibold text-slate-800 mb-2">
            Something went wrong
          </h2>
          <p className="text-sm text-slate-500 mb-4 max-w-md text-center">
            An unexpected error occurred while rendering this page.
            Try refreshing or click the button below.
          </p>
          {this.state.error && (
            <pre className="text-xs text-red-600 bg-red-50 border border-red-200 rounded p-3 mb-4 max-w-lg overflow-auto max-h-32">
              {this.state.error.message}
            </pre>
          )}
          <button
            onClick={this.handleRetry}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            Try Again
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
