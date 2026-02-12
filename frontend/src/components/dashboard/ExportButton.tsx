import { Download } from 'lucide-react'

interface ExportButtonProps {
  url: string
  label?: string
}

export default function ExportButton({ url, label = 'CSV' }: ExportButtonProps) {
  const handleClick = () => {
    const a = document.createElement('a')
    a.href = url
    a.download = ''
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <button
      onClick={handleClick}
      className="flex items-center gap-1 px-2 py-1 text-xs rounded border border-slate-300 text-slate-500 hover:text-slate-700 hover:border-slate-400 transition-colors"
      title={`Export ${label}`}
    >
      <Download className="h-3.5 w-3.5" />
      <span>{label}</span>
    </button>
  )
}
