import { useRef, useEffect } from 'react'
import Editor, { type OnMount, type OnChange } from '@monaco-editor/react'
import type { editor as MonacoEditor } from 'monaco-editor'

interface MonacoWrapperProps {
  content: string
  onChange: (value: string) => void
  filePath: string
  readOnly?: boolean
}

// Register custom MODFLOW languages once
let languagesRegistered = false

function registerModflowLanguages(monaco: Parameters<OnMount>[1]) {
  if (languagesRegistered) return
  languagesRegistered = true

  // MF6 block-structured language
  monaco.languages.register({ id: 'modflow6' })
  monaco.languages.setMonarchTokensProvider('modflow6', {
    ignoreCase: true,
    keywords: [
      'BEGIN', 'END', 'OPEN/CLOSE', 'CONSTANT', 'INTERNAL',
      'SAVE', 'PRINT', 'HEAD', 'BUDGET', 'FILEOUT',
      'OPTIONS', 'DIMENSIONS', 'GRIDDATA', 'PACKAGEDATA',
      'CONNECTIONDATA', 'PERIODDATA', 'PERIOD', 'VERTICES',
      'CELL2D', 'CROSSSECTIONS', 'EXCHANGEDATA',
      'STEADY-STATE', 'TRANSIENT',
    ],
    typeKeywords: [
      'SIMPLE', 'MODERATE', 'COMPLEX',
      'CG', 'BICGSTAB',
      'NONE', 'DBD', 'COOLEY', 'MOMENTUM',
      'RCM', 'MD',
    ],
    packageTypes: [
      'DIS6', 'DISV6', 'DISU6', 'TDIS6', 'IMS6',
      'NPF6', 'STO6', 'IC6', 'OC6',
      'WEL6', 'RCH6', 'EVT6', 'CHD6', 'GHB6', 'DRN6', 'RIV6',
      'SFR6', 'MAW6', 'LAK6', 'UZF6', 'MVR6', 'GNC6', 'HFB6',
      'CSUB6', 'BUY6',
    ],
    tokenizer: {
      root: [
        [/#.*$/, 'comment'],
        [/\b(BEGIN|END)\b/i, 'keyword.control'],
        [/\b(OPEN\/CLOSE|CONSTANT|INTERNAL)\b/i, 'keyword'],
        [/\b(SAVE|PRINT|FILEOUT)\b/i, 'keyword'],
        [/\b(OPTIONS|DIMENSIONS|GRIDDATA|PACKAGEDATA|CONNECTIONDATA|PERIODDATA|PERIOD|VERTICES|CELL2D)\b/i, 'keyword.block'],
        [/\b(STEADY-STATE|TRANSIENT)\b/i, 'keyword.type'],
        [/\b(SIMPLE|MODERATE|COMPLEX)\b/i, 'type'],
        [/[-+]?\d*\.?\d+[eEdD][-+]?\d+/, 'number.float'],
        [/[-+]?\d*\.\d+/, 'number.float'],
        [/[-+]?\d+/, 'number'],
        [/'[^']*'/, 'string'],
        [/"[^"]*"/, 'string'],
        [/[A-Z_][A-Z0-9_]*/i, 'variable'],
      ],
    },
  })

  // Classic MODFLOW language
  monaco.languages.register({ id: 'modflow-classic' })
  monaco.languages.setMonarchTokensProvider('modflow-classic', {
    ignoreCase: true,
    tokenizer: {
      root: [
        [/#.*$/, 'comment'],
        [/^#.*$/, 'comment'],
        [/[-+]?\d*\.?\d+[eEdD][-+]?\d+/, 'number.float'],
        [/[-+]?\d*\.\d+/, 'number.float'],
        [/[-+]?\d+/, 'number'],
        [/\b(INTERNAL|EXTERNAL|OPEN\/CLOSE|CONSTANT)\b/i, 'keyword'],
        [/\b(FREE|XSECTION|PRINTTIME)\b/i, 'keyword'],
        [/'[^']*'/, 'string'],
        [/"[^"]*"/, 'string'],
      ],
    },
  })
}

// Detect language from file extension
function detectLanguage(filePath: string): string {
  const ext = ('.' + filePath.split('.').pop()!).toLowerCase()
  const mf6Extensions = new Set([
    '.nam', '.tdis', '.ims', '.dis', '.disv', '.disu',
    '.npf', '.sto', '.ic', '.oc', '.wel', '.rch', '.evt',
    '.chd', '.ghb', '.drn', '.riv', '.sfr', '.maw', '.lak',
    '.uzf', '.hfb', '.mvr', '.gnc', '.csub', '.buy',
  ])
  const classicExtensions = new Set([
    '.bas', '.lpf', '.bcf', '.upw', '.pcg', '.nwt', '.sms',
    '.gmg', '.de4',
  ])

  // Check if file content starts with BEGIN (MF6-style)
  if (mf6Extensions.has(ext)) return 'modflow6'
  if (classicExtensions.has(ext)) return 'modflow-classic'
  if (ext === '.lst' || ext === '.list') return 'plaintext'
  return 'modflow6' // Default for unknown MODFLOW files
}

export default function MonacoWrapper({
  content,
  onChange,
  filePath,
  readOnly = false,
}: MonacoWrapperProps) {
  const editorRef = useRef<MonacoEditor.IStandaloneCodeEditor | null>(null)
  const language = detectLanguage(filePath)

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor
    registerModflowLanguages(monaco)

    // Update language model
    const model = editor.getModel()
    if (model) {
      monaco.editor.setModelLanguage(model, language)
    }
  }

  const handleChange: OnChange = (value) => {
    if (value !== undefined) {
      onChange(value)
    }
  }

  // Update language when file changes
  useEffect(() => {
    if (editorRef.current) {
      const monaco = (window as { monaco?: typeof import('monaco-editor') }).monaco
      if (monaco) {
        const model = editorRef.current.getModel()
        if (model) {
          monaco.editor.setModelLanguage(model, language)
        }
      }
    }
  }, [filePath, language])

  return (
    <Editor
      defaultLanguage={language}
      value={content}
      onChange={handleChange}
      onMount={handleMount}
      theme="vs"
      options={{
        readOnly,
        minimap: { enabled: true },
        fontSize: 13,
        fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
        lineNumbers: 'on',
        renderWhitespace: 'selection',
        wordWrap: 'on',
        scrollBeyondLastLine: false,
        automaticLayout: true,
        tabSize: 2,
        folding: true,
        bracketPairColorization: { enabled: false },
      }}
      loading={
        <div className="flex items-center justify-center h-full text-slate-400">
          Loading editor...
        </div>
      }
    />
  )
}
