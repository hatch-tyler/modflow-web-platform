import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { modflowDocsApi } from '../../services/api'

interface ModflowTooltipProviderProps {
  modelType: string
  filePath: string
  editorRef?: React.RefObject<unknown>
}

// Map file extension to package name
function extensionToPackage(filePath: string): string | null {
  const ext = ('.' + filePath.split('.').pop()!).toLowerCase()
  const map: Record<string, string> = {
    '.ims': 'ims', '.tdis': 'tdis', '.dis': 'dis', '.disv': 'disv',
    '.disu': 'disu', '.npf': 'npf', '.sto': 'sto', '.ic': 'ic',
    '.oc': 'oc', '.wel': 'wel', '.rch': 'rch', '.evt': 'evt',
    '.chd': 'chd', '.ghb': 'ghb', '.drn': 'drn', '.riv': 'riv',
    '.sfr': 'sfr', '.maw': 'maw', '.lak': 'lak', '.uzf': 'uzf',
    '.hfb': 'hfb', '.mvr': 'mvr', '.gnc': 'gnc', '.csub': 'csub',
    '.buy': 'buy',
    // Classic
    '.pcg': 'pcg', '.nwt': 'nwt', '.sms': 'sms', '.gmg': 'gmg',
    '.de4': 'de4', '.bas': 'bas6', '.lpf': 'lpf', '.bcf': 'bcf',
    '.upw': 'upw', '.mnw': 'mnw2',
  }
  return map[ext] || null
}

export default function ModflowTooltipProvider({
  modelType,
  filePath,
}: ModflowTooltipProviderProps) {
  const packageName = extensionToPackage(filePath)
  const disposableRef = useRef<{ dispose: () => void } | null>(null)

  const { data: definition } = useQuery({
    queryKey: ['modflow-definition', modelType, packageName],
    queryFn: () => modflowDocsApi.getDefinition(modelType, packageName!),
    enabled: !!packageName,
    staleTime: Infinity, // Definitions never change
  })

  useEffect(() => {
    if (!definition) return

    // Build a lookup map: variable name -> tooltip info
    const varMap = new Map<string, { description: string; tooltip?: string; type?: string; choices?: string[] }>()

    // MF6 block-based definitions
    if (definition.blocks) {
      for (const block of definition.blocks) {
        for (const variable of block.variables || []) {
          varMap.set(variable.name.toUpperCase(), {
            description: variable.description || '',
            tooltip: variable.tooltip || variable.description || '',
            type: variable.type,
            choices: variable.choices,
          })
        }
      }
    }

    // Classic dataset-based definitions
    if (definition.datasets) {
      for (const dataset of definition.datasets) {
        for (const variable of dataset.variables || []) {
          varMap.set(variable.name.toUpperCase(), {
            description: variable.description || '',
            tooltip: variable.tooltip || variable.description || '',
            type: variable.type,
          })
        }
      }
    }

    // Register Monaco hover provider
    const monaco = (window as { monaco?: typeof import('monaco-editor') }).monaco
    if (!monaco) return

    // Dispose previous provider
    if (disposableRef.current) {
      disposableRef.current.dispose()
    }

    disposableRef.current = monaco.languages.registerHoverProvider(
      ['modflow6', 'modflow-classic'],
      {
        provideHover(model, position) {
          const word = model.getWordAtPosition(position)
          if (!word) return null

          const varInfo = varMap.get(word.word.toUpperCase())
          if (!varInfo) return null

          const contents: { value: string }[] = [
            { value: `**${word.word.toUpperCase()}**` },
            { value: varInfo.tooltip || varInfo.description },
          ]

          if (varInfo.type) {
            contents.push({ value: `*Type: ${varInfo.type}*` })
          }
          if (varInfo.choices?.length) {
            contents.push({ value: `Options: \`${varInfo.choices.join('` | `')}\`` })
          }

          return {
            range: {
              startLineNumber: position.lineNumber,
              startColumn: word.startColumn,
              endLineNumber: position.lineNumber,
              endColumn: word.endColumn,
            },
            contents,
          }
        },
      }
    )

    return () => {
      if (disposableRef.current) {
        disposableRef.current.dispose()
        disposableRef.current = null
      }
    }
  }, [definition])

  // This component has no visual output - it just registers tooltip providers
  return null
}
