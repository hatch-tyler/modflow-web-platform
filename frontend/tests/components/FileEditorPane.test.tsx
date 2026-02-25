/**
 * Tests for FileEditorPane component
 *
 * Tests save mutation error handling (B6 fix) and cache invalidation (E7).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useFileEditorStore } from '@/store/fileEditorStore'

// Mock Monaco editor to avoid jsdom issues
vi.mock('@/components/files/MonacoWrapper', () => ({
  default: ({ content, onChange }: { content: string; onChange: (v: string) => void }) => (
    <textarea
      data-testid="mock-editor"
      value={content}
      onChange={(e) => onChange(e.target.value)}
    />
  ),
}))

// Mock the tooltip provider
vi.mock('@/components/files/ModflowTooltipProvider', () => ({
  default: () => null,
}))

// Mock EditorToolbar to expose save button
vi.mock('@/components/files/EditorToolbar', () => ({
  default: ({
    onSave,
    isDirty,
    saving,
  }: {
    filePath: string
    isDirty: boolean
    saving: boolean
    backups: unknown[]
    onSave: () => void
    onRevert: (ts: string) => void
  }) => (
    <div data-testid="toolbar">
      <button
        data-testid="save-btn"
        onClick={onSave}
        disabled={!isDirty || saving}
      >
        {saving ? 'Saving...' : 'Save'}
      </button>
    </div>
  ),
}))

// Control the fileEditorApi mock
const mockGetContent = vi.fn()
const mockSaveContent = vi.fn()
const mockGetBackups = vi.fn()
const mockRevert = vi.fn()

vi.mock('@/services/api', () => ({
  fileEditorApi: {
    getContent: (...args: unknown[]) => mockGetContent(...args),
    saveContent: (...args: unknown[]) => mockSaveContent(...args),
    getBackups: (...args: unknown[]) => mockGetBackups(...args),
    revert: (...args: unknown[]) => mockRevert(...args),
  },
}))

// Import the component AFTER mocks
import FileEditorPane from '@/components/files/FileEditorPane'

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  })

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe('FileEditorPane', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset store
    useFileEditorStore.setState({
      openFilePath: null,
      originalContent: null,
      editedContent: null,
      isDirty: false,
      backups: [],
    })

    // Default mock responses
    mockGetContent.mockResolvedValue({ content: 'BEGIN OPTIONS\nEND OPTIONS\n', size: 28, encoding: 'utf-8' })
    mockGetBackups.mockResolvedValue({ backups: [] })
    mockSaveContent.mockResolvedValue({ saved: true, size: 28, backup_timestamp: null })
  })

  it('should show placeholder when no file is selected', () => {
    const Wrapper = createWrapper()
    render(
      <Wrapper>
        <FileEditorPane projectId="proj-1" filePath={null} modelType="mf6" />
      </Wrapper>,
    )

    expect(screen.getByText('Select a file to edit')).toBeInTheDocument()
  })

  it('should load file content and display editor', async () => {
    const Wrapper = createWrapper()
    render(
      <Wrapper>
        <FileEditorPane projectId="proj-1" filePath="model.nam" modelType="mf6" />
      </Wrapper>,
    )

    await waitFor(() => {
      expect(mockGetContent).toHaveBeenCalledWith('proj-1', 'model.nam')
    })

    // After loading, editor should be rendered
    await waitFor(() => {
      expect(screen.getByTestId('mock-editor')).toBeInTheDocument()
    })
  })

  describe('save error handling (B6 fix)', () => {
    it('should display error banner when save fails', async () => {
      mockSaveContent.mockRejectedValue(new Error('Network error'))

      const Wrapper = createWrapper()

      // Pre-populate the store as if file is loaded and dirty
      useFileEditorStore.setState({
        openFilePath: 'model.nam',
        originalContent: 'original',
        editedContent: 'modified content',
        isDirty: true,
      })

      render(
        <Wrapper>
          <FileEditorPane projectId="proj-1" filePath="model.nam" modelType="mf6" />
        </Wrapper>,
      )

      // Click save
      const saveBtn = screen.getByTestId('save-btn')
      await act(async () => {
        fireEvent.click(saveBtn)
      })

      // Error banner should appear (use colon to distinguish from status bar text)
      await waitFor(() => {
        expect(screen.getByText(/Save failed:/)).toBeInTheDocument()
      })
    })

    it('should dismiss error banner when Dismiss is clicked', async () => {
      mockSaveContent.mockRejectedValue(new Error('Server error'))

      const Wrapper = createWrapper()

      useFileEditorStore.setState({
        openFilePath: 'model.nam',
        originalContent: 'original',
        editedContent: 'modified',
        isDirty: true,
      })

      render(
        <Wrapper>
          <FileEditorPane projectId="proj-1" filePath="model.nam" modelType="mf6" />
        </Wrapper>,
      )

      // Trigger save failure
      await act(async () => {
        fireEvent.click(screen.getByTestId('save-btn'))
      })

      await waitFor(() => {
        expect(screen.getByText('Dismiss')).toBeInTheDocument()
      })

      // Click dismiss
      await act(async () => {
        fireEvent.click(screen.getByText('Dismiss'))
      })

      // Error banner should be gone
      await waitFor(() => {
        expect(screen.queryByText(/Save failed:/)).not.toBeInTheDocument()
      })
    })

    it('should clear error on successful save after a failure', async () => {
      // First save fails
      mockSaveContent.mockRejectedValueOnce(new Error('Temporary error'))

      const Wrapper = createWrapper()

      useFileEditorStore.setState({
        openFilePath: 'model.nam',
        originalContent: 'original',
        editedContent: 'modified',
        isDirty: true,
      })

      render(
        <Wrapper>
          <FileEditorPane projectId="proj-1" filePath="model.nam" modelType="mf6" />
        </Wrapper>,
      )

      // First save attempt - fails
      await act(async () => {
        fireEvent.click(screen.getByTestId('save-btn'))
      })

      await waitFor(() => {
        expect(screen.getByText(/Save failed:/)).toBeInTheDocument()
      })

      // Second save attempt - succeeds
      mockSaveContent.mockResolvedValueOnce({ saved: true, size: 8, backup_timestamp: null })

      // Re-dirty the content to enable save button
      act(() => {
        useFileEditorStore.getState().updateContent('modified again')
      })

      await act(async () => {
        fireEvent.click(screen.getByTestId('save-btn'))
      })

      // Error banner (with colon) should be cleared after success
      await waitFor(() => {
        expect(screen.queryByText(/Save failed:/)).not.toBeInTheDocument()
      })
    })
  })

  describe('cache invalidation (E7)', () => {
    it('should invalidate queries on successful save', async () => {
      mockSaveContent.mockResolvedValue({ saved: true, size: 10, backup_timestamp: null })

      const queryClient = new QueryClient({
        defaultOptions: {
          queries: { retry: false, gcTime: 0 },
          mutations: { retry: false },
        },
      })

      const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries')

      useFileEditorStore.setState({
        openFilePath: 'model.nam',
        originalContent: 'original',
        editedContent: 'modified',
        isDirty: true,
      })

      render(
        <QueryClientProvider client={queryClient}>
          <FileEditorPane projectId="proj-1" filePath="model.nam" modelType="mf6" />
        </QueryClientProvider>,
      )

      // Click save
      await act(async () => {
        fireEvent.click(screen.getByTestId('save-btn'))
      })

      // Wait for mutation to complete
      await waitFor(() => {
        expect(mockSaveContent).toHaveBeenCalled()
      })

      // Check that invalidateQueries was called with file-content and project keys
      await waitFor(() => {
        const calls = invalidateSpy.mock.calls
        const queryKeys = calls.map((c) => (c[0] as { queryKey: unknown[] })?.queryKey)
        expect(queryKeys).toContainEqual(['file-content', 'proj-1'])
        expect(queryKeys).toContainEqual(['project', 'proj-1'])
      })

      invalidateSpy.mockRestore()
    })
  })
})
