/**
 * Tests for projectStore Zustand store
 *
 * Tests state management for projects, runs, sidebar, and upload tracking.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { act } from '@testing-library/react';
import { useProjectStore } from '@/store/projectStore';
import type { Project, Run } from '@/types';

// Helper to reset store between tests
const resetStore = () => {
  useProjectStore.setState({
    currentProject: null,
    currentRun: null,
    sidebarOpen: true,
    uploadProgress: 0,
    isUploading: false,
  });
};

describe('projectStore', () => {
  beforeEach(() => {
    resetStore();
  });

  describe('currentProject', () => {
    it('should start with null currentProject', () => {
      const { currentProject } = useProjectStore.getState();
      expect(currentProject).toBeNull();
    });

    it('should set currentProject', () => {
      const testProject: Project = {
        id: '550e8400-e29b-41d4-a716-446655440001',
        name: 'Test Project',
        description: 'A test project',
        model_type: 'mf6',
        nlay: 3,
        nrow: 100,
        ncol: 100,
        nper: 12,
        is_valid: true,
        created_at: '2024-01-15T10:00:00Z',
        updated_at: '2024-01-15T10:00:00Z',
      };

      act(() => {
        useProjectStore.getState().setCurrentProject(testProject);
      });

      const { currentProject } = useProjectStore.getState();
      expect(currentProject).toEqual(testProject);
    });

    it('should clear currentProject when set to null', () => {
      const testProject: Project = {
        id: '550e8400-e29b-41d4-a716-446655440001',
        name: 'Test Project',
        description: null,
        model_type: 'mf6',
        nlay: 3,
        nrow: 100,
        ncol: 100,
        nper: 12,
        is_valid: true,
        created_at: '2024-01-15T10:00:00Z',
        updated_at: '2024-01-15T10:00:00Z',
      };

      act(() => {
        useProjectStore.getState().setCurrentProject(testProject);
      });

      act(() => {
        useProjectStore.getState().setCurrentProject(null);
      });

      const { currentProject } = useProjectStore.getState();
      expect(currentProject).toBeNull();
    });

    it('should replace currentProject when setting a different project', () => {
      const project1: Project = {
        id: '550e8400-e29b-41d4-a716-446655440001',
        name: 'Project 1',
        description: null,
        model_type: 'mf6',
        nlay: 1,
        nrow: 10,
        ncol: 10,
        nper: 1,
        is_valid: true,
        created_at: '2024-01-15T10:00:00Z',
        updated_at: '2024-01-15T10:00:00Z',
      };

      const project2: Project = {
        id: '550e8400-e29b-41d4-a716-446655440002',
        name: 'Project 2',
        description: null,
        model_type: 'mf2005',
        nlay: 2,
        nrow: 20,
        ncol: 20,
        nper: 2,
        is_valid: true,
        created_at: '2024-01-16T10:00:00Z',
        updated_at: '2024-01-16T10:00:00Z',
      };

      act(() => {
        useProjectStore.getState().setCurrentProject(project1);
      });

      act(() => {
        useProjectStore.getState().setCurrentProject(project2);
      });

      const { currentProject } = useProjectStore.getState();
      expect(currentProject?.id).toBe(project2.id);
      expect(currentProject?.name).toBe('Project 2');
    });
  });

  describe('currentRun', () => {
    it('should start with null currentRun', () => {
      const { currentRun } = useProjectStore.getState();
      expect(currentRun).toBeNull();
    });

    it('should set currentRun', () => {
      const testRun: Run = {
        id: '660e8400-e29b-41d4-a716-446655440001',
        project_id: '550e8400-e29b-41d4-a716-446655440001',
        name: 'Test Run',
        run_type: 'forward',
        status: 'completed',
        started_at: '2024-01-15T15:00:00Z',
        completed_at: '2024-01-15T15:05:00Z',
        exit_code: 0,
        created_at: '2024-01-15T15:00:00Z',
        updated_at: '2024-01-15T15:05:00Z',
      };

      act(() => {
        useProjectStore.getState().setCurrentRun(testRun);
      });

      const { currentRun } = useProjectStore.getState();
      expect(currentRun).toEqual(testRun);
    });

    it('should clear currentRun when set to null', () => {
      const testRun: Run = {
        id: '660e8400-e29b-41d4-a716-446655440001',
        project_id: '550e8400-e29b-41d4-a716-446655440001',
        name: 'Test Run',
        run_type: 'forward',
        status: 'running',
        created_at: '2024-01-15T15:00:00Z',
        updated_at: '2024-01-15T15:00:00Z',
      };

      act(() => {
        useProjectStore.getState().setCurrentRun(testRun);
      });

      act(() => {
        useProjectStore.getState().setCurrentRun(null);
      });

      const { currentRun } = useProjectStore.getState();
      expect(currentRun).toBeNull();
    });
  });

  describe('sidebar', () => {
    it('should start with sidebar open', () => {
      const { sidebarOpen } = useProjectStore.getState();
      expect(sidebarOpen).toBe(true);
    });

    it('should toggle sidebar from open to closed', () => {
      act(() => {
        useProjectStore.getState().toggleSidebar();
      });

      const { sidebarOpen } = useProjectStore.getState();
      expect(sidebarOpen).toBe(false);
    });

    it('should toggle sidebar from closed to open', () => {
      act(() => {
        useProjectStore.getState().toggleSidebar();
      });

      act(() => {
        useProjectStore.getState().toggleSidebar();
      });

      const { sidebarOpen } = useProjectStore.getState();
      expect(sidebarOpen).toBe(true);
    });

    it('should set sidebar open explicitly', () => {
      act(() => {
        useProjectStore.getState().setSidebarOpen(false);
      });

      expect(useProjectStore.getState().sidebarOpen).toBe(false);

      act(() => {
        useProjectStore.getState().setSidebarOpen(true);
      });

      expect(useProjectStore.getState().sidebarOpen).toBe(true);
    });
  });

  describe('upload state', () => {
    it('should start with uploadProgress at 0', () => {
      const { uploadProgress } = useProjectStore.getState();
      expect(uploadProgress).toBe(0);
    });

    it('should start with isUploading false', () => {
      const { isUploading } = useProjectStore.getState();
      expect(isUploading).toBe(false);
    });

    it('should set uploadProgress', () => {
      act(() => {
        useProjectStore.getState().setUploadProgress(50);
      });

      const { uploadProgress } = useProjectStore.getState();
      expect(uploadProgress).toBe(50);
    });

    it('should set isUploading', () => {
      act(() => {
        useProjectStore.getState().setIsUploading(true);
      });

      const { isUploading } = useProjectStore.getState();
      expect(isUploading).toBe(true);
    });

    it('should track upload lifecycle', () => {
      // Start upload
      act(() => {
        useProjectStore.getState().setIsUploading(true);
        useProjectStore.getState().setUploadProgress(0);
      });

      expect(useProjectStore.getState().isUploading).toBe(true);
      expect(useProjectStore.getState().uploadProgress).toBe(0);

      // Progress updates
      act(() => {
        useProjectStore.getState().setUploadProgress(25);
      });
      expect(useProjectStore.getState().uploadProgress).toBe(25);

      act(() => {
        useProjectStore.getState().setUploadProgress(50);
      });
      expect(useProjectStore.getState().uploadProgress).toBe(50);

      act(() => {
        useProjectStore.getState().setUploadProgress(100);
      });
      expect(useProjectStore.getState().uploadProgress).toBe(100);

      // Complete upload
      act(() => {
        useProjectStore.getState().setIsUploading(false);
      });

      expect(useProjectStore.getState().isUploading).toBe(false);
    });

    it('should handle upload progress reset', () => {
      // Set progress to 100
      act(() => {
        useProjectStore.getState().setUploadProgress(100);
      });

      // Reset for new upload
      act(() => {
        useProjectStore.getState().setUploadProgress(0);
      });

      const { uploadProgress } = useProjectStore.getState();
      expect(uploadProgress).toBe(0);
    });
  });

  describe('state isolation', () => {
    it('should not affect other state when setting currentProject', () => {
      const initialState = useProjectStore.getState();

      act(() => {
        useProjectStore.getState().setCurrentProject({
          id: 'test-id',
          name: 'Test',
          description: null,
          model_type: 'mf6',
          nlay: 1,
          nrow: 1,
          ncol: 1,
          nper: 1,
          is_valid: true,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        });
      });

      const newState = useProjectStore.getState();
      expect(newState.sidebarOpen).toBe(initialState.sidebarOpen);
      expect(newState.uploadProgress).toBe(initialState.uploadProgress);
      expect(newState.isUploading).toBe(initialState.isUploading);
    });

    it('should not affect currentProject when toggling sidebar', () => {
      const testProject: Project = {
        id: 'test-id',
        name: 'Test',
        description: null,
        model_type: 'mf6',
        nlay: 1,
        nrow: 1,
        ncol: 1,
        nper: 1,
        is_valid: true,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
      };

      act(() => {
        useProjectStore.getState().setCurrentProject(testProject);
      });

      act(() => {
        useProjectStore.getState().toggleSidebar();
      });

      const { currentProject } = useProjectStore.getState();
      expect(currentProject).toEqual(testProject);
    });
  });
});
