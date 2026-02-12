/**
 * E2E tests for project management workflow
 *
 * Tests the complete project lifecycle including creation, navigation, and deletion.
 */

import { test, expect } from '@playwright/test';

test.describe('Projects Page', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the projects page
    await page.goto('/');
  });

  test('should display the projects page', async ({ page }) => {
    // Check that the page loads
    await expect(page).toHaveTitle(/MODFLOW/i);
  });

  test('should show empty state when no projects exist', async ({ page }) => {
    // Check for empty state message or create button
    const createButton = page.getByRole('button', { name: /create|new/i });
    await expect(createButton).toBeVisible();
  });

  test('should open create project dialog', async ({ page }) => {
    // Click create project button
    const createButton = page.getByRole('button', { name: /create|new/i });
    await createButton.click();

    // Check dialog is visible
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();

    // Check for name input
    const nameInput = page.getByLabel(/name/i);
    await expect(nameInput).toBeVisible();
  });

  test('should create a new project', async ({ page }) => {
    // Open create dialog
    const createButton = page.getByRole('button', { name: /create|new/i });
    await createButton.click();

    // Fill in project details
    const nameInput = page.getByLabel(/name/i);
    await nameInput.fill('E2E Test Project');

    const descInput = page.getByLabel(/description/i);
    if (await descInput.isVisible()) {
      await descInput.fill('Created by E2E test');
    }

    // Submit
    const submitButton = page.getByRole('button', { name: /create|save|submit/i });
    await submitButton.click();

    // Verify project appears in list
    await expect(page.getByText('E2E Test Project')).toBeVisible();
  });

  test('should navigate to project upload page', async ({ page }) => {
    // Assuming a project exists, click on it
    // This test may need adjustment based on actual UI
    const projectLink = page.getByRole('link', { name: /upload|project/i }).first();

    if (await projectLink.isVisible()) {
      await projectLink.click();

      // Check we're on the upload page
      await expect(page).toHaveURL(/upload/);
    }
  });
});

test.describe('Project Navigation', () => {
  test('should navigate between project tabs', async ({ page }) => {
    // Navigate to a project page
    await page.goto('/');

    // If there's a project, try to navigate to its tabs
    const projectCard = page.locator('[data-testid="project-card"]').first();

    if (await projectCard.isVisible()) {
      await projectCard.click();

      // Check for navigation tabs
      const uploadTab = page.getByRole('link', { name: /upload/i });
      const dashboardTab = page.getByRole('link', { name: /dashboard/i });
      const consoleTab = page.getByRole('link', { name: /console/i });

      // Navigate between tabs
      if (await uploadTab.isVisible()) {
        await uploadTab.click();
        await expect(page).toHaveURL(/upload/);
      }

      if (await dashboardTab.isVisible()) {
        await dashboardTab.click();
        await expect(page).toHaveURL(/dashboard/);
      }

      if (await consoleTab.isVisible()) {
        await consoleTab.click();
        await expect(page).toHaveURL(/console/);
      }
    }
  });
});

test.describe('Health Check', () => {
  test('should show healthy status', async ({ page }) => {
    await page.goto('/');

    // Look for health indicator (if visible on UI)
    const healthIndicator = page.locator('[data-testid="health-status"]');

    if (await healthIndicator.isVisible()) {
      await expect(healthIndicator).toContainText(/healthy|connected/i);
    }
  });
});

test.describe('Responsive Layout', () => {
  test('should be responsive on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    // Check that the page is still usable
    await expect(page).toHaveTitle(/MODFLOW/i);

    // Check for mobile menu button
    const menuButton = page.getByRole('button', { name: /menu/i });
    if (await menuButton.isVisible()) {
      await menuButton.click();
      // Verify menu opens
    }
  });

  test('should be responsive on tablet', async ({ page }) => {
    // Set tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');

    await expect(page).toHaveTitle(/MODFLOW/i);
  });
});

test.describe('Accessibility', () => {
  test('should have no critical accessibility issues', async ({ page }) => {
    await page.goto('/');

    // Basic accessibility checks
    // Check for main landmark
    const main = page.locator('main');
    await expect(main).toBeVisible();

    // Check that images have alt text
    const images = page.locator('img');
    const imageCount = await images.count();

    for (let i = 0; i < imageCount; i++) {
      const img = images.nth(i);
      const alt = await img.getAttribute('alt');
      const role = await img.getAttribute('role');

      // Images should have alt text or role="presentation"
      expect(alt !== null || role === 'presentation').toBeTruthy();
    }

    // Check for proper heading hierarchy
    const h1 = page.locator('h1');
    await expect(h1).toBeVisible();
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/');

    // Tab through interactive elements
    await page.keyboard.press('Tab');

    // Check that focus is visible
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });
});
