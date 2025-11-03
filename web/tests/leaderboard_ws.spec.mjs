// Playwright test: press Y to request Top 100 via WebSocket and see modal
import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('Leaderboard over WebSocket', () => {
  test('press Y shows top 100 modal', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);
    await page.waitForFunction(() => window.__hudesClient && window.__hudesClient.ControlType);

    // Press Y to request leaderboard
    await page.keyboard.press('KeyY');

    // Expect a modal with Top 100 content rendered
    await page.waitForSelector('#modalOverlay.open .glass-card');
  const title = await page.locator('#modalOverlay.open .glass-card h2').textContent();
  expect(title).toMatch(/Top 10/i);

    // If the server has no scores yet, the list might be empty; that's fine.
    // Just ensure the structure is present.
    await expect(page.locator('#modalOverlay.open .glass-card .scroll-wrap')).toBeVisible();
  });
});
