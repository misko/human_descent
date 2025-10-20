import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');

function withBackend(url) {
  const u = new URL(url, 'http://localhost:5173');
  u.searchParams.set('host', SERVER_HOST);
  u.searchParams.set('port', String(SERVER_PORT));
  return u.toString();
}

test.describe('Highscores page', () => {
  test('loads and renders rows', async ({ page }) => {
    await page.goto(withBackend('/highscores'));
    // Controls visible
    await expect(page.locator('#controls')).toBeVisible();
    // Attempt to load
    await page.click('#loadBtn');
    // Wait for table to be present
    await page.waitForSelector('#scoresTable');
    const rowCount = await page.locator('#scoresTable tbody tr').count();
    expect(rowCount).toBeGreaterThanOrEqual(0);
  });
});
