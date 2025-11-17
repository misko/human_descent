import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('Keyboard overlay toggle', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      try {
        window.sessionStorage?.clear();
      } catch {}
    });
  });

  test('Shift+? and X toggle keyboard overlay on desktop', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?mode=3d&help=off&host=${SERVER_HOST}&port=${SERVER_PORT}`);

    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient.ControlType,
      { timeout: 10000 },
    );

    const overlay = page.locator('.tour-screen--keys');
    await expect(overlay).toBeHidden();

    await page.keyboard.press('Shift+Slash');
    await expect(overlay).toBeVisible();

    await page.keyboard.press('Shift+Slash');
    await expect(overlay).toBeHidden();

    await page.keyboard.press('Shift+Slash');
    await expect(overlay).toBeVisible();

    await page.keyboard.press('x');
    await expect(overlay).toBeHidden();
  });
});
