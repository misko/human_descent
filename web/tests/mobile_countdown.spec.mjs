import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('Mobile countdown overlay', () => {
  test.use({ viewport: { width: 375, height: 667 }, deviceScaleFactor: 2 });

  test('countdown text stays visible on mobile', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?mode=3d&mobile=1&help=off&host=${SERVER_HOST}&port=${SERVER_PORT}`);

    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient.ControlType,
      { timeout: 10000 },
    );

    // Start countdown
    await page.evaluate(() => {
      window.__hudesClient?.startSpeedRun?.();
    });

    const overlay = page.locator('#speedrunCountdownOverlay');
    await expect(overlay).toBeVisible();

    const textLocator = overlay.locator('.speedrun-countdown__text');
    const textContent = await textLocator.innerText();
    expect(textContent.toUpperCase()).toContain('GET READY');

    const bbox = await textLocator.boundingBox();
    const viewport = page.viewportSize();
    expect(bbox).not.toBeNull();
    if (bbox) {
      expect(bbox.x).toBeGreaterThan(0);
      expect(bbox.x + bbox.width).toBeLessThanOrEqual(viewport.width + 1);
    }
  });
});
