import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('HUD toggle flag', () => {
  test('hideHUD=1 hides HUD on desktop', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?mode=3d&hideHUD=1&host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);
    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient.ControlType,
      { timeout: 10000 },
    );
    await expect(page.locator('#bottomTextContainer')).toBeHidden();
  });

  test('hideHUD=1 hides HUD on mobile', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?mode=3d&mobile=1&hideHUD=1&host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);
    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient.ControlType,
      { timeout: 10000 },
    );
    await expect(page.locator('#bottomTextContainer')).toBeHidden();
  });
});
