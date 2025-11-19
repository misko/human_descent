import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('HUD toggle flag', () => {
  test.use({ viewport: { width: 1200, height: 720 } });

  test('hideHUD=1 hides HUD on desktop', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?mode=3d&hideHUD=1&host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);
    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient.ControlType,
      { timeout: 10000 },
    );
    await expect(page.locator('#bottomTextContainer')).toHaveCount(0);
  });

  test('hideHUD=1 hides HUD on mobile', async ({ page }) => {
    await page.context().newCDPSession(page).then((session) =>
      session.send('Emulation.setDeviceMetricsOverride', {
        width: 375,
        height: 667,
        deviceScaleFactor: 2,
        mobile: true,
        scale: 1,
        screenWidth: 375,
        screenHeight: 667,
      }),
    );
    await page.goto(`${APP_ORIGIN}/?mode=3d&mobile=1&hideHUD=1&host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);
    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient.ControlType,
      { timeout: 10000 },
    );
    await expect(page.locator('#bottomTextContainer')).toHaveCount(0);
  });
});
