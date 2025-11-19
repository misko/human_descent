import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const SPEED_SECONDS = Number(process.env.HUDES_SPEED_RUN_SECONDS ?? '120');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('Speed Run keyboard restore', () => {
  test('keyboard resumes after submitting score', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);
    await page.waitForFunction(() => window.__hudesClient && window.__hudesClient.ControlType);

    await page.keyboard.press('KeyZ');
    await page.waitForFunction(
      () => {
        const c = window.__hudesClient;
        return c && c.state && c.state.speedRunActive === true;
      },
      { timeout: 10_000 },
    );

    await page.keyboard.press('Space');
    await page.waitForFunction(
      () => {
        const c = window.__hudesClient;
        return c && c.trainSteps && c.trainSteps.length > 0;
      },
    );
    await page.waitForSelector('#modalOverlay.open .glass-card .name-form', { timeout: (SPEED_SECONDS + 40) * 1000 });

    await page.fill('#modalOverlay.open .glass-card .name-form input', 'TEST');
    await page.click('#modalOverlay.open .glass-card .name-form button[type="submit"]');

    await expect(page.locator('#modalOverlay')).not.toHaveClass(/open/);
    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient._textCaptureActive === false,
    );

    await page.keyboard.press('Space');
    const trainStepsAfter = await page.evaluate(() => window.__hudesClient?.trainSteps?.length ?? 0);
    await page.waitForFunction(
      (prev) => {
        const current = window.__hudesClient?.trainSteps?.length ?? 0;
        return current >= prev;
      },
      trainStepsAfter,
      { timeout: 5_000 },
    );
  });
});
