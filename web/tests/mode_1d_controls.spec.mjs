import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('1D keyboard controls', () => {
  test('paired key presses update dims without speed run', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?mode=1d&host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);

  await page.waitForFunction(
    () => window.__hudesClient && window.__hudesClient.ControlType,
    { timeout: 10000 },
  );

  const closeButton = page.locator('.help-overlay__close');
  if (await closeButton.count()) {
    await closeButton.first().click();
  }

    await page.waitForFunction(() => {
      const client = window.__hudesClient;
      if (!client) return false;
      if (!Array.isArray(client.dimsAndStepsOnCurrentDims)) {
        client.zeroDimsAndStepsOnCurrentDims?.();
      }
      return Array.isArray(client.dimsAndStepsOnCurrentDims);
    });

    await page.evaluate(() => window.__hudesClient.zeroDimsAndStepsOnCurrentDims?.());
    const before = await page.evaluate(() => [...window.__hudesClient.dimsAndStepsOnCurrentDims]);

  await page.keyboard.press('KeyW');

    await page.waitForFunction(() => {
      const client = window.__hudesClient;
      if (!client || !Array.isArray(client.dimsAndStepsOnCurrentDims)) return false;
      return client.dimsAndStepsOnCurrentDims.some((value) => Math.abs(value) > 0);
    }, { timeout: 5000 });

    const after = await page.evaluate(() => [...window.__hudesClient.dimsAndStepsOnCurrentDims]);
    expect(after).not.toEqual(before);

    const speedRunActive = await page.evaluate(() => window.__hudesClient?.state?.speedRunActive);
    expect(speedRunActive).toBeFalsy();
  });
});
