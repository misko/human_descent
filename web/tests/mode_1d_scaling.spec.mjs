import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('1D plot scaling', () => {
  test('loss lines stay within frame at large step size', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?mode=1d&host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);

    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient.ControlType,
      { timeout: 10000 },
    );

    const closeButton = page.locator('.help-overlay__close');
    if (await closeButton.count()) {
      await closeButton.first().click();
    }

    await page.evaluate(() => {
      const client = window.__hudesClient;
      if (!client) {
        return;
      }
      client.state.stepSizeIdx = -80;
      client.state.updateStepSize();
      client.sendConfig?.();
    });

    await page.waitForFunction(() => window.__hudesClient?.state?.stepSize > 4, {
      timeout: 5000,
    });

    await page.waitForFunction(
      () => {
        const view = window.__hudesClient?.view?.impl;
        if (!view) return false;
        const lines = view.lineObjects || [];
        if (!lines.length) return false;
        return lines.every(
          (line) => line.geometry?.attributes?.position?.array?.length > 0,
        );
      },
      { timeout: 5000 },
    );

    const { maxValues, limit } = await page.evaluate(() => {
      const view = window.__hudesClient?.view?.impl;
      if (!view) {
        return { maxValues: [], limit: 0 };
      }
      const halfHeight = (view.gridSize * 0.6) / 2;
      const epsilon = 1e-3;
      const maxValues = view.lineObjects.map((line) => {
        const arr = line.geometry?.attributes?.position?.array || [];
        let maxAbs = 0;
        for (let i = 0; i < arr.length; i += 3) {
          const y = arr[i + 1];
          if (Number.isFinite(y)) {
            const abs = Math.abs(y);
            if (abs > maxAbs) {
              maxAbs = abs;
            }
          }
        }
        return maxAbs;
      });
      return { maxValues, limit: halfHeight + epsilon };
    });

    expect(maxValues.length).toBeGreaterThan(0);
    for (const value of maxValues) {
      expect(value).toBeLessThanOrEqual(limit);
    }
  });
});
