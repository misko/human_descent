import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('1D alternate stacked view', () => {
  test('renders all loss lines inside a single enlarged plot', async ({ page }) => {
    await page.goto(
      `${APP_ORIGIN}/?mode=1d&alt1d=1&host=${SERVER_HOST}&port=${SERVER_PORT}&help=off&debug=1`,
    );

    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient.ControlType,
      { timeout: 10000 },
    );

    const closeButton = page.locator('.help-overlay__close');
    if (await closeButton.count()) {
      await closeButton.first().click();
    }

    await page.waitForFunction(
      () => {
        const view = window.__hudesClient?.view?.impl;
        return Boolean(view && view.lineObjects && view.lineObjects.length);
      },
      { timeout: 5000 },
    );

    const stats = await page.evaluate(() => {
      const client = window.__hudesClient;
      const view = client?.view?.impl;
      if (!client || !view) {
        return null;
      }
      const frames = Array.isArray(view.alt1dFrames) ? view.alt1dFrames : [];
      const widths = frames.map((frame) => {
        const arr = frame?.geometry?.attributes?.position?.array;
        if (!arr || arr.length < 6) {
          return null;
        }
        let minX = null;
        let maxX = null;
        for (let i = 0; i < arr.length; i += 3) {
          const x = arr[i];
          if (Number.isFinite(x)) {
            minX = minX == null ? x : Math.min(minX, x);
            maxX = maxX == null ? x : Math.max(maxX, x);
          }
        }
        return minX != null && maxX != null ? maxX - minX : null;
      });
      const positions =
        Array.isArray(view.alt1dContainers) && view.alt1dContainers.length
          ? view.alt1dContainers.map((container) =>
              container?.position
                ? { x: container.position.x, y: container.position.y, z: container.position.z }
                : null,
            )
          : [];
      return {
        alt1dMode: Boolean(view.alt1dMode),
        containerCount: Array.isArray(view.lineContainers) ? view.lineContainers.length : null,
        lineCount: Array.isArray(view.lineObjects) ? view.lineObjects.length : null,
        lossLines: client.lossLines,
        frameWidths: widths,
        expectedWidth: view.gridSize * 0.9 * 2,
        containerPositions: positions,
      };
    });

    expect(stats).not.toBeNull();
    expect(stats.alt1dMode).toBe(true);
    expect(stats.containerCount).toBe(2);
    expect(stats.lineCount).toBe(stats.lossLines);
    expect(stats.frameWidths).toHaveLength(2);
    for (const width of stats.frameWidths) {
      expect(width).not.toBeNull();
      expect(Math.abs(width - stats.expectedWidth)).toBeLessThanOrEqual(0.5);
    }
    expect(stats.containerPositions).toHaveLength(2);
    const [left, right] = stats.containerPositions;
    expect(left).not.toBeNull();
    expect(right).not.toBeNull();
    expect(left.x).toBeLessThan(0);
    expect(right.x).toBeGreaterThan(0);
    expect(Math.abs(left.y)).toBeLessThan(1e-3);
    expect(Math.abs(right.y)).toBeLessThan(1e-3);
    expect(Math.abs(left.z)).toBeLessThan(1e-3);
    expect(Math.abs(right.z)).toBeLessThan(1e-3);
  });
});
