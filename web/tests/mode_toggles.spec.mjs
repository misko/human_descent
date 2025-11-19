import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

async function trackErrors(page) {
  const errors = [];
  page.on('pageerror', (err) => errors.push(err));
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      const text = msg.text();
      if (!text.includes('plotly-latest.min.js') && !text.includes('plotly-latest.js')) {
        errors.push(new Error(text));
      }
    }
  });
  return errors;
}

async function waitForClient(page) {
  await page.waitForFunction(
    () =>
      typeof window !== 'undefined' &&
      window.__hudesClient &&
      window.__hudesClient.ControlType,
    { timeout: 10000 },
  );
}

async function expectGlScene(page) {
  await waitForClient(page);
  await page.waitForSelector('#glContainer canvas', { timeout: 10000 });
  const canvasCount = await page.locator('#glContainer canvas').count();
  expect(canvasCount).toBeGreaterThan(0);
}

async function expectOneDScene(page) {
  await waitForClient(page);
  await page.waitForFunction(() => window.__hudesClient?.renderMode === '1d', { timeout: 10000 });
  await page.waitForSelector('#glContainer canvas', { timeout: 10000 });
  const canvasCount = await page.locator('#glContainer canvas').count();
  expect(canvasCount).toBeGreaterThan(0);
}

test.describe('Top bar mode toggles', () => {
  test('switch between 3D / 1D / 1D alt back to 3D via header buttons', async ({ page }) => {
    const errors = await trackErrors(page);
    await page.addInitScript(() => {
      try {
        window.sessionStorage?.setItem('hudes.help.dismissed', '1');
      } catch {}
    });
    await page.goto(`${APP_ORIGIN}/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off&mode=3d`);
    await expectGlScene(page);

    // Switch to 1D view
    await Promise.all([
      page.waitForURL(/mode=1d/i),
      page.getByRole('button', { name: /Switch to 1D view/i }).click(),
    ]);
    await expectOneDScene(page);

    // Enable Alt 1D
    await Promise.all([
      page.waitForURL(/alt1d=1/i),
      page.getByRole('button', { name: /Alt 1D:/i }).click(),
    ]);
    await page.waitForFunction(() => window.__hudesClient?.alt1d === true, { timeout: 10000 });
    await expectOneDScene(page);

    // Return to 3D view
    await Promise.all([
      page.waitForURL((url) => new URL(url).searchParams.get('mode') === '3d'),
      page.getByRole('button', { name: /Switch to 3D view/i }).click(),
    ]);
    await page.waitForFunction(() => window.__hudesClient?.renderMode === '3d', { timeout: 10000 });
    await expectGlScene(page);

    expect(errors).toHaveLength(0);
  });
});
