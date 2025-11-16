import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

async function trackErrors(page) {
  const errors = [];
  page.on('pageerror', (err) => errors.push(err));
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      errors.push(new Error(msg.text()));
    }
  });
  return errors;
}

async function waitForClient(page) {
  await page.waitForFunction(
    () => typeof window !== 'undefined' && window.__hudesClient && window.__hudesClient.ControlType,
    { timeout: 10000 },
  );
}

test.describe('Render modes', () => {
  test('3D landscape loads without console errors', async ({ page }) => {
    const errors = await trackErrors(page);
    await page.goto(`${APP_ORIGIN}/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off&mode=3d`);
    await waitForClient(page);
    await page.waitForTimeout(1000);

    const canvasCount = await page.locator('#glContainer canvas').count();
    expect(canvasCount).toBeGreaterThan(0);
    expect(errors).toHaveLength(0);
  });

  test('1D mode switches renderer without console errors', async ({ page }) => {
    const errors = await trackErrors(page);
    await page.goto(`${APP_ORIGIN}/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off&mode=1d`);
    await waitForClient(page);
    await page.waitForFunction(() => window.__hudesClient?.renderMode === '1d', { timeout: 10000 });
    await page.waitForSelector('#glContainer canvas', { timeout: 10000 });
    expect(await page.locator('#glContainer canvas').count()).toBeGreaterThan(0);
    expect(errors).toHaveLength(0);
  });
});
