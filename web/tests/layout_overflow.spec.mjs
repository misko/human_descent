import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

async function waitForClient(page) {
  await page.waitForFunction(
    () => typeof window !== 'undefined' && window.__hudesClient && window.__hudesClient.ControlType,
    { timeout: 10000 },
  );
}

async function waitForOneDScene(page) {
  await waitForClient(page);
  await page.waitForFunction(() => window.__hudesClient?.renderMode === '1d', { timeout: 10000 });
  await page.waitForSelector('#glContainer canvas', { timeout: 10000 });
}

async function expectNoPageOverflow(page, tolerancePx = 1) {
  const overflow = await page.evaluate(() => {
    const doc = document.scrollingElement || document.documentElement;
    return {
      horizontal: Math.max(0, doc.scrollWidth - window.innerWidth),
      vertical: Math.max(0, doc.scrollHeight - window.innerHeight),
    };
  });

  expect(overflow.horizontal).toBeLessThanOrEqual(tolerancePx);
  expect(overflow.vertical).toBeLessThanOrEqual(tolerancePx);
}

test.describe('Layout overflow guard', () => {
  test('3D mode fits within the viewport', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off&mode=3d`);
    await waitForClient(page);
    await page.waitForSelector('#glContainer canvas', { timeout: 10000 });
    await expectNoPageOverflow(page);
  });

  test('1D mode fits within the viewport', async ({ page }) => {
    await page.goto(`${APP_ORIGIN}/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off&mode=1d`);
    await waitForOneDScene(page);
    await expectNoPageOverflow(page);
  });
});
