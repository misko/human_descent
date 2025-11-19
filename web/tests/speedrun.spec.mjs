// Playwright E2E test for speed run flow
import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const SPEED_SECONDS = Number(process.env.HUDES_SPEED_RUN_SECONDS ?? '120');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

async function runSpeedRunScenario(page, { extraQuery = '', name = 'TEST' } = {}) {
  const search = `host=${SERVER_HOST}&port=${SERVER_PORT}&help=off${extraQuery}`;
  await page.goto(`${APP_ORIGIN}/?${search}`);

  await page.waitForFunction(
    () => window.__hudesClient && window.__hudesClient.ControlType,
  );

  await page.keyboard.press('KeyZ');

  await page.waitForFunction(
    () => {
      const c = window.__hudesClient;
      return c && c.state && c.state.speedRunActive === true;
    },
    { timeout: 10000 },
  );

  await page.keyboard.press('Space');

  await page.waitForFunction(() => {
    const c = window.__hudesClient;
    return c && c.trainSteps && c.trainSteps.length > 0;
  });
  const n1 = await page.evaluate(() => window.__hudesClient.trainSteps.length);
  await page.waitForTimeout(1000);
  const n2 = await page.evaluate(() => window.__hudesClient.trainSteps.length);
  expect(n2).toBeGreaterThanOrEqual(n1);

  await page.keyboard.press('Delete');
  const sgd = await page.evaluate(() => window.__hudesClient.state.sgdSteps);
  expect(sgd).toBe(0);

  await page.waitForSelector(
    '#modalOverlay.open .glass-card .name-form',
    { timeout: (SPEED_SECONDS + 40) * 1000 },
  );
  await page.fill('#modalOverlay.open .glass-card .name-form input', name);
  await page.click('#modalOverlay.open .glass-card .name-form button[type="submit"]');

  await page.waitForSelector('#modalOverlay.open .glass-card .top10-list');
}

test.describe('Speed Run flow', () => {
  test('3D view: starts, counts down, and UI stays responsive', async ({ page }) => {
    await runSpeedRunScenario(page, { name: 'TEST' });
  });

  test('1D view: starts, counts down, and UI stays responsive', async ({ page }) => {
    await runSpeedRunScenario(page, { extraQuery: '&mode=1d', name: 'T1DM' });
  });
});
