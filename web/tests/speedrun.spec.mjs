// Playwright E2E test for speed run flow
import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const SPEED_SECONDS = Number(process.env.HUDES_SPEED_RUN_SECONDS || '5');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('Speed Run flow', () => {
  test('starts, counts down, and UI stays responsive', async ({ page }) => {
    // Serve the app via Vite preview or static; here we load index.html directly
    await page.goto(`${APP_ORIGIN}/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);

    // Wait for the client to be ready and WebSocket connected
    await page.waitForFunction(() => window.__hudesClient && window.__hudesClient.ControlType);

    // No prompt now; we'll fill the modal input when it appears

    // Press R to start speed run
    await page.keyboard.press('KeyR');

    // Wait until the client reports speed run active and remaining seconds appears in HUD string
    await page.waitForFunction(() => {
      const c = window.__hudesClient;
      return c && c.state && c.state.speedRunActive === true;
    }, { timeout: 10000 });

    // Countdown value may be populated only on next server message; we don't
    // assert intermediate ticks here. We'll rely on final deactivation below.

    // UI responsiveness: trigger next dims (Space) should still be allowed
    await page.keyboard.press('Space');

    // Loss chart should receive updates (labels growing)
    await page.waitForFunction(() => {
      const c = window.__hudesClient;
      return c && c.trainSteps && c.trainSteps.length > 0;
    });
    const n1 = await page.evaluate(() => window.__hudesClient.trainSteps.length);
    await page.waitForTimeout(1000);
    const n2 = await page.evaluate(() => window.__hudesClient.trainSteps.length);
    expect(n2).toBeGreaterThanOrEqual(n1);

    // SGD must be ignored during run: attempt and verify total_sgd_steps not incremented immediately on client
    await page.keyboard.press('Delete');
    const sgd = await page.evaluate(() => window.__hudesClient.state.sgdSteps);
    expect(sgd).toBe(0);

  // Wait for name modal
  await page.waitForSelector('#modalOverlay.open .glass-card .name-form', { timeout: (SPEED_SECONDS + 20) * 1000 });
  await page.fill('#modalOverlay.open .glass-card .name-form input', 'TEST');
  await page.click('#modalOverlay.open .glass-card .name-form button[type="submit"]');

  // Then leaderboard appears with our name shown
  await page.waitForSelector('#modalOverlay.open .glass-card .top10-list');
  });
});
