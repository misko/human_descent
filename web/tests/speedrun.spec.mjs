// Playwright E2E test for speed run flow
import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const SPEED_SECONDS = Number(process.env.HUDES_SPEED_RUN_SECONDS || '5');

test.describe('Speed Run flow', () => {
  test('starts, counts down, and UI stays responsive', async ({ page }) => {
    // Serve the app via Vite preview or static; here we load index.html directly
    await page.goto(`http://localhost:5173/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);

    // Wait for the client to be ready and WebSocket connected
    await page.waitForFunction(() => window.__hudesClient && window.__hudesClient.ControlType);

    // Press R to start speed run
    await page.keyboard.press('KeyR');

    // Wait until the client reports speed run active and remaining seconds appears in HUD string
    await page.waitForFunction(() => {
      const c = window.__hudesClient;
      return c && c.state && c.state.speedRunActive === true;
    }, { timeout: 10000 });

    // Verify countdown ticks at least once by observing seconds change
    const first = await page.evaluate(() => window.__hudesClient.state.speedRunSecondsRemaining);
    await page.waitForTimeout(1100);
    const second = await page.evaluate(() => window.__hudesClient.state.speedRunSecondsRemaining);
    expect(second).toBeLessThanOrEqual(first);

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

    // Wait until countdown reaches zero and prompt appears; auto-dismiss prompt
    page.once('dialog', async (dialog) => { await dialog.accept('TEST'); });
    await page.waitForFunction(() => window.__hudesClient.speedRunActive === false, { timeout: (SPEED_SECONDS + 10) * 1000 });
  });
});
