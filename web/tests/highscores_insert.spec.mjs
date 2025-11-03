import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const SPEED_SECONDS = Number(process.env.HUDES_SPEED_RUN_SECONDS || '5');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

function makeName() {
  // Generate a unique 4-char uppercase A-Z0-9 name
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let s = '';
  for (let i = 0; i < 4; i++) s += alphabet[Math.floor(Math.random() * alphabet.length)];
  return s;
}

test.describe('Highscores API insertion', () => {
  test('adds a high score and retrieves it via API', async ({ page }) => {
    const name = makeName();
    const apiProto = 'http';
    const apiHost = SERVER_HOST;
    const apiPort = SERVER_PORT + 1; // API/health lives on ws+1
    const apiBase = `${apiProto}://${apiHost}:${apiPort}/api`;

    // Load app and start a speed run
    await page.goto(`${APP_ORIGIN}/?host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`);
    await page.waitForFunction(() => window.__hudesClient && window.__hudesClient.ControlType);
    await page.keyboard.press('KeyR');

    // Wait for the name modal and submit our unique name
    await page.waitForSelector('#modalOverlay.open .glass-card .name-form', { timeout: (SPEED_SECONDS + 20) * 1000 });
    await page.fill('#modalOverlay.open .glass-card .name-form input', name);
    await page.click('#modalOverlay.open .glass-card .name-form button[type="submit"]');

    // Leaderboard modal should appear
    await page.waitForSelector('#modalOverlay.open .glass-card .top10-list');

    // Poll the highscores API until our name appears or timeout
    const deadline = Date.now() + 15000; // up to 15s
    let found = false;
    let lastCount = 0;
    while (Date.now() < deadline && !found) {
      const resp = await page.request.get(`${apiBase}/highscores?offset=0&limit=10000`);
      expect(resp.ok()).toBeTruthy();
      const rows = await resp.json();
      lastCount = rows.length;
      found = rows.some(r => r.name === name);
      if (!found) await page.waitForTimeout(500);
    }

    expect(found, `Expected to find inserted name ${name} in highscores (last count=${lastCount})`).toBe(true);
  });
});
