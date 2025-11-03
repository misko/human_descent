import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

test.describe('1D keyboard controls (alt keys)', () => {
  test('alt key pairs drive the correct dimensions', async ({ page }) => {
    await page.goto(
      `${APP_ORIGIN}/?mode=1d&host=${SERVER_HOST}&port=${SERVER_PORT}&help=off&altkeys=1&debug=1`,
    );

    await page.waitForFunction(
      () => window.__hudesClient && window.__hudesClient.ControlType,
      { timeout: 10000 },
    );

    const mapping = await page.evaluate(() => {
      const client = window.__hudesClient;
      if (!client) return null;
      return client.keyToParamAndSign;
    });

    expect(mapping).not.toBeNull();

    const expectations = [
      ['u', 0, 1],
      ['w', 0, -1],
      ['i', 1, 1],
      ['e', 1, -1],
      ['o', 2, 1],
      ['r', 2, -1],
      ['j', 3, 1],
      ['s', 3, -1],
      ['k', 4, 1],
      ['d', 4, -1],
      ['l', 5, 1],
      ['f', 5, -1],
    ];

    for (const [key, dim, sign] of expectations) {
      expect(mapping[key]).toBeDefined();
      expect(mapping[key].dim).toBe(dim);
      expect(mapping[key].sign).toBe(sign);
    }
  });
});
