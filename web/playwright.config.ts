import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';

process.env.HUDES_PORT ??= '18100';
process.env.HIGH_SCORES_PATH ??= path.resolve(import.meta.dirname, '../dev-data/playwright-high-scores.sqlite3');
process.env.HUDES_ADMIN_TOKEN ??= 'isolated-playwright-admin';
process.env.HUDES_HEALTH_PORT ??= String(Number(process.env.HUDES_PORT) + 1);
const DEFAULT_SPEED_RUN_SECONDS = process.env.HUDES_SPEED_RUN_SECONDS ?? '5';
process.env.HUDES_SPEED_RUN_SECONDS = DEFAULT_SPEED_RUN_SECONDS;

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  reporter: [['list']],
  globalTeardown: './tests/global-teardown.mjs',
  use: {
    baseURL: 'http://localhost:6173',
    trace: 'on-first-retry',
  },
  // Start both the Vite dev server (frontend) and the Python WebSocket backend
  // on a non-production port before running tests. Playwright will teardown
  // these processes automatically after the test run.
  webServer: [
    {
      command: 'npm run dev',
      url: 'http://localhost:6173',
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      // Run from repo root to find the Python env and modules
      command: process.env.HUDES_BACKEND_COMMAND || 'cd .. && ${HUDES_PYTHON:-./dev-data/test-venv/bin/python} -u -m hudes.websocket_server --run-in thread --port "$HUDES_PORT" --device ${HUDES_TEST_DEVICE:-cpu} --model cnn3',
      url: `http://localhost:${process.env.HUDES_HEALTH_PORT}/health`,
      reuseExistingServer: false,
      env: {
        ...process.env,
        HUDES_SPEED_RUN_SECONDS: DEFAULT_SPEED_RUN_SECONDS,
      },
      timeout: 120_000,
    },
  ],
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
