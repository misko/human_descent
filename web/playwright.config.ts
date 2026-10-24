import { defineConfig, devices } from '@playwright/test';
import { prepareTestRuntime } from './tests/helpers/test-runtime.mjs';

const runtime = await prepareTestRuntime();
const DEFAULT_SPEED_RUN_SECONDS = process.env.HUDES_SPEED_RUN_SECONDS ?? '5';
process.env.HUDES_SPEED_RUN_SECONDS = DEFAULT_SPEED_RUN_SECONDS;

export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.mjs',
  testIgnore: '**/multiplayer/**',
  // The CPU inference worker is shared by this suite; avoid queuing many
  // expensive full landscapes from otherwise independent browser tests.
  fullyParallel: false,
  workers: 1,
  timeout: Math.max(60_000, (Number(DEFAULT_SPEED_RUN_SECONDS) + 45) * 1000),
  reporter: [['list']],
  globalTeardown: './tests/global-teardown.mjs',
  use: {
    baseURL: runtime.origin,
    trace: 'on-first-retry',
  },
  // Start both the Vite dev server (frontend) and the Python WebSocket backend
  // on a non-production port before running tests. Playwright will teardown
  // these processes automatically after the test run.
  webServer: [
    {
      // Generation is an explicit build step; never mutate protobuf from a test server.
      command: `npx vite --host 127.0.0.1 --port ${runtime.frontendPort} --strictPort`,
      url: runtime.origin,
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      // Run from repo root to find the Python env and modules
      command: process.env.HUDES_BACKEND_COMMAND || 'bash ../scripts/test-python.sh -u -m hudes.websocket_server --run-in thread --port "$HUDES_PORT" --device ${HUDES_TEST_DEVICE:-cpu} --model ${HUDES_TEST_MODEL:-ffnn}',
      url: `http://127.0.0.1:${runtime.healthPort}/health`,
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
