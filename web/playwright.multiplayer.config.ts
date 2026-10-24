import { defineConfig, devices } from '@playwright/test';
import { prepareTestRuntime } from './tests/helpers/test-runtime.mjs';

const runtime = await prepareTestRuntime();

export default defineConfig({
  testDir: './tests/multiplayer',
  testMatch: '**/*.spec.mjs',
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  reporter: [['list']],
  globalTeardown: './tests/global-teardown.mjs',
  use: {
    baseURL: runtime.origin,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: [
    {
      command: `npx vite --host 127.0.0.1 --port ${runtime.frontendPort} --strictPort`,
      url: runtime.origin,
      reuseExistingServer: false,
    },
    {
      command: 'bash ../scripts/test-python.sh -u -m scripts.multiplayer_test_server --port "$HUDES_PORT" --health-port "$HUDES_HEALTH_PORT" --duration ${HUDES_MULTIPLAYER_TEST_SECONDS:-6}',
      url: `http://127.0.0.1:${runtime.healthPort}/health`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
});
