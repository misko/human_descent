import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
  },
  // Start both the Vite dev server (frontend) and the Python WebSocket backend
  // on a non-production port before running tests. Playwright will teardown
  // these processes automatically after the test run.
  webServer: [
    {
      command: 'npm run dev',
      url: 'http://localhost:5173',
      reuseExistingServer: true,
      timeout: 60_000,
    },
    {
      // Run from repo root to find the Python env and modules
      command:
        "bash -lc 'cd .. && PORT=10001; HPORT=$((PORT+1)); PIDS=$(lsof -ti :$PORT || true); if [ -n \"$PIDS\" ]; then echo Killing stale PIDs on $PORT: $PIDS; kill -9 $PIDS || true; sleep 0.5; fi; HPIDS=$(lsof -ti :$HPORT || true); if [ -n \"$HPIDS\" ]; then echo Killing stale PIDs on $HPORT: $HPIDS; kill -9 $HPIDS || true; sleep 0.5; fi; HUDES_SPEED_RUN_SECONDS=5 LOGLEVEL=DEBUG ./hudes_env/bin/python -u hudes/websocket_server.py --run-in thread --port $PORT'",
      // Wait on dedicated health server (PORT+1)
      url: 'http://localhost:10002/health',
      reuseExistingServer: false,
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
