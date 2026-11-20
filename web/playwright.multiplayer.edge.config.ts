import { defineConfig, devices } from '@playwright/test';
import { prepareTestRuntime } from './tests/helpers/test-runtime.mjs';

function requiredOrigin(name: string) {
  const url = new URL(process.env[name] ?? '');
  if (url.protocol !== 'https:' || url.href !== `${url.origin}/`) {
    throw new Error(`${name} must be an HTTPS origin`);
  }
  return url;
}

const primary = requiredOrigin('HUDES_EDGE_ORIGIN');
const alias = requiredOrigin('HUDES_EDGE_ALIAS');
if (alias.hostname === primary.hostname) throw new Error('Use distinct staging hostnames');
await prepareTestRuntime();
process.env.HUDES_APP_ORIGIN = primary.origin;
process.env.HUDES_EDGE_ALIAS = alias.origin;

const port = Number(process.env.HUDES_EDGE_LOOPBACK_PORT ?? 0);
if (process.env.HUDES_EDGE_LOOPBACK_PORT && (!Number.isInteger(port) || port < 1024 || port > 65535)) {
  throw new Error('Use a valid, unprivileged staging loopback port');
}
const resolver = [primary, alias].map(url => `MAP ${url.hostname} 127.0.0.1:${port}`);

export default defineConfig({
  testDir: './tests/staging',
  fullyParallel: false,
  workers: 1,
  timeout: 120_000,
  reporter: 'list',
  globalTeardown: './tests/global-teardown.mjs',
  use: {
    ...devices['Desktop Chrome'],
    baseURL: primary.origin,
    launchOptions: {
      args: ['--enable-unsafe-swiftshader', ...(port ? ['--no-proxy-server', `--host-resolver-rules=${resolver.join(', ')}`] : [])],
    },
    screenshot: 'only-on-failure',
  },
});
