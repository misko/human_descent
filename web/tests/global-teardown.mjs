import { readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

export default async function globalTeardown() {
  const directory = process.env.HUDES_TEST_RUN_DIR;
  const token = process.env.HUDES_TEST_RUN_TOKEN;
  if (!directory || !token) return;
  const resolved = path.resolve(directory);
  if (path.dirname(resolved) !== os.tmpdir() || !path.basename(resolved).startsWith('hudes-playwright-')) {
    throw new Error('Refusing to remove a directory outside the owned Playwright test run');
  }
  const marker = await readFile(path.join(resolved, '.owned-test-run'), 'utf8');
  if (marker !== token) throw new Error('Playwright test directory ownership marker does not match');
  await rm(resolved, { recursive: true });
}
