#!/usr/bin/env node
import { readdir } from 'fs/promises';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function run() {
  const entries = await readdir(__dirname);
  const tests = entries
    .filter((f) => f.endsWith('.test.mjs'))
    // exclude playwright e2e specs if they also match pattern in the future
    .filter((f) => !f.endsWith('.spec.mjs'))
    .sort();

  if (tests.length === 0) {
    console.log('[unit] No *.test.mjs files found');
    return;
  }

  let failed = 0;
  for (const f of tests) {
    const full = join(__dirname, f);
    console.log(`[unit] Running ${f}`);
    const code = await new Promise((resolve) => {
      const child = spawn(process.execPath, [full], {
        stdio: 'inherit',
        env: process.env,
      });
      child.on('exit', (code) => resolve(code ?? 1));
    });
    if (code !== 0) {
      console.error(`[unit] ${f} failed with exit code ${code}`);
      failed++;
    }
  }

  if (failed > 0) {
    console.error(`[unit] ${failed} test file(s) failed.`);
    process.exit(1);
  } else {
    console.log('[unit] All unit tests passed.');
  }
}

run().catch((err) => {
  console.error('[unit] Runner error:', err);
  process.exit(1);
});
