import { randomUUID } from 'node:crypto';
import { mkdtempSync, writeFileSync } from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

const unsafePorts = new Set([10000, 10001]);

async function reservePort(port = 0) {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });
  return server;
}

async function release(server) {
  await new Promise((resolve) => server.close(resolve));
}

async function freePortPair() {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    const first = await reservePort();
    const port = first.address().port;
    if (unsafePorts.has(port) || unsafePorts.has(port + 1) || port === 65535) {
      await release(first);
      continue;
    }
    let second;
    try {
      second = await reservePort(port + 1);
      return port;
    } catch (error) {
      if (error.code !== 'EADDRINUSE') throw error;
    } finally {
      if (second) await release(second);
      await release(first);
    }
  }
  throw new Error('Could not allocate isolated backend test ports');
}

export async function prepareTestRuntime() {
  if (!process.env.HUDES_TEST_RUN_DIR) {
    const directory = mkdtempSync(path.join(os.tmpdir(), 'hudes-playwright-'));
    const token = randomUUID();
    writeFileSync(path.join(directory, '.owned-test-run'), token, { flag: 'wx' });
    process.env.HUDES_TEST_RUN_DIR = directory;
    process.env.HUDES_TEST_RUN_TOKEN = token;
  }
  // Tests always own a fresh database. Never inherit a deployment score path.
  process.env.HIGH_SCORES_PATH = path.join(process.env.HUDES_TEST_RUN_DIR, 'high-scores.sqlite3');
  process.env.HUDES_PORT ??= String(await freePortPair());
  process.env.HUDES_HEALTH_PORT = String(Number(process.env.HUDES_PORT) + 1);
  for (const port of [process.env.HUDES_PORT, process.env.HUDES_HEALTH_PORT]) {
    if (!/^\d+$/.test(port) || unsafePorts.has(Number(port)) || Number(port) < 1024 || Number(port) > 65535) {
      throw new Error(`Refusing unsafe test port ${port}`);
    }
  }
  if (!process.env.HUDES_FRONTEND_PORT) {
    for (let attempt = 0; attempt < 30; attempt += 1) {
      const server = await reservePort();
      const port = server.address().port;
      await release(server);
      if (port !== Number(process.env.HUDES_PORT) && port !== Number(process.env.HUDES_HEALTH_PORT) && !unsafePorts.has(port)) {
        process.env.HUDES_FRONTEND_PORT = String(port);
        break;
      }
    }
  }
  const frontendPort = Number(process.env.HUDES_FRONTEND_PORT);
  if (!Number.isInteger(frontendPort) || frontendPort < 1024 || frontendPort > 65535 || unsafePorts.has(frontendPort)
      || frontendPort === Number(process.env.HUDES_PORT) || frontendPort === Number(process.env.HUDES_HEALTH_PORT)) {
    throw new Error(`Refusing unsafe or conflicting frontend test port ${process.env.HUDES_FRONTEND_PORT}`);
  }
  process.env.HUDES_TEST_PORTS = `${process.env.HUDES_PORT} ${process.env.HUDES_HEALTH_PORT}`;
  process.env.HUDES_ADMIN_TOKEN ??= randomUUID();
  process.env.HUDES_APP_ORIGIN = `http://127.0.0.1:${process.env.HUDES_FRONTEND_PORT}`;
  return {
    frontendPort: Number(process.env.HUDES_FRONTEND_PORT),
    backendPort: Number(process.env.HUDES_PORT),
    healthPort: Number(process.env.HUDES_HEALTH_PORT),
    origin: process.env.HUDES_APP_ORIGIN,
  };
}
