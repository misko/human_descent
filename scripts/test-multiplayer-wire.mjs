import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import protobuf from '../web/node_modules/protobufjs/index.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const fixtures = JSON.parse(await readFile(path.join(root, 'tests/fixtures/multiplayer/game-v1.json'), 'utf8'));
const schema = await protobuf.load(path.join(root, 'hudes/hudes.proto'));
const Control = schema.lookupType('Control');
const encoded = {};
for (const [name, fixture] of Object.entries(fixtures)) {
  const message = Control.fromObject(fixture);
  assert.equal(Control.verify(message), null, `invalid fixture ${name}`);
  encoded[name] = Buffer.from(Control.encode(message).finish()).toString('base64');
}

const python = spawnSync('bash', [
  path.join(root, 'scripts/test-python.sh'),
  '/app/tests/helpers/multiplayer_wire.py',
  JSON.stringify({ fixtures, encoded }),
], { encoding: 'utf8', env: process.env });
if (python.status !== 0) {
  process.stderr.write(python.stdout || '');
  process.stderr.write(python.stderr || '');
  process.exit(python.status || 1);
}
const returned = JSON.parse(python.stdout.trim());

function compare(actual, expected, location) {
  if (typeof expected === 'number') {
    assert.ok(Math.abs(Number(actual) - expected) < Math.max(1e-6, Math.abs(expected) * 1e-7), `${location}: ${actual} != ${expected}`);
  } else if (expected && typeof expected === 'object') {
    for (const [key, value] of Object.entries(expected)) compare(actual?.[key], value, `${location}.${key}`);
  } else {
    assert.equal(actual, expected, location);
  }
}
for (const [name, fixture] of Object.entries(fixtures)) {
  const message = Control.decode(Buffer.from(returned.encoded[name], 'base64'));
  const object = Control.toObject(message, { longs: Number, enums: Number });
  compare(object, fixture, name);
}
console.log(`Gameplay wire compatibility passed in both directions (${Object.keys(fixtures).length} canonical fixtures).`);
