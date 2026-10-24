import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createLobbyClient, resolveLobbyUrl } from '../lobby/client.js';

const flush = async () => { await Promise.resolve(); await Promise.resolve(); };

function harness({ saved, onLaunch } = {}) {
  let now = 0;
  let nextTimer = 0;
  const timers = new Map();
  const instances = [];
  const values = new Map();
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  const url = 'wss://example.test/lobby/ws';
  if (saved) values.set(`hudes.lobby.v1:${url}`, JSON.stringify(saved));
  class FakeSocket {
    readyState = 0;
    listeners = new Map();
    sent = [];
    constructor(endpoint) { this.url = endpoint; instances.push(this); }
    addEventListener(type, callback) {
      if (!this.listeners.has(type)) this.listeners.set(type, []);
      this.listeners.get(type).push(callback);
    }
    emit(type, event = {}) { for (const callback of this.listeners.get(type) || []) callback(event); }
    open() { this.readyState = 1; this.emit('open'); }
    send(data) { assert.equal(this.readyState, 1); this.sent.push(JSON.parse(data)); }
    receive(data) { this.emit('message', { data: typeof data === 'string' ? data : JSON.stringify(data) }); }
    close(reason = '') { if (this.readyState === 3) return; this.readyState = 3; this.emit('close', { reason }); }
    response(request, payload = {}, error) {
      this.receive({ v: 1, type: 'response', requestId: request.requestId, ok: !error, ...(error ? { error } : { payload }) });
    }
  }
  const client = createLobbyClient({
    wsUrl: '/lobby/ws', location: { href: 'https://example.test/?lobby=1' }, WebSocketImpl: FakeSocket, storage, onLaunch,
    setTimeout(callback, delay) { const timer = ++nextTimer; timers.set(timer, { at: now + delay, callback }); return timer; },
    clearTimeout(timer) { timers.delete(timer); }, reconnectDelayMs: 50, commandTimeoutMs: 1000,
  });
  async function identify({ name = 'Alice', playerId = 'alice', token = 'resume-private', socket } = {}) {
    if (!socket) { client.connect(name); socket = instances.at(-1); }
    socket.open();
    socket.response(socket.sent.at(-1), { playerId, resumeToken: token });
    await flush();
    return socket;
  }
  function advance(duration) {
    const target = now + duration;
    while (true) {
      const first = [...timers.entries()].filter(([, timer]) => timer.at <= target).sort((a, b) => a[1].at - b[1].at)[0];
      if (!first) break;
      timers.delete(first[0]);
      now = first[1].at;
      first[1].callback();
    }
    now = target;
  }
  return { client, instances, storage, values, timers, identify, advance };
}

function snapshot(teamId = null) {
  return {
    you: { id: 'alice', name: 'Alice', teamId },
    players: [{ id: 'alice', name: 'Alice', connected: true, teamId }],
    teams: teamId ? [{ id: teamId, name: 'Minima', captainId: 'alice', memberIds: ['alice'], revision: 1, status: 'available' }] : [],
    invitations: [], challenges: [], matches: [], chat: [],
  };
}
function sendSnapshot(socket, seq, teamId = null) { socket.receive({ v: 1, type: 'snapshot', seq, payload: snapshot(teamId) }); }
const launch = { matchId: 'match', teamId: 'team', playerId: 'alice', sessionId: 'game-session', panelIndex: 0, token: 'game-private', wsUrl: '/ws' };

test('uses a separate JSON endpoint and stores resume credentials only in per-tab storage', async () => {
  const h = harness();
  const socket = await h.identify();
  assert.equal(socket.url, 'wss://example.test/lobby/ws');
  assert.deepEqual(socket.sent[0].payload, { name: 'Alice' });
  assert.equal(socket.sent[0].v, 1);
  assert.equal(socket.sent[0].type, 'hello');
  assert.ok(socket.sent[0].requestId.length <= 80);
  assert.equal(h.client.state.connection, 'connected');
  assert.deepEqual(h.client.state.identity, { playerId: 'alice' });
  assert.equal(JSON.stringify(h.client.state).includes('resume-private'), false);
  assert.equal(JSON.parse(h.storage.getItem(h.client.storageKey)).resumeToken, 'resume-private');
  assert.equal(resolveLobbyUrl('/lobby/ws', { href: 'http://localhost:6173/?lobby=1' }), 'ws://localhost:6173/lobby/ws');
  assert.throws(() => resolveLobbyUrl('file:///lobby'));
  h.client.destroy();
});

test('correlates errors and waits for snapshots rather than inventing team membership', async () => {
  const h = harness();
  const socket = await h.identify();
  sendSnapshot(socket, 1);
  const command = h.client.command('team.create', { name: 'Minima' });
  const request = socket.sent.at(-1);
  assert.equal(h.client.state.pendingCount, 1);
  assert.equal(h.client.state.snapshot.you.teamId, null);
  socket.response(request, { teamId: 'team' });
  assert.deepEqual(await command, { teamId: 'team' });
  assert.equal(h.client.state.snapshot.you.teamId, null);
  sendSnapshot(socket, 2, 'team');
  assert.equal(h.client.state.snapshot.you.teamId, 'team');
  const denied = h.client.command('team.rename', { name: 'Denied' });
  const assertion = assert.rejects(denied, error => error.code === 'not_captain');
  socket.response(socket.sent.at(-1), {}, { code: 'not_captain', message: 'Only the captain can rename this team.' });
  await assertion;
  assert.equal(h.client.state.error.code, 'not_captain');
  assert.equal(h.client.state.snapshot.teams[0].name, 'Minima');
  assert.equal(h.client.state.pendingCount, 0);
  await assert.rejects(h.client.command('team.join', { teamId: 'team' }), /Unknown lobby action/);
  h.client.destroy();
});

test('reconnect resumes the identity, replays unresolved requests exactly, and adopts new sequence space', async () => {
  const h = harness();
  const first = await h.identify();
  sendSnapshot(first, 50);
  const payload = { name: 'Minima' };
  const result = h.client.command('team.create', payload);
  const original = first.sent.at(-1);
  payload.name = 'Caller mutation must not change replay';
  first.close();
  assert.equal(h.client.state.connection, 'reconnecting');
  assert.equal(h.client.state.pendingCount, 1);
  await assert.rejects(h.client.command('chat.send', { text: 'offline' }), error => error.code === 'disconnected');
  h.advance(50);
  const second = h.instances.at(-1);
  second.open();
  assert.equal(second.sent[0].payload.resumeToken, 'resume-private');
  second.response(second.sent[0], { playerId: 'alice', resumeToken: 'resume-private' });
  await flush();
  assert.deepEqual(second.sent[1], original);
  sendSnapshot(second, 1, 'team');
  assert.equal(h.client.state.snapshot.you.teamId, 'team');
  sendSnapshot(first, 51); // A late event from the replaced socket is inert.
  assert.equal(h.client.state.snapshot.you.teamId, 'team');
  second.response(second.sent[1], { teamId: 'team' });
  assert.deepEqual(await result, { teamId: 'team' });
  assert.equal(h.client.state.pendingCount, 0);
  h.client.destroy();
  assert.equal(h.timers.size, 0);
});

test('launch callback fires once per connection and live admissions are redelivered after reconnect', async () => {
  const received = [];
  const h = harness({ onLaunch: descriptor => received.push(descriptor) });
  const first = await h.identify();
  sendSnapshot(first, 1, 'team');
  first.receive({ v: 1, type: 'game.launch', seq: 2, payload: launch });
  first.receive({ v: 1, type: 'game.launch', seq: 3, payload: launch });
  assert.equal(received.length, 1);
  assert.deepEqual(received[0], launch);
  assert.equal(JSON.stringify(h.client.state).includes('game-private'), false);
  first.receive({ v: 1, type: 'game.launch', seq: 4, payload: { ...launch, sessionId: 'other', playerId: 'bob' } });
  assert.equal(received.length, 1);
  first.close();
  h.advance(50);
  const second = await h.identify({ socket: h.instances.at(-1) });
  sendSnapshot(second, 1, 'team');
  second.receive({ v: 1, type: 'game.launch', seq: 2, payload: launch });
  assert.equal(received.length, 2);
  h.client.destroy();
});

test('a rejected resume is explicit and cannot silently create a second identity', async () => {
  const h = harness({ saved: { name: 'Alice', playerId: 'alice', resumeToken: 'expired-private' } });
  h.client.connect();
  const first = h.instances[0];
  first.open();
  assert.equal(first.sent[0].payload.resumeToken, 'expired-private');
  first.response(first.sent[0], {}, { code: 'unauthorized', message: 'Invalid or expired resume token' });
  await flush();
  assert.equal(h.client.state.connection, 'identity-rejected');
  assert.equal(h.client.state.error.code, 'unauthorized');
  h.advance(20000);
  assert.equal(h.instances.length, 1);
  assert.equal(JSON.parse(h.storage.getItem(h.client.storageKey)).resumeToken, 'expired-private');
  h.client.resetIdentity();
  h.client.connect('Alice');
  const fresh = h.instances.at(-1);
  fresh.open();
  assert.deepEqual(fresh.sent[0].payload, { name: 'Alice' });
  fresh.response(fresh.sent[0], { playerId: 'alice-new', resumeToken: 'new-private' });
  await flush();
  assert.equal(h.client.state.identity.playerId, 'alice-new');
  h.client.destroy();
});

test('a session replaced elsewhere stops reconnecting instead of repeatedly stealing the connection', async () => {
  const h = harness();
  const socket = await h.identify();
  socket.close('replaced');
  assert.equal(h.client.state.connection, 'identity-rejected');
  assert.equal(h.client.state.error.code, 'session_replaced');
  h.advance(20000);
  assert.equal(h.instances.length, 1);
  h.client.destroy();
});

test('timeout and malformed transport data do not corrupt snapshots or leak private state', async () => {
  const h = harness();
  const socket = await h.identify();
  sendSnapshot(socket, 1);
  socket.receive('not JSON');
  assert.equal(h.client.state.error.code, 'invalid_response');
  assert.equal(h.client.state.snapshot.you.id, 'alice');
  const result = h.client.command('team.create', { name: 'Minima' });
  const timeout = assert.rejects(result, error => error.code === 'timeout');
  h.advance(1000);
  await timeout;
  assert.equal(h.client.state.pendingCount, 0);
  assert.equal(h.client.state.snapshot.you.teamId, null);
  h.client.destroy();
});

test('destroy is idempotent, rejects outstanding work and prevents reconnect or late notifications', async () => {
  const h = harness();
  const socket = await h.identify();
  let notifications = 0;
  h.client.subscribe(() => { notifications++; });
  const result = h.client.command('chat.send', { text: 'Hello' });
  const rejected = assert.rejects(result, error => error.code === 'destroyed');
  h.client.destroy();
  h.client.destroy();
  await rejected;
  const afterDestroy = notifications;
  sendSnapshot(socket, 1, 'team');
  socket.emit('open');
  h.advance(20000);
  assert.equal(h.instances.length, 1);
  assert.equal(notifications, afterDestroy);
  assert.equal(h.timers.size, 0);
  assert.equal(h.client.state.connection, 'closed');
});

test('blocked sessionStorage still allows an ephemeral connection', async () => {
  const h = harness();
  h.client.storage = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } };
  await h.identify();
  assert.equal(h.client.state.connection, 'connected');
  h.client.destroy();
});
