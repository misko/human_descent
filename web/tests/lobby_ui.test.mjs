import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mountLobby } from '../lobby/index.js';

// A small DOM fixture exercises actual delegated UI events without adding a
// browser dependency to the generation-free unit runner. Real browser layout
// and two-socket composition are covered by the multiplayer integration suite.
class TextNode {
  constructor(text, doc) { this.textContent = text; this.ownerDocument = doc; this.parentNode = null; }
}
class Node {
  constructor(tag, doc) {
    this.tagName = tag.toUpperCase(); this.ownerDocument = doc; this.parentNode = null;
    this.children = []; this.attrs = new Map(); this.listeners = new Map(); this.disabled = false; this.hidden = false;
    this.value = ''; this.scrollHeight = 0; this.scrollTop = 0; this.clientHeight = 100;
    this.dataset = new Proxy({}, {
      get: (_, key) => this.getAttribute(`data-${key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`) ?? undefined,
      set: (_, key, value) => { this.setAttribute(`data-${key.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`, value); return true; },
    });
  }
  set textContent(text) { this.replaceChildren(new TextNode(String(text), this.ownerDocument)); }
  get textContent() { return this.children.map(n => n.textContent).join(''); }
  setAttribute(key, value) { this.attrs.set(key, String(value)); if (key === 'value') this.value = String(value); }
  getAttribute(key) { return this.attrs.get(key) ?? null; }
  append(...nodes) { for (const node of nodes) { node.parentNode = this; this.children.push(node); } }
  replaceChildren(...nodes) { for (const node of this.children) node.parentNode = null; this.children = []; this.append(...nodes); }
  contains(node) { return this === node || this.children.some(c => c instanceof Node && c.contains(node)); }
  matches(selector) {
    const match = selector.match(/^([a-z]+)?(?:\[([^=\]]+)(?:="([^\"]*)")?\])?$/i);
    if (!match) throw new Error(`Unsupported test selector: ${selector}`);
    return (!match[1] || this.tagName === match[1].toUpperCase()) && (!match[2] || (this.attrs.has(match[2]) && (match[3] === undefined || this.getAttribute(match[2]) === match[3])));
  }
  closest(selector) { let node = this; while (node) { if (node.matches(selector)) return node; node = node.parentNode; } return null; }
  querySelectorAll(selector) { return this.children.flatMap(c => c instanceof Node ? [...(c.matches(selector) ? [c] : []), ...c.querySelectorAll(selector)] : []); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  addEventListener(type, callback) { if (!this.listeners.has(type)) this.listeners.set(type, new Set()); this.listeners.get(type).add(callback); }
  removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
  dispatch(type, extras = {}) {
    const event = { type, target: this, stopped: false, prevented: false,
      stopPropagation() { this.stopped = true; }, preventDefault() { this.prevented = true; }, ...extras };
    let node = this;
    while (node) {
      for (const callback of node.listeners.get(type) || []) callback(event);
      if (event.stopped) break;
      node = node.parentNode;
    }
    return event;
  }
  focus() { this.ownerDocument.activeElement = this; }
  get form() { return this.closest('form'); }
  requestSubmit() { this.dispatch('submit'); }
  remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(c => c !== this); this.parentNode = null; }
}

async function mounted() {
  const previous = Object.fromEntries(['WebSocket', 'sessionStorage', 'location'].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const sockets = [];
  const stored = new Map();
  class Socket {
    readyState = 0; callbacks = new Map(); sent = [];
    constructor() { sockets.push(this); }
    addEventListener(type, cb) { this.callbacks.set(type, cb); }
    send(data) { this.sent.push(JSON.parse(data)); }
    receive(envelope) { this.callbacks.get('message')?.({ data: JSON.stringify(envelope) }); }
    close() { this.readyState = 3; this.callbacks.get('close')?.({ reason: '' }); }
    reply(payload = {}) { this.receive({ v: 1, type: 'response', requestId: this.sent.at(-1).requestId, ok: true, payload }); }
  }
  for (const [key, value] of Object.entries({ WebSocket: Socket, location: { href: 'https://example.test/?lobby=1' }, sessionStorage: { getItem: key => stored.get(key), setItem: (key, value) => stored.set(key, value) } })) {
    Object.defineProperty(globalThis, key, { configurable: true, value });
  }
  const doc = { activeElement: null, createElement(tag) { return new Node(tag, this); }, createTextNode(text) { return new TextNode(text, this); } };
  const root = new Node('main', doc);
  const launches = [];
  const mounted = mountLobby({ root, wsUrl: '/lobby/ws', onLaunch: d => launches.push(d) });
  const find = id => root.querySelectorAll('[data-testid]').find(n => n.getAttribute('data-testid') === id);
  find('lobby-name').value = 'Alice';
  find('lobby-name').form.dispatch('submit');
  const socket = sockets[0];
  socket.readyState = 1;
  socket.callbacks.get('open')();
  socket.reply({ playerId: 'alice', resumeToken: 'private' });
  await Promise.resolve(); await Promise.resolve();
  return {
    root, doc, find, socket, launches, client: mounted.client,
    snapshot(payload, seq = 1) { socket.receive({ v: 1, type: 'snapshot', seq, payload }); },
    cleanup() {
      mounted.destroy(); mounted.destroy();
      for (const [key, descriptor] of Object.entries(previous)) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
      }
    },
  };
}

function fixture() {
  return {
    you: { id: 'alice', name: 'Alice', teamId: 'one' },
    players: [{ id: 'alice', name: 'Alice', connected: true, teamId: 'one' }, { id: 'bob', name: 'Bob', connected: true, teamId: 'other' }, { id: 'eve', name: '<img src=x onerror=alert(1)>', connected: true, teamId: null }],
    teams: [{ id: 'one', name: 'Minima', captainId: 'alice', memberIds: ['alice'], revision: 1, status: 'busy' }, { id: 'other', name: 'Gradients', captainId: 'bob', memberIds: ['bob'], revision: 1, status: 'busy' }],
    invitations: [], challenges: [],
    matches: [{ id: 'match', teamIds: ['one', 'other'], phase: 'ready', readyPlayerIds: [], startAt: null, deadline: null, results: null, reason: null }],
    chat: [{ id: 'chat', playerId: 'eve', name: '<img src=x onerror=alert(1)>', text: '<script>alert(1)</script>', sentAt: 1780000000 }],
  };
}

test('ready and not-ready clicks send literal true and false through the delegated DOM path', async () => {
  const h = await mounted();
  try {
    const state = fixture();
    h.snapshot(state);
    assert.equal(h.find('lobby-ready').getAttribute('data-ready'), 'true');
    h.find('lobby-ready').dispatch('click');
    assert.equal(h.socket.sent.at(-1).type, 'match.ready');
    assert.deepEqual(h.socket.sent.at(-1).payload, { matchId: 'match', ready: true });
    h.socket.reply({ matchId: 'match' });
    state.matches[0].readyPlayerIds = ['alice'];
    h.snapshot(state, 2);
    assert.equal(h.find('lobby-ready-count').textContent, '1/2 ready');
    assert.equal(h.find('lobby-ready').getAttribute('data-ready'), 'false');
    h.find('lobby-ready').dispatch('click');
    assert.deepEqual(h.socket.sent.at(-1).payload, { matchId: 'match', ready: false });
    h.socket.reply({ matchId: 'match' });
  } finally { h.cleanup(); }
});

test('chat uses text nodes, keeps native editing and stops gameplay keyboard propagation', async () => {
  const h = await mounted();
  try {
    h.snapshot(fixture());
    assert.equal(h.root.querySelectorAll('script').length, 0);
    assert.equal(h.root.querySelectorAll('img').length, 0);
    assert.ok(h.find('lobby-chat-log').textContent.includes('<script>alert(1)</script>'));
    assert.ok(h.find('lobby-presence').textContent.includes('<img src=x onerror=alert(1)>'));
    const input = h.find('lobby-chat-input');
    const space = input.dispatch('keydown', { key: ' ' });
    assert.equal(space.stopped, true);
    assert.equal(space.prevented, false);
    input.value = 'Hello';
    const newline = input.dispatch('keydown', { key: 'Enter', shiftKey: true });
    assert.equal(newline.prevented, false);
    const send = input.dispatch('keydown', { key: 'Enter', shiftKey: false, repeat: false });
    assert.equal(send.prevented, true);
    assert.deepEqual(h.socket.sent.at(-1).payload, { text: 'Hello' });
    h.socket.reply();
    await Promise.resolve(); await Promise.resolve();
    assert.equal(input.value, '');
  } finally { h.cleanup(); }
  assert.equal(h.root.children.length, 0);
});

test('captain-only invitations and arbitrary-sized challenges are available without public join controls', async () => {
  const h = await mounted();
  try {
    const state = fixture();
    state.matches = [];
    state.teams.forEach(t => { t.status = 'available'; });
    state.teams[1].memberIds.push('carol', 'dan');
    state.players.push({ id: 'carol', name: 'Carol', connected: true, teamId: 'other' }, { id: 'dan', name: 'Dan', connected: true, teamId: 'other' });
    h.snapshot(state);
    assert.equal(h.find('lobby-challenge-other').textContent, 'Challenge · 1 vs 3');
    assert.ok(h.find('lobby-invite-eve'));
    h.find('lobby-invite-eve').dispatch('click');
    assert.deepEqual(h.socket.sent.at(-1).payload, { playerId: 'eve' });
    h.socket.reply({ inviteId: 'invite' });
    assert.equal(h.root.querySelectorAll('[data-action]').some(n => n.dataset.action === 'team.join'), false);
    state.teams[0].captainId = 'bob';
    h.snapshot(state, 2);
    assert.equal(h.find('lobby-invite-eve'), undefined);
    assert.equal(h.find('lobby-challenge-other'), undefined);
    assert.equal(h.find('lobby-start-speedrun'), undefined);
    assert.equal(h.find('lobby-rename-team'), undefined);
    assert.equal(h.find('lobby-leave-team').textContent, 'Leave team');
  } finally { h.cleanup(); }
});
