import { createLobbyStore, normalizeLaunch } from './state.js';

const COMMANDS = new Set([
  'snapshot', 'chat.send', 'team.create', 'team.rename', 'team.invite', 'team.leave',
  'invite.accept', 'invite.decline', 'team.start_speedrun', 'challenge.create',
  'challenge.accept', 'challenge.decline', 'challenge.cancel', 'match.ready', 'match.cancel',
]);

export class LobbyError extends Error {
  constructor(code, message) { super(message); this.name = 'LobbyError'; this.code = code; }
}

function defaultStorage() {
  try { return globalThis.sessionStorage; } catch { return null; }
}

export function resolveLobbyUrl(value, location = globalThis.location) {
  const base = location?.href || 'http://localhost/';
  const url = new URL(value || '/lobby/ws', base);
  if (url.protocol === 'https:') url.protocol = 'wss:';
  if (url.protocol === 'http:') url.protocol = 'ws:';
  if (!['ws:', 'wss:'].includes(url.protocol)) throw new LobbyError('invalid_request', 'Invalid lobby endpoint.');
  return url.href;
}

export class LobbyClient {
  #socket = null;
  #pending = new Map();
  #identity = null;
  #name = '';
  #desired = false;
  #destroyed = false;
  #reconnectTimer = null;
  #attempts = 0;
  #counter = 0;
  #helloId = null;
  #launchSubscribers = new Set();
  #launchedSessions = new Set();
  #prefix;

  constructor({ wsUrl = '/lobby/ws', WebSocketImpl = globalThis.WebSocket, storage = defaultStorage(), location = globalThis.location,
    setTimeout: schedule = globalThis.setTimeout.bind(globalThis), clearTimeout: unschedule = globalThis.clearTimeout.bind(globalThis),
    reconnectDelayMs = 700, maxReconnectDelayMs = 10000, commandTimeoutMs = 30000, onLaunch } = {}) {
    this.url = resolveLobbyUrl(wsUrl, location);
    this.WebSocketImpl = WebSocketImpl;
    this.storage = storage;
    this.storageKey = `hudes.lobby.v1:${this.url}`;
    this.schedule = schedule;
    this.unschedule = unschedule;
    this.reconnectDelayMs = reconnectDelayMs;
    this.maxReconnectDelayMs = maxReconnectDelayMs;
    this.commandTimeoutMs = commandTimeoutMs;
    this.store = createLobbyStore();
    this.#prefix = globalThis.crypto?.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    try {
      const saved = JSON.parse(storage?.getItem(this.storageKey) || 'null');
      if (typeof saved?.name === 'string') this.#name = saved.name.slice(0, 48);
      if (typeof saved?.resumeToken === 'string' && saved.resumeToken.length <= 8192 && typeof saved?.playerId === 'string') {
        this.#identity = { playerId: saved.playerId, resumeToken: saved.resumeToken };
      }
    } catch { /* Storage restrictions must not prevent a fresh lobby session. */ }
    if (onLaunch) this.subscribeLaunch(onLaunch);
  }

  get state() { return this.store.state; }
  get savedName() { return this.#name; }
  subscribe(listener) { return this.store.subscribe(listener); }
  subscribeLaunch(listener) { this.#launchSubscribers.add(listener); return () => this.#launchSubscribers.delete(listener); }

  #persist() {
    try { this.storage?.setItem(this.storageKey, JSON.stringify({ name: this.#name, ...this.#identity })); } catch { /* Private browsing may disable storage. */ }
  }

  connect(name = this.#name) {
    if (this.#destroyed) return;
    const normalized = typeof name === 'string' ? name.trim() : '';
    if (!normalized || normalized.length > 48) {
      this.store.patch({ error: { code: 'invalid_request', message: 'Choose a display name of 1–48 characters.' } });
      return;
    }
    if (['connecting', 'identifying', 'connected', 'reconnecting'].includes(this.state.connection) && this.#desired) return;
    this.#name = normalized;
    this.#persist();
    this.#desired = true;
    this.#attempts = 0;
    this.#open(false);
  }

  #open(reconnecting) {
    if (!this.#desired || this.#destroyed) return;
    if (this.#reconnectTimer != null) this.unschedule(this.#reconnectTimer);
    this.#reconnectTimer = null;
    // The server redelivers live admissions on reconnect. Composition owns
    // active-game deduplication; retrying a failed/disposed mount must work.
    this.#launchedSessions.clear();
    this.store.beginConnection(reconnecting ? 'reconnecting' : 'connecting');
    this.store.patch({ error: null });
    let socket;
    try {
      socket = new this.WebSocketImpl(this.url);
    } catch {
      this.#scheduleReconnect();
      return;
    }
    this.#socket = socket;
    socket.addEventListener('open', () => {
      if (this.#socket !== socket || this.#destroyed) return;
      this.store.patch({ connection: 'identifying' });
      const payload = { name: this.#name };
      if (this.#identity) payload.resumeToken = this.#identity.resumeToken;
      const id = this.#requestId();
      this.#helloId = id;
      this.#send('hello', payload, id, true).then(identity => {
        if (this.#socket !== socket || this.#destroyed || !this.#desired) return;
        if (typeof identity?.playerId !== 'string' || !identity.playerId || typeof identity?.resumeToken !== 'string' || !identity.resumeToken || identity.resumeToken.length > 8192) {
          this.#identificationFailed(new LobbyError('invalid_response', 'The lobby returned an invalid identity.'));
          return;
        }
        this.#identity = { playerId: identity.playerId, resumeToken: identity.resumeToken };
        this.#persist();
        this.#attempts = 0;
        this.store.patch({ identity: { playerId: identity.playerId }, connection: 'connected', error: null });
        for (const request of this.#pending.values()) {
          if (!request.hello) this.#write(request.envelope);
        }
      }).catch(error => {
        // A transport interruption is retried with the same stored identity.
        if (this.#socket !== socket || this.#destroyed || !this.#desired) return;
        this.#identificationFailed(error);
      });
    });
    socket.addEventListener('message', event => {
      if (this.#socket === socket && !this.#destroyed) this.#receive(event.data);
    });
    socket.addEventListener('error', () => {
      // Browsers follow a failed socket with close; closing also covers fakes
      // and hosts that otherwise leave identification hanging indefinitely.
      if (this.#socket === socket && socket.readyState < 2) socket.close();
    });
    socket.addEventListener('close', event => {
      if (this.#socket !== socket || this.#destroyed) return;
      this.#socket = null;
      if (this.#helloId) this.#settle(this.#helloId, new LobbyError('disconnected', 'The lobby connection was interrupted.'));
      this.#helloId = null;
      if (event.reason === 'replaced') {
        this.#desired = false;
        const error = new LobbyError('session_replaced', 'This lobby session was opened elsewhere. Start a new session to continue here.');
        this.#rejectPending(error);
        this.store.patch({ connection: 'identity-rejected', error: { code: error.code, message: error.message } });
        return;
      }
      if (this.#desired) this.#scheduleReconnect();
    });
  }

  #identificationFailed(error) {
    this.#desired = false;
    this.store.patch({ connection: this.#identity ? 'identity-rejected' : 'error', error: { code: error.code || 'unauthorized', message: error.message } });
    this.#rejectPending(error);
    this.#socket?.close();
  }

  #scheduleReconnect() {
    if (!this.#desired || this.#destroyed || this.#reconnectTimer != null) return;
    this.store.patch({ connection: 'reconnecting' });
    const delay = Math.min(this.maxReconnectDelayMs, this.reconnectDelayMs * (2 ** Math.min(this.#attempts++, 6)));
    this.#reconnectTimer = this.schedule(() => { this.#reconnectTimer = null; this.#open(true); }, delay);
  }

  #requestId() { return `${this.#prefix}-${++this.#counter}`; }
  #write(envelope) {
    if (this.#socket?.readyState !== 1) return false;
    try { this.#socket.send(JSON.stringify(envelope)); return true; }
    catch { this.#socket.close(); return false; }
  }

  #send(type, payload, requestId = this.#requestId(), hello = false) {
    if (this.#pending.size >= 64) return Promise.reject(new LobbyError('rate_limited', 'Wait for the current lobby requests to finish.'));
    let wirePayload;
    try {
      // A pending command retains its exact content and request ID on retry,
      // even if a caller subsequently mutates the object passed to command().
      const serialized = JSON.stringify(payload);
      if (serialized.length > 15000) throw new Error();
      wirePayload = JSON.parse(serialized);
      if (!wirePayload || typeof wirePayload !== 'object' || Array.isArray(wirePayload)) throw new Error();
    } catch { return Promise.reject(new LobbyError('invalid_request', 'Invalid lobby action payload.')); }
    return new Promise((resolve, reject) => {
      const envelope = { v: 1, type, requestId, payload: wirePayload };
      const timer = this.schedule(() => {
        const error = new LobbyError('timeout', 'The lobby did not confirm that action. Reconnect to check the latest state.');
        this.#settle(requestId, error);
      }, this.commandTimeoutMs);
      this.#pending.set(requestId, { envelope, resolve, reject, timer, hello });
      this.store.patch({ pendingCount: this.#pending.size });
      this.#write(envelope);
    });
  }

  command(type, payload = {}) {
    if (!COMMANDS.has(type)) return Promise.reject(new LobbyError('invalid_request', 'Unknown lobby action.'));
    if (this.#destroyed || this.state.connection !== 'connected') return Promise.reject(new LobbyError('disconnected', 'Reconnect to the lobby before taking an action.'));
    this.store.patch({ error: null });
    return this.#send(type, payload);
  }

  #settle(requestId, error, payload) {
    const request = this.#pending.get(requestId);
    if (!request) return;
    this.unschedule(request.timer);
    this.#pending.delete(requestId);
    if (requestId === this.#helloId) this.#helloId = null;
    this.store.patch({ pendingCount: this.#pending.size, ...(error && !request.hello ? { error: { code: error.code, message: error.message } } : {}) });
    if (error) request.reject(error); else request.resolve(payload ?? {});
  }

  #receive(data) {
    try {
      if (typeof data !== 'string' || data.length > 8 * 1024 * 1024) throw new Error('Invalid lobby message.');
      const envelope = JSON.parse(data);
      if (envelope.v !== 1 || typeof envelope.type !== 'string') throw new Error('Unsupported lobby message.');
      if (envelope.type === 'response') {
        if (typeof envelope.requestId !== 'string' || typeof envelope.ok !== 'boolean') throw new Error('Invalid lobby response.');
        const error = envelope.ok ? null : new LobbyError(
          typeof envelope.error?.code === 'string' ? envelope.error.code : 'invalid_response',
          typeof envelope.error?.message === 'string' ? envelope.error.message : 'The lobby could not complete that action.',
        );
        this.#settle(envelope.requestId, error, envelope.payload);
      } else if (envelope.type === 'snapshot') {
        this.store.snapshot(envelope.payload, envelope.seq);
      } else if (envelope.type === 'game.launch') {
        const descriptor = normalizeLaunch(envelope.payload);
        if (!this.store.acceptSequence(envelope.seq)) return;
        if (descriptor.playerId !== this.#identity?.playerId || this.#launchedSessions.has(descriptor.sessionId)) return;
        this.#launchedSessions.add(descriptor.sessionId);
        // Keep duplicate suppression bounded across many runs in a single tab.
        if (this.#launchedSessions.size > 128) this.#launchedSessions.delete(this.#launchedSessions.values().next().value);
        for (const listener of this.#launchSubscribers) {
          try { listener(descriptor); }
          catch { this.store.patch({ error: { code: 'launch_failed', message: 'The game could not open. Return to the lobby and reconnect.' } }); }
        }
      }
    } catch (error) {
      this.store.patch({ error: { code: 'invalid_response', message: error.message || 'Invalid lobby response.' } });
    }
  }

  #rejectPending(error) {
    for (const requestId of [...this.#pending.keys()]) this.#settle(requestId, error);
  }

  resetIdentity() {
    if (this.#destroyed) return;
    this.#desired = false;
    if (this.#reconnectTimer != null) this.unschedule(this.#reconnectTimer);
    this.#reconnectTimer = null;
    const old = this.#socket;
    this.#socket = null;
    old?.close();
    this.#rejectPending(new LobbyError('session_reset', 'A new lobby session was requested.'));
    this.#identity = null;
    this.#launchedSessions.clear();
    this.#persist();
    this.store.patch({ connection: 'idle', identity: null, snapshot: null, error: null, sequence: -1 });
  }

  destroy() {
    if (this.#destroyed) return;
    this.#destroyed = true;
    this.#desired = false;
    if (this.#reconnectTimer != null) this.unschedule(this.#reconnectTimer);
    this.#reconnectTimer = null;
    const socket = this.#socket;
    this.#socket = null;
    socket?.close();
    this.#rejectPending(new LobbyError('destroyed', 'The lobby was closed.'));
    this.store.patch({ connection: 'closed' });
    this.#launchSubscribers.clear();
    this.store.clearSubscribers();
  }
}

export function createLobbyClient(options) { return new LobbyClient(options); }
