// The lobby store contains public social state only. Admission and resume tokens
// belong to the connection, never to a snapshot or the rendered roster.
import { normalizeSettings } from './matchSettings.js';
const COLLECTIONS = ['players', 'teams', 'invitations', 'challenges', 'matches', 'chat'];
const TERMINAL_PHASES = new Set(['finished', 'cancelled', 'failed']);

function freeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

function string(value, name, max = 4096) {
  if (typeof value !== 'string' || value.length > max) throw new Error(`Invalid lobby ${name}.`);
  return value;
}

function id(value) {
  const result = string(value, 'identifier', 256);
  if (!result) throw new Error('Invalid lobby identifier.');
  return result;
}
function optionalId(value) { return value == null ? null : id(value); }
function ids(value) {
  if (!Array.isArray(value) || value.length > 4096) throw new Error('Invalid lobby roster.');
  return value.map(id);
}
function time(value) {
  if (value == null) return null;
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('Invalid lobby time.');
  return value;
}

export function normalizeSnapshot(payload) {
  if (!payload || typeof payload !== 'object' || !payload.you) throw new Error('Invalid lobby snapshot.');
  for (const key of COLLECTIONS) {
    // Pending challenges count against both captains' limits. The backend's
    // default maximum plus bounded terminal history can exceed 4096 entries.
    const limit = key === 'challenges' ? 8192 : 4096;
    if (!Array.isArray(payload[key]) || payload[key].length > limit) throw new Error(`Invalid lobby ${key}.`);
  }
  const you = payload.you;
  // Copy only wire-contract fields. Unknown fields cannot leak private tokens
  // into subscriptions, diagnostic UI, or public state inspection.
  return freeze({
    you: { id: id(you.id), name: string(you.name, 'name', 512), teamId: optionalId(you.teamId) },
    players: payload.players.map(p => ({ id: id(p.id), name: string(p.name, 'name', 512), connected: p.connected === true, teamId: optionalId(p.teamId) })),
    teams: payload.teams.map(t => ({ id: id(t.id), name: string(t.name, 'team name', 512), captainId: id(t.captainId), memberIds: ids(t.memberIds), revision: t.revision, status: string(t.status, 'team status', 64) })),
    invitations: payload.invitations.map(i => ({ id: id(i.id), teamId: id(i.teamId), inviterId: id(i.inviterId), recipientId: id(i.recipientId), status: string(i.status, 'invitation status', 64), expiresAt: time(i.expiresAt) })),
    challenges: payload.challenges.map(c => ({ id: id(c.id), challengerTeamId: id(c.challengerTeamId), targetTeamId: id(c.targetTeamId), status: string(c.status, 'challenge status', 64), expiresAt: time(c.expiresAt), settings: normalizeSettings(c.settings) })),
    matches: payload.matches.map(m => ({
      id: id(m.id), teamIds: ids(m.teamIds), phase: string(m.phase, 'match phase', 64),
      settings: normalizeSettings(m.settings),
      readyPlayerIds: m.readyPlayerIds == null ? [] : ids(m.readyPlayerIds),
      startAt: time(m.startAt), deadline: time(m.deadline),
      results: m.results == null ? null : m.results.map(r => ({ teamId: id(r.teamId), loss: typeof r.loss === 'number' && Number.isFinite(r.loss) ? r.loss : null })),
      reason: m.reason == null ? null : string(m.reason, 'match reason'),
    })),
    chat: payload.chat.map(c => ({ id: id(c.id), playerId: id(c.playerId), name: string(c.name, 'name', 512), text: string(c.text, 'chat message', 8000), sentAt: time(c.sentAt) })),
  });
}

export function normalizeLaunch(payload) {
  if (!payload || typeof payload !== 'object' || !Number.isInteger(payload.panelIndex) || payload.panelIndex < 0 || payload.panelIndex > 2) {
    throw new Error('Invalid gameplay launch.');
  }
  const token = string(payload.token, 'admission token', 8192);
  const wsUrl = string(payload.wsUrl, 'gameplay endpoint', 2048);
  if (!token || !wsUrl) throw new Error('Invalid gameplay launch.');
  return freeze({
    matchId: id(payload.matchId), teamId: id(payload.teamId), playerId: id(payload.playerId),
    sessionId: id(payload.sessionId), panelIndex: payload.panelIndex,
    token, wsUrl,
  });
}

export function deriveLobby(snapshot) {
  const players = new Map((snapshot?.players || []).map(p => [p.id, p]));
  const teams = new Map((snapshot?.teams || []).map(t => [t.id, t]));
  const you = snapshot?.you ?? null;
  const team = teams.get(you?.teamId) ?? null;
  const captain = !!team && team.captainId === you?.id;
  const ownMatches = (snapshot?.matches || []).filter(m => team && m.teamIds.includes(team.id));
  return {
    you, team, captain, players, teams,
    activeMatch: ownMatches.findLast(m => !TERMINAL_PHASES.has(m.phase)) ?? null,
    lastResult: ownMatches.findLast(m => TERMINAL_PHASES.has(m.phase)) ?? null,
    invitations: (snapshot?.invitations || []).filter(i => i.status === 'pending'),
    challenges: (snapshot?.challenges || []).filter(c => c.status === 'pending'),
  };
}

export function createLobbyStore() {
  let state = freeze({ connection: 'idle', snapshot: null, identity: null, error: null, pendingCount: 0, sequence: -1 });
  const subscribers = new Set();
  function patch(changes) {
    state = freeze({ ...state, ...changes });
    for (const listener of subscribers) listener(state);
    return state;
  }
  return {
    get state() { return state; },
    patch,
    beginConnection(connection) { patch({ connection, sequence: -1 }); },
    snapshot(payload, sequence) {
      if (!Number.isSafeInteger(sequence) || sequence < 0 || sequence <= state.sequence) return false;
      const snapshot = normalizeSnapshot(payload);
      patch({ snapshot, sequence });
      return true;
    },
    acceptSequence(sequence) {
      if (!Number.isSafeInteger(sequence) || sequence < 0 || sequence <= state.sequence) return false;
      patch({ sequence });
      return true;
    },
    subscribe(listener) {
      subscribers.add(listener);
      listener(state);
      return () => subscribers.delete(listener);
    },
    clearSubscribers() { subscribers.clear(); },
  };
}
