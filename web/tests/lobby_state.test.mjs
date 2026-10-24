import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createLobbyStore, deriveLobby, normalizeLaunch, normalizeSnapshot } from '../lobby/state.js';

// Representative wire data follows docs/lobby_protocol.md, including nulls,
// private invitation filtering and unequal-team challenge participants.
function fixture() {
  return {
    you: { id: 'alice', name: 'Alice', teamId: 'one' },
    players: [
      { id: 'alice', name: 'Alice', connected: true, teamId: 'one' },
      { id: 'bob', name: 'Bob', connected: true, teamId: 'three' },
      { id: 'carol', name: 'Carol', connected: false, teamId: 'three' },
      { id: 'dan', name: 'Dan', connected: true, teamId: 'three' },
      { id: 'eve', name: '<img src=x onerror=alert(1)>', connected: true, teamId: null },
    ],
    teams: [
      { id: 'one', name: 'Local Minima', captainId: 'alice', memberIds: ['alice'], revision: 1, status: 'available' },
      { id: 'three', name: 'Gradient Crew', captainId: 'bob', memberIds: ['bob', 'carol', 'dan'], revision: 3, status: 'available' },
    ],
    invitations: [{ id: 'invite', teamId: 'one', inviterId: 'alice', recipientId: 'eve', status: 'pending', expiresAt: 1780000120 }],
    challenges: [{ id: 'challenge', challengerTeamId: 'one', targetTeamId: 'three', status: 'pending', expiresAt: 1780000120 }],
    matches: [{ id: 'previous', teamIds: ['one', 'three'], phase: 'finished', readyPlayerIds: [], startAt: 1779999000, deadline: 1779999120, results: [{ teamId: 'one', loss: 0.75 }, { teamId: 'three', loss: 0.75 }], reason: null }],
    chat: [{ id: 'message', playerId: 'eve', name: '<img src=x onerror=alert(1)>', text: '<script>alert(1)</script>\nHello!', sentAt: 1780000000 }],
  };
}

test('documented snapshots preserve authoritative social state, safe text and finite tied scores', () => {
  const input = fixture();
  input.resumeToken = 'must-not-be-public';
  input.you.token = 'must-not-be-public';
  input.players[0].resumeToken = 'must-not-be-public';
  const snapshot = normalizeSnapshot(input);
  const view = deriveLobby(snapshot);
  assert.equal(view.captain, true);
  assert.equal(view.team.id, 'one');
  assert.equal(view.teams.get('three').memberIds.length, 3);
  assert.equal(view.invitations[0].recipientId, 'eve');
  assert.equal(view.challenges[0].targetTeamId, 'three');
  assert.equal(view.activeMatch, null);
  assert.equal(view.lastResult.results[0].loss, view.lastResult.results[1].loss);
  assert.equal(snapshot.chat[0].text, '<script>alert(1)</script>\nHello!');
  assert.equal(snapshot.players[4].name, '<img src=x onerror=alert(1)>');
  assert.equal(JSON.stringify(snapshot).includes('must-not-be-public'), false);
  input.teams[0].memberIds.push('eve');
  assert.deepEqual(snapshot.teams[0].memberIds, ['alice']);
  assert.throws(() => snapshot.teams[0].memberIds.push('eve'), TypeError);
});

test('snapshots replace local state, ignore old events, and reset sequence after reconnection', () => {
  const store = createLobbyStore();
  store.beginConnection('connected');
  assert.equal(store.snapshot(fixture(), 12), true);
  const old = store.state.snapshot;
  assert.equal(store.snapshot({ broken: true }, 11), false);
  assert.equal(store.state.snapshot, old);
  const next = fixture();
  next.you.teamId = null;
  next.teams = next.teams.filter(t => t.id !== 'one');
  next.players[0].teamId = null;
  assert.equal(store.snapshot(next, 13), true);
  assert.equal(deriveLobby(store.state.snapshot).team, null);
  assert.equal(store.snapshot(fixture(), 13), false);
  store.beginConnection('reconnecting');
  assert.equal(store.state.sequence, -1);
  assert.equal(store.snapshot(fixture(), 1), true);
  assert.equal(deriveLobby(store.state.snapshot).team.id, 'one');
});

test('malformed snapshots are rejected atomically; bounded 4096-player presence is accepted', () => {
  const store = createLobbyStore();
  store.snapshot(fixture(), 1);
  const old = store.state;
  const malformed = fixture();
  malformed.matches[0].readyPlayerIds = 'alice';
  assert.throws(() => store.snapshot(malformed, 2), /roster/);
  assert.equal(store.state, old);
  const capacity = fixture();
  capacity.players = Array.from({ length: 4096 }, (_, index) => ({ id: `p${index}`, name: 'Player', connected: false, teamId: null }));
  assert.equal(normalizeSnapshot(capacity).players.length, 4096);
  capacity.players.push({ id: 'overflow', name: 'Player', connected: false, teamId: null });
  assert.throws(() => normalizeSnapshot(capacity), /players/);
  const challenges = fixture();
  challenges.challenges = Array.from({ length: 4352 }, (_, index) => ({ ...challenges.challenges[0], id: `challenge-${index}` }));
  assert.equal(normalizeSnapshot(challenges).challenges.length, 4352);
});

test('own ready state and terminal history are derived independently without changing the team', () => {
  const input = fixture();
  input.teams.forEach(t => { t.status = 'busy'; });
  input.matches.push({ id: 'next', teamIds: ['one', 'three'], phase: 'ready', readyPlayerIds: ['alice'], startAt: null, deadline: null, results: null, reason: null });
  const view = deriveLobby(normalizeSnapshot(input));
  assert.equal(view.activeMatch.id, 'next');
  assert.deepEqual(view.activeMatch.readyPlayerIds, ['alice']);
  assert.equal(view.lastResult.id, 'previous');
  assert.deepEqual(view.team.memberIds, ['alice']);
  input.matches[1].phase = 'failed';
  input.matches[1].reason = 'evaluation_failed';
  input.matches[1].results = [{ teamId: 'one', loss: Infinity }];
  const failed = deriveLobby(normalizeSnapshot(input));
  assert.equal(failed.activeMatch, null);
  assert.equal(failed.lastResult.results[0].loss, null);
});

test('private launch descriptor is copied, bounded and restricted to one of three panels', () => {
  const launch = { matchId: 'match', teamId: 'one', playerId: 'alice', sessionId: 'session', panelIndex: 0, token: 'private-admission', wsUrl: '/ws' };
  const copy = normalizeLaunch({ ...launch, model: { mutable: true } });
  assert.deepEqual(copy, launch);
  assert.equal(Object.isFrozen(copy), true);
  for (const panelIndex of [-1, 3, 0.5, '0']) assert.throws(() => normalizeLaunch({ ...launch, panelIndex }));
  assert.throws(() => normalizeLaunch({ ...launch, token: '' }));
  assert.throws(() => normalizeLaunch({ ...launch, sessionId: '' }));
});
