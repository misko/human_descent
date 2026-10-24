import assert from 'node:assert/strict';
import TeamGameState from '../client/TeamGameState.js';

const launch = { matchId: 'match', sessionId: 'team', playerId: 'a', token: 'private' };
const state = (overrides = {}) => ({ ...launch, teamId: 'social', panelIndex: 1, generation: 0,
    modelRevision: 0, dimsOffset: 0, phase: 'running', config: { meshStepSize: .1 },
    roster: [{ playerId: 'a', panelIndex: 1, voted: false }, { playerId: 'b', panelIndex: 0, voted: false }], ...overrides });
const result = (overrides = {}) => ({ teamSessionId: 'team', teamMatchId: 'match', generation: 0, modelRevision: 0, jobId: 1, ...overrides });

assert.throws(() => new TeamGameState({}), /Missing launch/);
const team = new TeamGameState(launch);
assert.equal(team.inputEnabled, false);
assert.equal(team.apply(state({ sessionId: 'wrong' })), null);
assert.equal(team.apply(state(), 1).generationChanged, true);
assert.equal(team.inputEnabled, false, 'snapshot waits for matching surface');
assert.equal(team.acceptsResult(result()), true);
team.updating = false;
assert.equal(team.inputEnabled, true);
assert.equal(team.apply(state({ phase: 'countdown' })), null, 'obsolete lifecycle states cannot restart the countdown');
assert.equal(team.ownsDimension(2), true);
assert.equal(team.ownsDimension(3), true);
assert.equal(team.ownsDimension(0), false);
assert.deepEqual(team.selectablePanels, [1, 2]);
assert.equal(team.ownsDimension(4), true);
assert.equal(team.ownsDimension(5), true);
for (const dim of [-1, 6, 1.5, NaN, Infinity]) assert.equal(team.ownsDimension(dim), false);
for (const size of [1, 2, 3]) {
    for (let owner = 0; owner < size; owner++) {
        const permissions = new TeamGameState(launch);
        permissions.apply(state({ panelIndex: owner,
            roster: Array.from({ length: size }, (_, panelIndex) => ({ panelIndex, connected: false })) }));
        for (let dim = 0; dim < 6; dim++) {
            assert.equal(permissions.ownsDimension(dim), Math.floor(dim / 2) === owner || Math.floor(dim / 2) >= size);
        }
    }
}

team.voteSentGeneration = 0;
assert.equal(team.canVote, false);
assert.equal(team.apply(state({ roster: [{ playerId: 'a', panelIndex: 1, voted: true }, { playerId: 'b', panelIndex: 0, voted: false }] })).generationChanged, false);
assert.equal(team.updating, false, 'partial votes do not interrupt movement');
assert.equal(team.inputEnabled, true);

assert.equal(team.apply(state({ generation: 1, dimsOffset: 4, modelRevision: 2 }), 3).generationChanged, true);
assert.equal(team.voteSentGeneration, null);
assert.equal(team.updating, true);
assert.equal(team.acceptsResult(result({ modelRevision: 3, jobId: 4 })), false, 'old generations never populate new surfaces');
assert.equal(team.apply(state()), null, 'stale snapshots are ignored');
assert.equal(team.acceptsResult(result({ generation: 1, modelRevision: 1, jobId: 4 })), false);
assert.equal(team.acceptsResult(result({ generation: 1, modelRevision: 2, jobId: 2 })), false);
assert.equal(team.acceptsResult(result({ generation: 1, modelRevision: 2, jobId: 3 })), true);
team.updating = false;
team.apply(state({ generation: 1, dimsOffset: 4, modelRevision: 2, config: { meshStepSize: .2 } }), 4);
assert.equal(team.updating, true, 'same-generation shared scale updates wait for new surfaces');
assert.equal(team.acceptsResult(result({ generation: 1, modelRevision: 2, jobId: 3 })), false);
assert.equal(team.acceptsResult(result({ generation: 1, modelRevision: 2, jobId: 4 })), true);
team.apply(state({ generation: 1, modelRevision: 2, phase: 'finished', results: [{ teamId: 'social', loss: 1.2 }] }));
assert.equal(team.inputEnabled, false);
assert.equal(team.acceptsResult(result({ generation: 1, modelRevision: 2, jobId: 5 })), false, 'late work cannot replace terminal scores');
assert.equal(team.apply(state({ generation: 2, modelRevision: 3 })), null, 'terminal state never regresses');
assert.equal(team.apply(state({ generation: 1, modelRevision: 2, phase: 'finished', results: [{ teamId: 'social', loss: 99 }] })), null);
assert.equal(team.results[0].loss, 1.2, 'duplicate terminal events cannot replace the frozen result');

const limited = new TeamGameState(launch);
limited.apply(state({ dimensionSwitchesRemaining: 1 }));
limited.updating = false;
assert.equal(limited.canVote, true);
limited.apply(state({ generation: 1, dimensionSwitchesRemaining: 0 }));
limited.updating = false;
assert.equal(limited.canVote, false, 'explicit zero prevents a new vote');
assert.equal(limited.inputEnabled, true, 'budget exhaustion still allows optimization');
const unlimited = new TeamGameState(launch);
const inheritedDefault = Object.assign(Object.create({ dimensionSwitchesRemaining: 0 }), state());
unlimited.apply(inheritedDefault);
unlimited.updating = false;
assert.equal(unlimited.switchesRemaining, null, 'absent protobuf field is unlimited, not its prototype zero');
assert.equal(unlimited.canVote, true);
