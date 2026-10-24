import assert from 'node:assert/strict';
import HudesClient from '../client/HudesClient.js';
import KeyboardClient from '../client/KeyboardClient.js';
import KeyboardClientGL from '../client/KeyboardClientGL.js';
import ClientState from '../client/ClientState.js';
import TeamGameState from '../client/TeamGameState.js';

globalThis.WebSocket = { OPEN: 1 };
globalThis.window = new EventTarget();
globalThis.document = new EventTarget();
document.body = { classList: { remove() {} } };
const launch = { matchId: 'm', sessionId: 's', playerId: 'p', token: 'private' };
const team = new TeamGameState(launch);
team.apply({ ...launch, panelIndex: 1, generation: 7, dimsOffset: 28, modelRevision: 3, phase: 'running',
    roster: [{ playerId: 'q', panelIndex: 0 }, { playerId: 'p', panelIndex: 1 }], config: { meshStepSize: .1 } });
team.updating = false;
const sent = [];
const client = Object.assign(Object.create(HudesClient.prototype), { team,
    state: new ClientState(-.05, 12), Control: {}, ControlType: { values: { CONTROL_NEXT_DIMS: 102, CONTROL_DIMS: 0, CONTROL_CONFIG: 103 } },
    socket: { readyState: 1 }, _readyForInput: true, dimsOffset: 28, dimsAndStepsOnCurrentDims: [1, 2, 3, 4],
    sendQ(payload) { sent.push(payload); }, _pressedCodes: new Set(), _blockedCodes: new Set(), keyHolds: {} });
client.state.n = 4;
client.state.dimsUsed = 28;
await client.getNextDims();
await client.getNextDims();
assert.equal(sent.length, 1, 'one request per generation');
assert.equal(client.dimsOffset, 28, 'votes never increment counters optimistically');
assert.deepEqual(client.dimsAndStepsOnCurrentDims, [1, 2, 3, 4], 'partial vote preserves accepted movement tracking');
await client.sendDimsAndSteps({ 0: 1 });
assert.equal(sent.length, 1, 'foreign dimensions rejected locally');
await client.sendDimsAndSteps({ 2: 1 });
assert.equal(sent.length, 2, 'moving remains possible after partial vote');
assert.equal(client.dimsAndStepsOnCurrentDims[2], 4);

let processed = 0;
client.processKeyPress = () => processed++;
client._handleKeyDown({ code: 'Space', repeat: true });
assert.equal(processed, 0, 'Space autorepeat never votes');
client._pressedCodes.add('KeyW');
client._stopHeldInput();
client._handleKeyDown({ code: 'KeyW', repeat: false });
assert.equal(processed, 0, 'held keys stay blocked after refresh');
client._handleKeyUp({ code: 'KeyW', target: { tagName: 'INPUT' } });
client._handleKeyDown({ code: 'KeyW', repeat: false });
assert.equal(processed, 1, 'keyup releases blocked controls even from chat');

const gl = Object.assign(Object.create(KeyboardClientGL.prototype), { team, state: { stepSize: 1 },
    view: { getSelectedGrid: () => 0 }, sendDimsAndSteps: payload => sent.push(payload) });
const sentBeforeForbidden = sent.length;
gl.stepInSelectedGrid(1, 1);
assert.equal(sent.length, sentBeforeForbidden, 'forged selection cannot move a teammate’s panel');
gl.view.getSelectedGrid = () => 2;
gl.stepInSelectedGrid(1, 1);
assert.deepEqual(Object.keys(sent.at(-1)).sort(), ['4', '5'], 'movement targets the selected shared landscape');
await client.sendDimsAndSteps({ 4: 1, 5: 2 });
assert.deepEqual(sent.at(-1).dimsAndSteps, [{ dim: 4, step: 1 }, { dim: 5, step: 2 }]);

let repeats = 0;
const keyboard = Object.assign(Object.create(KeyboardClient.prototype), { _destroyed: false, _timers: new Set(), _intervals: new Set(),
    activeRepeats: new Map(), keyRepeatDelayMs: 1, keyRepeatIntervalMs: 1, state: { helpScreenIdx: -1 },
    _pressedCodes: new Set(['KeyW']), _blockedCodes: new Set(), keyHolds: {} });
keyboard._startRepeat('w', () => repeats++);
keyboard._setTextCaptureActive(true);
await new Promise(resolve => setTimeout(resolve, 10));
assert.equal(repeats, 0, 'chat focus cancels pending movement repeats');
assert.equal(keyboard.activeRepeats.size, 0);

let closed = 0, disposed = 0;
const lifecycle = Object.assign(Object.create(HudesClient.prototype), { _timers: new Set(), _intervals: new Set(),
    _pressedCodes: new Set(), _blockedCodes: new Set(), keyHolds: {}, _abortController: new AbortController(),
    socket: { close() { closed++; } }, view: { destroy() { disposed++; } } });
let late = 0;
lifecycle._setTimeout(() => late++, 1);
lifecycle._setInterval(() => late++, 1);
lifecycle.destroy();
lifecycle.destroy();
lifecycle._scheduleReconnect();
await new Promise(resolve => setTimeout(resolve, 10));
assert.equal(late, 0);
assert.equal(closed, 1);
assert.equal(disposed, 1);
assert.equal(lifecycle._shouldReconnect, false);
assert.equal(lifecycle._timers.size, 0);

// Team inference carries a job index shared by all players, rather than a solo
// browser's request index. Consecutive jobs must both update the common history.
client.ControlType.values.CONTROL_TRAIN_LOSS_AND_PREDS = 1;
client.view = { updateLossChart() {}, updateLastStepsChart() {}, updateConfusionMatrix() {}, updateExamplePreds() {} };
client.trainSteps = [];
client.trainLosses = [];
client.valSteps = [];
client.valLosses = [];
client._serverAckRequestIdx = 0;
for (const jobId of [1, 2]) {
    client.Control.decode = () => ({ type: 1, teamSessionId: 's', teamMatchId: 'm', generation: 7,
        modelRevision: 3, jobId, totalSgdSteps: 0,
        trainLossAndPreds: { trainLoss: 2 / jobId, preds: [.1], predsShape: [1, 1], confusionMatrix: [1], confusionMatrixShape: [1, 1] } });
    client._handleMessage({ data: new ArrayBuffer(0) });
}
assert.deepEqual(client.trainSteps, [1, 2]);
assert.equal(client.state.sharedLoss, 1);
assert.equal(client.team.currentLoss, 1, 'mobile and desktop use the same shared loss');
