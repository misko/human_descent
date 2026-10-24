import assert from 'node:assert/strict';
import KeyboardClientGL from '../client/KeyboardClientGL.js';
import TeamGameHud from '../client/TeamGameHud.js';
import { supportsMultiplayer } from '../mobile.js';

class Target extends EventTarget {
    listeners = new Map();
    addEventListener(type, callback, options) {
        super.addEventListener(type, callback, options);
        const callbacks = this.listeners.get(type) || new Set();
        callbacks.add(callback);
        this.listeners.set(type, callbacks);
    }
    removeEventListener(type, callback, options) {
        super.removeEventListener(type, callback, options);
        this.listeners.get(type)?.delete(callback);
    }
    get listenerCount() { return [...this.listeners.values()].reduce((sum, set) => sum + set.size, 0); }
}
class Element extends Target {
    constructor(tagName) { super(); this.tagName = tagName.toUpperCase(); }
    style = {};
    dataset = {};
    children = [];
    classList = { add() {}, remove() {}, toggle() {} };
    append(...elements) { this.children.push(...elements); }
    appendChild(element) { this.append(element); }
    setAttribute() {}
    replaceChildren(...elements) { this.children = elements; }
    remove() { this.removed = true; }
}
globalThis.HTMLElement = Element;
globalThis.window = new Target();
window.location = { href: 'http://web.local:6173/?mode=1d', search: '?mode=1d', protocol: 'http:', hostname: 'web.local' };
globalThis.document = new Target();
document.body = new Element('body');
document.getElementById = () => null;
document.createElement = tag => new Element(tag);
document.querySelector = () => null;
globalThis.WebSocket = class {
    static OPEN = 1;
    readyState = 1;
    sends = [];
    constructor(url) { this.url = url; }
    send(payload) { this.sends.push(payload); }
    close() { this.readyState = 3; }
};

class Probe extends KeyboardClientGL {
    initInput() {
        this.inputInitializations = (this.inputInitializations || 0) + 1;
        this.testCanvas = new Element('canvas');
        if (Probe.pointerSupported !== false) this.testCanvas.onpointerdown = null;
        this.view.getCanvasElement = () => this.testCanvas;
        super.initInput();
    }
    async loadProto() {
        await Promise.resolve();
        if (this._destroyed) return;
        this.ControlType = { values: { CONTROL_TEAM_ADMIT: 110, CONTROL_CONFIG: 103 } };
        this.Control = { verify: () => null, create: payload => payload, encode: payload => ({ finish: () => payload }) };
    }
    async _loadLevels() {}
}
const launchDescriptor = { matchId: 'm', sessionId: 's', teamId: 't', playerId: 'p', token: 'secret', wsUrl: '/ws' };
const beforeBlockedLaunch = document.body.children.length;
let blockedSocketAllocations = 0;
let blockedDomAccess = 0;
const getElementById = document.getElementById;
document.getElementById = () => { blockedDomAccess++; return null; };
const Socket = globalThis.WebSocket;
globalThis.WebSocket = class extends Socket {
    constructor(...args) { super(...args); blockedSocketAllocations++; }
};
const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { userAgent: 'Mozilla/5.0 iPhone Mobile', maxTouchPoints: 1 } });
assert.throws(() => new Probe('game.local', 8765, { launchDescriptor, isMobile: true }), { code: 'desktop_required' });
window.innerWidth = 390;
window.location.search = '?mobile=0';
assert.throws(() => new Probe('game.local', 8765, { launchDescriptor, isMobile: false }), { code: 'desktop_required' }, 'a caller cannot override actual mobile detection');
assert.equal(document.body.children.length, beforeBlockedLaunch, 'unsupported launches allocate no renderer or other DOM');
assert.equal(blockedDomAccess, 0, 'unsupported launches never reach renderer or chart initialization');
assert.equal(blockedSocketAllocations, 0, 'unsupported launches allocate no gameplay socket');
assert.equal(window.listenerCount, 0, 'unsupported launches install no input handlers');
globalThis.WebSocket = Socket;
document.getElementById = getElementById;
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { userAgent: 'Mozilla/5.0 X11 Linux x86_64', maxTouchPoints: 0 } });
assert.equal(supportsMultiplayer(), true, 'a narrow desktop viewport remains supported');
window.location.search = '?mode=1d&mobile=1';
for (let cycle = 0; cycle < 3; cycle++) {
    Probe.pointerSupported = cycle !== 2;
    const client = new Probe('game.local', 8765, { wsUrl: 'ws://game.local:8765/ws', launchDescriptor, isMobile: true });
    assert.equal(client.inputInitializations, 1, 'GL initializes input once through the existing constructor chain');
    assert.equal(client.renderMode, '3d', 'team landscape mode takes precedence over solo URL mode');
    assert.equal(client.isMobile, false, 'a desktop layout override cannot enable mobile gameplay');
    assert.equal(client.socket.url, 'ws://game.local:8765/ws', 'relative descriptor keeps the configured backend');
    assert.equal(client.state.batchSize, undefined, 'team does not adopt device defaults');
    const socket = client.socket;
    socket.onopen();
    await client._protoReadyPromise;
    await Promise.resolve();
    assert.equal(socket.sends.length, 1);
    assert.equal(socket.sends[0].type, 110, 'team enters through admission');
    assert.equal(socket.sends[0].config, undefined, 'team never sends legacy solo Config');
    assert.equal(socket.sends[0].teamAdmission.token, 'secret');
    assert.equal(client._readyForInput, false, 'prepared connection cannot move before authoritative state');
    const loop = client.runLoop();
    const overlay = client.teamHud.root;
    const down = Probe.pointerSupported ? 'pointerdown' : 'mousedown';
    const move = Probe.pointerSupported ? 'pointermove' : 'mousemove';
    const mouseEvent = (type, x) => Object.assign(new Event(type), { button: 0, clientX: x, clientY: 10, pointerId: 1 });
    let rotations = 0;
    client.state.helpScreenIdx = -1;
    client.view.getAngles = () => ({ angleH: 0, angleV: 20 });
    client.view.adjustAngles = () => rotations++;
    client.testCanvas.dispatchEvent(mouseEvent(down, 10));
    window.dispatchEvent(mouseEvent(move, 20));
    assert.ok(rotations > 0, 'real mouse controls rotate before disposal');
    const rotationsBeforeDestroy = rotations;
    client.destroy();
    client.destroy();
    await loop;
    window.dispatchEvent(mouseEvent(move, 30));
    assert.equal(rotations, rotationsBeforeDestroy, 'leaving mid-drag cannot call a disposed view');
    assert.equal(client.testCanvas.listenerCount, 0, 'return removes canvas mouse, wheel and context-menu handlers');
    assert.equal(window.listenerCount, 0, 'return removes every client window listener');
    assert.equal(document.listenerCount, 0, 'return removes every client document listener');
    assert.equal(overlay.removed, true);
    assert.equal(socket.readyState, 3);
    assert.equal(client._intervals.size, 0);
}

// A prototype request settling after disposal must not install controls or admit.
const late = new Probe('game.local', 8765, { wsUrl: 'ws://game.local:8765/ws', launchDescriptor });
late.socket.onopen();
const lateSocket = late.socket;
late.destroy();
await late._protoReadyPromise;
await Promise.resolve();
assert.equal(lateSocket.sends.length, 0);
assert.equal(late.Control, null);

for (const [ownLoss, otherLoss, phase, expected] of [
    [1, 2, 'finished', 'win'], [2, 1, 'finished', 'loss'], [1, 1, 'finished', 'tie'],
    [Infinity, 1, 'failed', 'failed'],
]) {
    let returned = 0;
    const hud = new TeamGameHud(() => returned++);
    hud.showResults({ launch: { teamId: 'ours' }, phase,
        results: [{ teamId: 'ours', loss: ownLoss }, { teamId: 'theirs', loss: otherLoss }] });
    assert.equal(hud.results.dataset.result, expected);
    assert.match(hud.results.children[1].textContent, /Your team.*Opponent/);
    hud.results.children[2].onclick();
    assert.equal(returned, 1, 'terminal return action invokes composition callback');
    hud.destroy();
}
if (navigatorDescriptor) Object.defineProperty(globalThis, 'navigator', navigatorDescriptor);
else delete globalThis.navigator;
