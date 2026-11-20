import { access, writeFile } from 'node:fs/promises';
import { test, expect } from '@playwright/test';
import { openPlayer, createTeam, startChallenge, launchPlayers, returnPlayers, closePlayers } from '../helpers/multiplayer-browser.mjs';

const primary = process.env.HUDES_APP_ORIGIN;
const alias = process.env.HUDES_EDGE_ALIAS;
const reloadMarker = process.env.HUDES_EDGE_RELOAD_MARKER;
const idleSeconds = Number(process.env.HUDES_EDGE_IDLE_SECONDS ?? 0);
const absoluteSeconds = Number(process.env.HUDES_EDGE_ABSOLUTE_SECONDS ?? 0);

function socketURL(path) {
  const url = new URL(path, primary);
  url.protocol = 'wss:';
  return url.href;
}

async function livePaths(page) {
  return page.evaluate(() => window.__multiplayerSockets
    .filter(socket => socket.readyState === WebSocket.OPEN).map(socket => socket.url).sort());
}

async function restoredTeam(page) {
  return page.evaluate(() => {
    const team = window.__hudesClient?.team;
    if (!team?.inputEnabled || !Number.isFinite(team.currentLoss)) return null;
    return {
      match: team.launch.matchId,
      session: team.launch.sessionId,
      player: team.launch.playerId,
      team: team.teamId,
      slot: team.panelIndex,
      revision: team.modelRevision,
      score: team.currentLoss,
      members: team.roster.map(({ playerId, panelIndex }) => ({ playerId, panelIndex })),
    };
  });
}

async function verifyScores(page) {
  const response = await page.evaluate(async () => {
    const preflight = await fetch('/api/highscores', { method: 'OPTIONS' });
    const scores = await fetch('/api/highscores');
    return {
      status: preflight.status,
      bytes: (await preflight.arrayBuffer()).byteLength,
      framed: preflight.headers.has('content-length') || preflight.headers.has('transfer-encoding'),
      scores: scores.status,
      isArray: Array.isArray(await scores.json()),
    };
  });
  expect(response).toEqual({ status: 204, bytes: 0, framed: false, scores: 200, isArray: true });
}

test('both TLS hostnames serve the built release and API', async ({ page }, testInfo) => {
  for (const [origin, file, assetFile] of [[primary, 'h2-02.png', 'h2-03.png'], [alias, 'h2-02-alias.png']]) {
    const asset = page.waitForResponse(response => /\/assets\/[^/]+\.js$/.test(new URL(response.url()).pathname));
    const index = await page.goto(`${origin}/?lobby=1`);
    expect(new URL(page.url()).origin).toBe(origin);
    expect(index.status()).toBe(200);
    const script = await asset;
    expect(new URL(script.url()).origin).toBe(origin);
    expect(script.status()).toBe(200);
    expect(script.headers()['content-type']).toMatch(/(?:java|ecma)script/i);
    await expect(page.getByTestId('lobby-name')).toBeVisible();
    expect(await page.evaluate(() => performance.getEntriesByType('navigation')[0].nextHopProtocol)).toBe('http/1.1');
    expect(await page.evaluate(async () => (await fetch('/assets/hudes-staging-missing.js')).status)).toBe(404);
    await verifyScores(page);
    await page.screenshot({ path: testInfo.outputPath(file) });
    if (assetFile) await page.screenshot({ path: testInfo.outputPath(assetFile) });
  }
});

test('1v1 keeps both sockets at the edge and recovers the same team slot and score', async ({ browser }, testInfo) => {
  const players = [];
  try {
    for (const name of ['Edge left', 'Edge right']) {
      const options = process.env.HUDES_EDGE_RECORD_VIDEO ? {
        recordVideo: { dir: testInfo.outputPath('videos'), size: { width: 720, height: 524 } },
      } : {};
      players.push(await openPlayer(browser, name, options));
    }
    const [left, right] = players;
    const lobby = [socketURL('/lobby/ws')];
    const both = [...lobby, socketURL('/ws')].sort();
    for (const { page } of players) {
      expect(new URL(page.url()).origin).toBe(primary);
      await expect.poll(() => livePaths(page)).toEqual(lobby);
    }

    if (idleSeconds > 60) {
      const socketCount = () => left.page.evaluate(() => window.__multiplayerSockets.length);
      const before = await socketCount();
      await new Promise(resolve => setTimeout(resolve, idleSeconds * 1_000));
      expect(await socketCount()).toBe(before);
      expect(await livePaths(left.page)).toEqual(lobby);
    }

    const leftTeam = await createTeam([left], 'Left team');
    const rightTeam = await createTeam([right], 'Right team');
    const binaryFrames = [];
    left.page.on('websocket', socket => {
      if (socket.url() !== socketURL('/ws')) return;
      socket.on('framereceived', ({ payload }) => {
        if (Buffer.isBuffer(payload)) binaryFrames.push(payload.length);
      });
    });
    await startChallenge(left, right, rightTeam);
    const acceptedAt = performance.now();
    await launchPlayers(players);
    for (const { page } of players) await expect.poll(() => livePaths(page)).toEqual(both);
    const initial = await Promise.all(players.map(({ page }) => restoredTeam(page)));
    expect(initial.map(state => state.team)).toEqual([leftTeam, rightTeam]);
    expect(initial[0].match).toBe(initial[1].match);
    expect(initial[0].session).not.toBe(initial[1].session);
    for (const [index, state] of initial.entries()) {
      expect(state.members).toEqual([{ playerId: players[index].id, panelIndex: state.slot }]);
    }
    expect(binaryFrames.some(bytes => bytes > 0)).toBe(true);
    await expect(left.page.locator('#gameRoot canvas').first()).toBeVisible();
    const firstRenderedUpdateMs = Math.round(performance.now() - acceptedAt);
    await left.page.screenshot({ path: testInfo.outputPath('h2-04.png') });
    await right.page.screenshot({ path: testInfo.outputPath('h2-05.png') });
    if (idleSeconds > 60) await left.page.screenshot({ path: testInfo.outputPath('h2-07.png') });

    await left.page.keyboard.press('KeyW');
    await expect.poll(async () => (await restoredTeam(left.page))?.revision).toBeGreaterThan(initial[0].revision);
    if (process.env.HUDES_EDGE_REAL_INFERENCE) await left.page.screenshot({ path: testInfo.outputPath('h2-06.png') });
    if (reloadMarker) {
      const socketsBefore = await left.page.evaluate(() => window.__multiplayerSockets.length);
      await writeFile(reloadMarker, 'reload\n', { flag: 'wx' });
      await expect.poll(async () => access(`${reloadMarker}.ready`).then(() => true, () => false)).toBe(true);
      await expect.poll(() => livePaths(left.page)).toEqual(both);
      expect(await left.page.evaluate(() => window.__multiplayerSockets.length)).toBe(socketsBefore);
      await verifyScores(left.page);
      await left.page.screenshot({ path: testInfo.outputPath('h2-10.png') });
    }

    const before = await restoredTeam(left.page);
    const reconnectAt = performance.now();
    await left.page.evaluate(() => {
      const game = window.__multiplayerSockets.find(socket => new URL(socket.url).pathname === '/ws' && socket.readyState === WebSocket.OPEN);
      game.close();
    });
    await expect.poll(async () => (await livePaths(left.page)).filter(url => url === socketURL('/ws')).length).toBe(1);
    await expect.poll(() => restoredTeam(left.page), { timeout: 30_000 }).toEqual(before);
    const reconnectMs = Math.round(performance.now() - reconnectAt);
    expect(reconnectMs).toBeLessThan(30_000);
    expect(new Set(before.members.map(member => member.panelIndex)).size).toBe(before.members.length);
    expect(new Set(before.members.map(member => member.playerId)).size).toBe(before.members.length);
    await left.page.screenshot({ path: testInfo.outputPath('h2-08.png') });
    await returnPlayers(players);
    await expect(left.page.getByTestId('lobby-results')).toBeVisible();
    const metrics = { firstRenderedUpdateMs, reconnectMs, duplicateSlots: 0, scoreRestored: true };
    await testInfo.attach('journey metrics', { body: JSON.stringify(metrics), contentType: 'application/json' });
    console.log(JSON.stringify(metrics));
  } finally {
    await closePlayers(players);
  }
});

test('a shortened absolute lifetime expires and reconnects the lobby socket', async ({ browser }, testInfo) => {
  test.skip(!absoluteSeconds, 'Requires the shortened-lifetime fixture');
  const player = await openPlayer(browser, 'Expiry player');
  try {
    const start = performance.now();
    await player.page.evaluate(() => { window.__expirySocket = window.__multiplayerSockets[0]; });
    await expect.poll(() => player.page.evaluate(() => window.__expirySocket.readyState), {
      timeout: (absoluteSeconds + 5) * 1_000,
    }).toBe(3);
    expect(performance.now() - start).toBeLessThan((absoluteSeconds + 5) * 1_000);
    await expect.poll(() => livePaths(player.page)).toEqual([socketURL('/lobby/ws')]);
    await expect.poll(() => player.page.evaluate(() => window.__hudesApp.lobbyClient.state.connection)).toBe('connected');
    expect(await player.page.evaluate(() => window.__hudesApp.lobbyClient.state.snapshot.you.id)).toBe(player.id);
    await player.page.screenshot({ path: testInfo.outputPath('h2-09.png') });
  } finally {
    await closePlayers([player]);
  }
});
