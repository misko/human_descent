import { expect } from '@playwright/test';

export async function openPlayer(browser, name, options = {}) {
  const context = await browser.newContext({ viewport: { width: 1100, height: 800 }, ...options });
  await context.addInitScript(() => {
    const Native = window.WebSocket;
    window.__multiplayerSockets = [];
    window.WebSocket = class extends Native {
      constructor(...args) {
        super(...args);
        window.__multiplayerSockets.push(this);
      }
    };
  });
  const page = await context.newPage();
  await page.goto(`${process.env.HUDES_APP_ORIGIN}/?lobby=1`);
  await page.getByTestId('lobby-name').fill(name);
  await page.getByTestId('lobby-enter').click();
  await expect.poll(() => page.evaluate(() => window.__hudesApp?.lobbyClient?.state.connection)).toBe('connected');
  const id = await page.evaluate(() => window.__hudesApp.lobbyClient.state.snapshot.you.id);
  return { context, page, id };
}

export async function createTeam(players, name) {
  const captain = players[0];
  await captain.page.getByTestId('lobby-team-name').fill(name);
  await captain.page.getByTestId('lobby-create-team').click();
  await expect(captain.page.getByTestId('lobby-own-team-name')).toHaveText(name);
  const teamId = await captain.page.evaluate(() => window.__hudesApp.lobbyClient.state.snapshot.you.teamId);
  for (const player of players.slice(1)) {
    await captain.page.getByTestId(`lobby-invite-${player.id}`).click();
    const invitation = player.page.locator('[data-testid^="lobby-invite-accept-"]');
    await invitation.click();
    await expect(player.page.getByTestId('lobby-own-team-name')).toHaveText(name);
    await expect(player.page.getByTestId('lobby-start-speedrun')).toHaveCount(0);
  }
  return teamId;
}

export async function startChallenge(left, right, rightTeamId) {
  await left.page.getByTestId(`lobby-challenge-${rightTeamId}`).click();
  await right.page.locator('[data-testid^="lobby-challenge-accept-"]').click();
}

export async function launchPlayers(players) {
  for (const player of players) await player.page.getByTestId('lobby-ready').click();
  await Promise.all(players.map(async ({ page }) => {
    await expect(page.locator('#gameRoot')).toBeVisible();
    await page.waitForFunction(() => window.__hudesClient?.team?.phase === 'running');
    await page.waitForFunction(() => window.__hudesClient?.team?.updating === false);
  }));
}

export async function returnPlayers(players) {
  await Promise.all(players.map(async ({ page }) => {
    await page.locator('#teamGameResults [data-action="return-lobby"]').click({ timeout: 20_000 });
    await expect(page.getByTestId('lobby')).toBeVisible();
    await expect(page.locator('#gameRoot canvas')).toHaveCount(0);
    await expect.poll(() => page.evaluate(() => window.__multiplayerSockets.filter(socket => new URL(socket.url).pathname === '/ws' && socket.readyState === 1).length)).toBe(0);
    await expect.poll(() => page.evaluate(() => window.__multiplayerSockets.filter(socket => new URL(socket.url).pathname === '/lobby/ws' && socket.readyState === 1).length)).toBe(1);
  }));
}

export async function closePlayers(players) {
  await Promise.all(players.map(player => player.context.close()));
}

export async function assertTeamFrameInViewport(page, panel, expectedVisible = null) {
  const frame = await page.evaluate((index) => {
    const renderer = window.__hudesClient.view.impl;
    const object = renderer.readyFrames[index];
    renderer.scene.updateMatrixWorld(true);
    renderer.camera.updateMatrixWorld(true);
    const coordinates = object.geometry.attributes.position.array;
    const points = [];
    for (let offset = 0; offset < coordinates.length; offset += 3) {
      points.push(renderer.camera.position.clone()
        .set(coordinates[offset], coordinates[offset + 1], coordinates[offset + 2])
        .applyMatrix4(object.matrixWorld).project(renderer.camera));
    }
    return {
      visible: object.visible, color: object.material.color.getHex(),
      minimumX: Math.min(...points.map(point => point.x)), maximumX: Math.max(...points.map(point => point.x)),
      minimumY: Math.min(...points.map(point => point.y)), maximumY: Math.max(...points.map(point => point.y)),
    };
  }, panel);
  if (expectedVisible !== null) expect(frame.visible).toBe(expectedVisible);
  expect(frame.color).toBe(0x36ed75);
  expect(frame.minimumX).toBeGreaterThanOrEqual(-1.02);
  expect(frame.maximumX).toBeLessThanOrEqual(1.02);
  expect(frame.minimumY).toBeGreaterThanOrEqual(-1.02);
  expect(frame.maximumY).toBeLessThanOrEqual(1.02);
}
