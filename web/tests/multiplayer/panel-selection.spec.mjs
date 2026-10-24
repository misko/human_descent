import { test, expect } from '@playwright/test';
import { openPlayer, createTeam, launchPlayers, returnPlayers, closePlayers, assertTeamFrameInViewport } from '../helpers/multiplayer-browser.mjs';

async function captureMoves(page) {
  await page.evaluate(() => {
    const client = window.__hudesClient;
    const send = client.sendQ.bind(client);
    client.selectionTestMoves = [];
    client.sendQ = message => {
      if (message.dimsAndSteps) client.selectionTestMoves.push(message.dimsAndSteps.map(step => step.dim).sort());
      return send(message);
    };
  });
}

async function clickPanel(page, panel, button = 'left') {
  const point = await page.evaluate(index => {
    const view = window.__hudesClient.view.impl;
    view.scene.updateMatrixWorld(true);
    view.camera.updateMatrixWorld(true);
    const mesh = view.gridObjects[index];
    const canvas = view.getCanvasElement();
    const rect = canvas.getBoundingClientRect();
    const positions = mesh.geometry.attributes.position;
    const indices = mesh.geometry.index;
    // Pick an actual visible triangle; a curved surface's bounding-box center
    // can project outside its silhouette or under a chart overlay.
    for (let offset = 0; offset < indices.count; offset += 3) {
      const center = view.camera.position.clone().set(0, 0, 0);
      for (let vertex = 0; vertex < 3; vertex++) {
        center.add(view.camera.position.clone().fromBufferAttribute(positions, indices.getX(offset + vertex)));
      }
      center.divideScalar(3).applyMatrix4(mesh.matrixWorld).project(view.camera);
      view.pointer.set(center.x, center.y);
      view.raycaster.setFromCamera(view.pointer, view.camera);
      const hit = view.raycaster.intersectObjects(view.gridObjects, false)[0];
      const point = { x: rect.left + (center.x + 1) * rect.width / 2, y: rect.top + (1 - center.y) * rect.height / 2 };
      if (hit?.object === mesh && document.elementFromPoint(point.x, point.y) === canvas) return point;
    }
    throw new Error(`No clickable triangle on panel ${index}`);
  }, panel);
  await page.mouse.click(point.x, point.y, { button });
}

const selected = page => page.evaluate(() => window.__hudesClient.view.getSelectedGrid());

test('one-player team uses solo selection controls without spending its exhausted budget', async ({ browser }) => {
  const players = [await openPlayer(browser, 'All landscapes')];
  const { page } = players[0];
  try {
    await createTeam(players, 'Solo selection');
    await page.getByTestId('lobby-max-dimensions').fill('6');
    await page.getByTestId('lobby-start-speedrun').click();
    await launchPlayers(players);
    await captureMoves(page);
    await page.keyboard.press('ShiftRight');
    expect(await selected(page)).toBe(1);
    await page.keyboard.press('KeyW');
    await expect.poll(() => page.evaluate(() => window.__hudesClient.team.modelRevision)).toBe(1);
    expect(await page.evaluate(() => window.__hudesClient.selectionTestMoves.at(-1))).toEqual([2, 3]);
    await clickPanel(page, 2);
    expect(await selected(page)).toBe(2);
    await page.mouse.wheel(0, 80);
    await expect.poll(() => page.evaluate(() => window.__hudesClient.team.modelRevision)).toBeGreaterThan(1);
    expect(await page.evaluate(() => window.__hudesClient.selectionTestMoves.at(-1))).toEqual([4, 5]);
    await clickPanel(page, 2, 'right');
    expect(await selected(page)).toBe(0);
    await page.keyboard.press('ShiftLeft');
    expect(await selected(page)).toBe(2);
    await page.keyboard.press('Enter');
    await expect.poll(() => page.evaluate(() => window.__hudesClient.team.batchIdx)).toBe(1);
    expect(await selected(page)).toBe(2);
    expect(await page.evaluate(() => window.__hudesClient.team.generation)).toBe(0);
    await expect(page.getByTestId('team-switches-remaining')).toHaveText('Switches left: 0');
    await returnPlayers(players);
  } finally { await closePlayers(players); }
});

test('two players share the third landscape while preserving ownership, votes and reconnect focus', async ({ browser }, info) => {
  const players = [await openPlayer(browser, 'Shared captain'), await openPlayer(browser, 'Shared teammate')];
  const [captain, teammate] = players.map(player => player.page);
  try {
    await createTeam(players, 'Shared landscape');
    await captain.getByTestId('lobby-max-dimensions').fill('12');
    await captain.getByTestId('lobby-start-speedrun').click();
    await launchPlayers(players);
    for (const { page } of players) {
      await captureMoves(page);
      await page.keyboard.press('ShiftRight');
      expect(await selected(page)).toBe(2);
      expect(await page.evaluate(() => window.__hudesClient.view.impl.gridObjects.map(mesh => mesh.scale.x))).toEqual([1, 1, 3]);
    }
    await clickPanel(captain, 1);
    expect(await selected(captain)).toBe(2);
    await clickPanel(teammate, 0);
    expect(await selected(teammate)).toBe(2);
    await Promise.all(players.map(({ page }) => page.keyboard.press('KeyW')));
    for (const { page } of players) {
      await expect.poll(() => page.evaluate(() => window.__hudesClient.team.modelRevision)).toBe(2);
      expect(await page.evaluate(() => window.__hudesClient.selectionTestMoves.at(-1))).toEqual([4, 5]);
    }
    await captain.keyboard.press('Space');
    await expect(teammate.locator('#teamGameStatus [data-voted="true"]')).toHaveCount(1);
    expect(await selected(captain)).toBe(2);
    await assertTeamFrameInViewport(captain, 0, true);
    await assertTeamFrameInViewport(captain, 2, false);
    await captain.screenshot({ path: info.outputPath('shared-third-landscape.png') });
    await teammate.keyboard.press('Space');
    for (const { page } of players) {
      await page.waitForFunction(() => window.__hudesClient.team.generation === 1 && window.__hudesClient.team.inputEnabled);
      expect(await selected(page)).toBe(2);
      await expect(page.getByTestId('team-switches-remaining')).toHaveText('Switches left: 0');
    }
    await teammate.evaluate(() => window.__hudesClient.socket.close());
    await expect.poll(() => teammate.evaluate(() => window.__hudesClient.team.inputEnabled && window.__hudesClient.socket.readyState === 1)).toBe(true);
    expect(await selected(teammate)).toBe(2);
    await teammate.keyboard.press('ShiftLeft');
    expect(await selected(teammate)).toBe(1);
    await teammate.keyboard.press('ShiftRight');
    await teammate.keyboard.press('KeyW');
    await expect.poll(() => teammate.evaluate(() => window.__hudesClient.team.modelRevision)).toBe(3);
    await returnPlayers(players);
  } finally { await closePlayers(players); }
});
