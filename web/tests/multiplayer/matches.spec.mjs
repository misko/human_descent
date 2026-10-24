import { test, expect } from '@playwright/test';
import { openPlayer, createTeam, startChallenge, launchPlayers, returnPlayers, closePlayers, assertTeamFrameInViewport } from '../helpers/multiplayer-browser.mjs';

for (const leftSize of [1, 2, 3]) {
  for (const rightSize of [1, 2, 3]) {
    test(`${leftSize}v${rightSize} invitations, ready check, shared clock, results and return`, async ({ browser }, testInfo) => {
      const players = [];
      try {
        for (let index = 0; index < leftSize + rightSize; index += 1) {
          players.push(await openPlayer(browser, `Player ${leftSize}-${rightSize}-${index}`));
        }
        const left = players.slice(0, leftSize);
        const right = players.slice(leftSize);
        const leftTeamId = await createTeam(left, `Left ${leftSize}-${rightSize}`);
        const rightTeamId = await createTeam(right, `Right ${leftSize}-${rightSize}`);
        await startChallenge(left[0], right[0], rightTeamId);
        await launchPlayers(players);
        const clocks = await Promise.all(players.map(({ page }) => page.evaluate(() => ({
          start: window.__hudesClient.team.startAt,
          deadline: window.__hudesClient.team.deadline,
          panels: window.__hudesClient.team.roster.length,
          landscapes: window.__hudesClient.view.impl.gridObjects.length,
          owner: window.__hudesClient.team.panelIndex,
          selected: window.__hudesClient.view.impl.getSelectedGrid(),
          scales: window.__hudesClient.view.impl.gridObjects.map(mesh => mesh.scale.x),
          session: window.__hudesClient.team.launch.sessionId,
        }))));
        expect(new Set(clocks.map(clock => clock.start)).size).toBe(1);
        expect(new Set(clocks.map(clock => clock.deadline)).size).toBe(1);
        expect(new Set(clocks.map(clock => clock.session)).size).toBe(2);
        expect(clocks.map(clock => clock.panels)).toEqual([
          ...Array(leftSize).fill(leftSize), ...Array(rightSize).fill(rightSize),
        ]);
        for (const clock of clocks) {
          expect(clock.landscapes).toBe(3);
          expect(clock.selected).toBe(clock.owner);
          expect(clock.scales).toEqual([0, 1, 2].map(panel => panel === clock.owner ? 3 : 1));
        }
        await left[0].page.keyboard.press('Shift');
        await left[0].page.evaluate(() => {
          const view = window.__hudesClient.view.impl;
          view.incrementSelectedGrid();
          view.selectGrid(2);
          view.updateMeshGrids();
        });
        expect(await left[0].page.evaluate(() => window.__hudesClient.view.impl.getSelectedGrid())).toBe(leftSize < 3 ? 2 : 0);
        if (leftSize === 1 && rightSize === 2) {
          await right[1].page.screenshot({ path: testInfo.outputPath('three-landscapes-owned-focus.png') });
        }
        if (leftSize === 3 && rightSize === 3) {
          const bounds = await left[0].page.evaluate(() => {
            const box = selector => {
              const r = document.querySelector(selector).getBoundingClientRect();
              return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, height: r.height };
            };
            return { hud: box('#teamGameStatus'), plots: box('#gameRoot .container'), side: box('#sideContainer') };
          });
          expect(bounds.hud.top).toBeGreaterThanOrEqual(bounds.plots.bottom + 6);
          expect(bounds.hud.right).toBeLessThan(bounds.side.left);
          expect(bounds.hud.height).toBeLessThanOrEqual(65);
          await left[0].page.keyboard.press('Space');
          for (const player of left) await expect(player.page.locator('#teamGameStatus [data-voted="true"]')).toHaveCount(1);
          for (const player of right) await expect(player.page.locator('#teamGameStatus [data-voted="true"]')).toHaveCount(0);
          for (let panel = 0; panel < 3; panel += 1) await assertTeamFrameInViewport(left[0].page, panel, panel === 0);
          await left[0].page.screenshot({ path: testInfo.outputPath('3v3-private-team-vote.png') });
          await left[1].page.keyboard.press('Space');
          await left[2].page.keyboard.press('Space');
          for (const player of left) await expect.poll(() => player.page.evaluate(() => window.__hudesClient.team.generation)).toBe(1);
          for (const player of right) expect(await player.page.evaluate(() => window.__hudesClient.team.generation)).toBe(0);
        }
        await returnPlayers(players);
        for (const [teamId, members] of [[leftTeamId, left], [rightTeamId, right]]) {
          const membership = await members[0].page.evaluate((id) => window.__hudesApp.lobbyClient.state.snapshot.teams.find(team => team.id === id).memberIds, teamId);
          expect(membership).toEqual(members.map(member => member.id));
        }
        await expect(left[0].page.getByTestId('lobby-results')).toBeVisible();
      } finally {
        await closePlayers(players);
      }
    });
  }
}

test('three-player votes persist through movement and rebuild; a second run disposes the first', async ({ browser }, testInfo) => {
  const players = [];
  try {
    for (let index = 0; index < 3; index += 1) players.push(await openPlayer(browser, `Co-op ${index}`));
    const teamId = await createTeam(players, 'Persistent team');
    for (let cycle = 0; cycle < 2; cycle += 1) {
      await players[0].page.getByTestId('lobby-start-speedrun').click();
      await launchPlayers(players);
      const page = players[0].page;
      await page.keyboard.press('Space');
      for (const player of players) {
        await expect(player.page.locator('#teamGameStatus [data-voted="true"]')).toHaveCount(1);
      }
      await page.keyboard.press('KeyW');
      await expect.poll(() => page.evaluate(() => window.__hudesClient.team.modelRevision)).toBeGreaterThan(0);
      await page.evaluate(() => {
        const renderer = window.__hudesClient.view.impl;
        renderer.updateMeshGrids();
      });
      for (const player of players) {
        for (let panel = 0; panel < 3; panel += 1) await assertTeamFrameInViewport(player.page, panel, panel === 0);
      }
      if (cycle === 0) {
        await page.screenshot({ path: testInfo.outputPath('unanimous-vote-green-frame.png') });
        await testInfo.attach('persistent green readiness frame', { path: testInfo.outputPath('unanimous-vote-green-frame.png'), contentType: 'image/png' });
      }
      await players[1].page.keyboard.press('Space');
      await players[2].page.keyboard.press('Space');
      for (const player of players) {
        await expect.poll(() => player.page.evaluate(() => window.__hudesClient.team.generation)).toBe(1);
        await expect(player.page.locator('#teamGameStatus [data-voted="true"]')).toHaveCount(0);
      }
      await returnPlayers(players);
      expect(await page.evaluate(id => window.__hudesApp.lobbyClient.state.snapshot.you.teamId === id, teamId)).toBe(true);
    }
  } finally {
    await closePlayers(players);
  }
});
