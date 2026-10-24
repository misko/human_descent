import { test, expect } from '@playwright/test';
import { openPlayer, createTeam, launchPlayers, returnPlayers, closePlayers } from '../helpers/multiplayer-browser.mjs';

test('captain settings are accepted by both teams and each team has an independent switch budget', async ({ browser }, info) => {
  const players = [];
  try {
    for (let i = 0; i < 3; i++) players.push(await openPlayer(browser, `Settings ${i}`));
    await createTeam(players.slice(0, 2), 'Custom duo');
    const rightId = await createTeam(players.slice(2), 'Custom solo');
    const [captain, member, opponent] = players;
    await expect(member.page.getByTestId('lobby-batch-size')).toHaveCount(0);
    await captain.page.getByTestId('lobby-batch-size').fill('64');
    await captain.page.getByTestId('lobby-max-dimensions').fill('12');
    await captain.page.getByTestId(`lobby-challenge-${rightId}`).click();
    await expect(opponent.page.getByTestId('lobby-challenge-settings')).toHaveText('Batch size 64 · 12 dimensions (1 switch)');
    await captain.page.screenshot({path:info.outputPath('match-setup.png')});
    // Changing the draft cannot change the challenge already offered.
    await captain.page.getByTestId('lobby-batch-size').fill('128');
    await opponent.page.locator('[data-testid^="lobby-challenge-accept-"]').click();
    for (const {page} of players) await expect(page.getByTestId('lobby-confirmed-settings')).toContainText('Batch size 64');
    await launchPlayers(players);
    for (const {page} of players) await expect(page.getByTestId('team-switches-remaining')).toHaveText('Switches left: 1');
    await captain.page.keyboard.press('Space');
    await expect(member.page.locator('#teamGameStatus [data-voted="true"]')).toHaveCount(1);
    await expect(captain.page.getByTestId('team-switches-remaining')).toContainText('left: 1');
    await member.page.keyboard.press('Space');
    for (const {page} of players.slice(0,2)) {
      await expect(page.getByTestId('team-switches-remaining')).toContainText('left: 0');
      await page.waitForFunction(() => window.__hudesClient.team.updating === false);
    }
    await expect(opponent.page.getByTestId('team-switches-remaining')).toContainText('left: 1');
    await captain.page.keyboard.press('Space');
    expect(await captain.page.evaluate(() => window.__hudesClient.team.generation)).toBe(1);
    await captain.page.keyboard.press('KeyW');
    await expect.poll(() => captain.page.evaluate(() => window.__hudesClient.team.modelRevision)).toBeGreaterThan(0);
    await captain.page.screenshot({path:info.outputPath('no-switches-remaining.png')});
    await returnPlayers(players);
  } finally { await closePlayers(players); }
});

test('speedrun can use only the starting directions and rejects invalid setup', async ({ browser }) => {
  const players = [await openPlayer(browser, 'Fixed directions')];
  try {
    const {page} = players[0];
    await createTeam(players, 'Stay here');
    await page.getByTestId('lobby-max-dimensions').fill('7');
    await page.getByTestId('lobby-start-speedrun').click();
    await expect(page.getByTestId('lobby-error')).toContainText('multiple of 6');
    await page.getByTestId('lobby-batch-size').fill('17');
    await page.getByTestId('lobby-max-dimensions').fill('6');
    await page.getByTestId('lobby-start-speedrun').click();
    await launchPlayers(players);
    await expect(page.getByTestId('team-switches-remaining')).toHaveText('Switches left: 0');
    await page.keyboard.press('Space');
    expect(await page.evaluate(() => window.__hudesClient.team.generation)).toBe(0);
    await page.keyboard.press('KeyW');
    await expect.poll(() => page.evaluate(() => window.__hudesClient.team.modelRevision)).toBeGreaterThan(0);
    await returnPlayers(players);
  } finally { await closePlayers(players); }
});
