import { test, expect } from '@playwright/test';
import { openPlayer, createTeam, launchPlayers, returnPlayers, closePlayers } from '../helpers/multiplayer-browser.mjs';

test('desktop readiness, held keys and independent lobby/game reconnects', async ({ browser }, testInfo) => {
  const players = [];
  try {
    players.push(await openPlayer(browser, 'Desktop captain'));
    players.push(await openPlayer(browser, 'Desktop teammate'));
    players.push(await openPlayer(browser, 'Third desktop teammate'));
    const page = players[0].page;
    await createTeam(players, 'Across devices');
    await page.getByTestId('lobby-chat-input').fill('Space stays in chat');
    await page.getByTestId('lobby-chat-input').press('Space');
    await page.getByTestId('lobby-chat-send').click();
    await expect(players[2].page.getByTestId('lobby-chat-message')).toContainText('Space stays in chat');
    expect(await page.evaluate(() => window.__hudesClient)).toBeNull();
    await page.locator('#lobbyRoot').evaluate(root => { root.scrollTop = 0; });
    await page.screenshot({ path: testInfo.outputPath('multiplayer-lobby.png'), fullPage: true });
    await page.getByTestId('lobby-start-speedrun').click();
    await launchPlayers(players);
    await page.keyboard.down('Space');
    await expect(players[2].page.locator('#teamGameStatus [data-voted="true"]')).toHaveCount(1);
    await expect(players[2].page.locator('#teamGameStatus [data-team-player]')).toHaveCount(3);
    await players[2].page.screenshot({ path: testInfo.outputPath('desktop-team-readiness.png') });

    // Only the lobby socket is interrupted; gameplay remains admitted.
    await page.evaluate(() => window.__multiplayerSockets.find(socket => new URL(socket.url).pathname === '/lobby/ws' && socket.readyState === 1).close());
    await page.keyboard.press('KeyW');
    await expect.poll(() => page.evaluate(() => window.__hudesClient.team.modelRevision)).toBeGreaterThan(0);
    await expect.poll(() => page.evaluate(() => window.__hudesApp.lobbyClient.state.connection)).toBe('connected');
    expect(await page.evaluate(() => window.__multiplayerSockets.filter(socket => new URL(socket.url).pathname === '/ws' && socket.readyState === 1).length)).toBe(1);

    // Only one game socket is interrupted; the independent chat client works.
    const teammate = players[1].page;
    await teammate.evaluate(() => window.__multiplayerSockets.find(socket => new URL(socket.url).pathname === '/ws' && socket.readyState === 1).close());
    await teammate.evaluate(() => window.__hudesApp.lobbyClient.command('chat.send', { text: 'Chat survives a game reconnect' }));
    await expect.poll(() => teammate.evaluate(() => window.__hudesClient.team.inputEnabled)).toBe(true);
    await expect(teammate.locator('#teamGameStatus [data-voted="true"]')).toHaveCount(1);
    await teammate.keyboard.press('Space');
    await players[2].page.evaluate(() => window.__hudesClient.getNextDims());
    await expect.poll(() => page.evaluate(() => window.__hudesClient.team.generation)).toBe(1);
    await page.evaluate(() => window.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space', key: ' ', repeat: true })));
    await expect(page.locator('#teamGameStatus [data-voted="true"]')).toHaveCount(0);
    await page.keyboard.up('Space');
    await returnPlayers(players);
    await expect(page.getByTestId('lobby-chat-message').filter({ hasText: 'Chat survives a game reconnect' })).toBeVisible();
  } finally {
    await closePlayers(players);
  }
});

for (const override of ['', '&mobile=0']) {
  test(`mobile multiplayer entry is blocked before connecting${override}`, async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true,
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148',
    });
    try {
      const page = await context.newPage();
      const connections = [];
      page.on('websocket', socket => connections.push(new URL(socket.url()).pathname));
      await page.goto(`${process.env.HUDES_APP_ORIGIN}/?lobby=1${override}`);
      await expect(page.getByTestId('multiplayer-desktop-required')).toBeVisible();
      await expect(page.getByTestId('desktop-required-solo')).toHaveAttribute('href', /^(?!.*lobby=).*$/);
      await expect(page.locator('.header-lobby')).toBeHidden();
      expect(await page.evaluate(() => window.__hudesApp.lobbyClient)).toBeNull();
      expect(await page.evaluate(() => window.__hudesClient)).toBeNull();
      expect(connections.filter(path => ['/ws', '/lobby/ws'].includes(path))).toEqual([]);
    } finally {
      await context.close();
    }
  });
}
