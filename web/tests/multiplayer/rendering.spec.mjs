import { test, expect } from '@playwright/test';

test('solo and assigned multiplayer landscapes render identical pixels', async ({ page }) => {
  await page.goto('/?lobby=1');
  const snapshots = await page.evaluate(async () => {
    const { default: LandscapeView } = await import('/client/views/LandscapeView.js');
    const results = [];
    for (const owner of [0, 1, 2]) {
      const images = [];
      for (const teamMode of [false, true]) {
        document.body.innerHTML = '<div id="glContainer"></div>';
        const view = new LandscapeView(31, 3, { helpScreenIdx: -1 }, { teamMode, disableInitialTour: true });
        if (teamMode) view.configureTeam({ panelIndex: owner, config: { meshGridSize: 31, meshGrids: 3 } });
        else view.setSelectedGrid(owner);
        view.updateMeshGrids(Array.from({length:3}, (_,p) => Array.from({length:31}, (_,y) =>
          Array.from({length:31}, (_,x) => 2 + (x-15)*.02 + (y-15)**2 * .001*(p+1)))));
        await new Promise(resolve => requestAnimationFrame(resolve));
        view.composer.render();
        const image = view.renderer.domElement.toDataURL();
        if (image.length < 10000) throw new Error('Expected a visible landscape, not a blank canvas');
        images.push(image);
        view.destroy();
      }
      results.push({owner, identical: images[0] === images[1]});
    }
    return results;
  });
  expect(snapshots).toEqual([0,1,2].map(owner => ({owner,identical:true})));
});
