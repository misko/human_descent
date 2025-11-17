import { test, expect } from '@playwright/test';

const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const APP_ORIGIN = process.env.HUDES_APP_ORIGIN || 'http://localhost:6173';

const mobileUrl = `${APP_ORIGIN}/?mode=3d&mobile=1&host=${SERVER_HOST}&port=${SERVER_PORT}&help=off`;

async function bootstrapMobile(page) {
  await page.goto(mobileUrl);
  await page.waitForFunction(
    () => window.__hudesClient && window.__hudesClient.ControlType,
    { timeout: 10000 },
  );
  await page.waitForFunction(
    () => window.__hudesClient?.view?.impl?.mobileMode === true,
    { timeout: 10000 },
  );
}

test.describe('Mobile 3D mode', () => {
  test('renders single grid and mobile HUD controls', async ({ page }) => {
    await bootstrapMobile(page);

    const mobileFlag = await page.evaluate(() => Boolean(window.__hudesIsMobile));
    expect(mobileFlag).toBeTruthy();

    const gridCount = await page.evaluate(
      () => window.__hudesClient?.view?.impl?.numGrids ?? -1,
    );
    expect(gridCount).toBe(1);

    await expect(page.locator('#sideContainer')).toBeHidden();

    const hud = page.locator('#bottomTextContainer');
    await expect(hud.locator('[data-hud-action="step-plus"]')).toBeVisible();
    await expect(hud.locator('[data-hud-action="step-minus"]')).toBeVisible();
    await expect(hud.locator('[data-hud-action="toggle-fp"]')).toBeVisible();
    const hudText = await hud.innerText();
    expect(/WASD/i.test(hudText)).toBeFalsy();

    const matrixBox = await page.locator('#confusionMatrixChart').boundingBox();
    const panelBox = await page.locator('#mobileActionPanel').boundingBox();
    expect(matrixBox).not.toBeNull();
    expect(panelBox).not.toBeNull();
    if (matrixBox && panelBox) {
      expect(panelBox.y).toBeGreaterThanOrEqual(matrixBox.y + matrixBox.height - 2);
    }
  });

  test('mobile controls dispatch actions and analog steps', async ({ page }) => {
    await bootstrapMobile(page);

    const dimsBtn = page.locator('[data-hud-action="next-dims"]');
    const batchBtnHud = page.locator('[data-hud-action="next-batch"]');
    await expect(dimsBtn).toBeVisible();
    await expect(batchBtnHud).toBeVisible();

    await page.evaluate(() => {
      const client = window.__hudesClient;
      window.__dimsCalls = 0;
      window.__batchCalls = 0;
      if (client) {
        const origDims = client.getNextDims?.bind(client);
        client.getNextDims = (...args) => {
          window.__dimsCalls += 1;
          return origDims?.(...args);
        };
        const origBatch = client.getNextBatch?.bind(client);
        client.getNextBatch = (...args) => {
          window.__batchCalls += 1;
          return origBatch?.(...args);
        };
      }
    });

    await dimsBtn.click();
    await page.waitForFunction(() => (window.__dimsCalls || 0) > 0, { timeout: 5000 });

    await batchBtnHud.click();
    await page.waitForFunction(() => (window.__batchCalls || 0) > 0, { timeout: 5000 });

    const batchPanelButton = page.locator('#mobileActionPanel button[data-mobile-action="batch"]');
    await expect(batchPanelButton).toBeVisible();
    const initialLabel = await batchPanelButton.innerText();
    await batchPanelButton.click();
    await page.waitForFunction((label) => {
      const btn = document.querySelector('#mobileActionPanel button[data-mobile-action="batch"]');
      return btn && btn.innerText !== label;
    }, initialLabel, { timeout: 5000 }).catch(() => {});

    await page.evaluate(() => {
      window.__hudesClient?.zeroDimsAndStepsOnCurrentDims?.();
    });
    await page.evaluate(() => {
      window.__hudesClient?.__applyTouchVector?.({ x: 0, y: -1 });
    });
    await page.waitForFunction(() => {
      const dims = window.__hudesClient?.dimsAndStepsOnCurrentDims;
      if (!Array.isArray(dims)) return false;
      return dims.some((value) => Math.abs(value) > 0);
    }, { timeout: 5000 });
  });
});
