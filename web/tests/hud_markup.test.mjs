import { CONTROL_GROUPS, formatHudMarkup } from '../client/hud.js';

const status = 'val:1.23 bs:64 (f32) t:12.3s dims:12 sgd:99';
const markup = formatHudMarkup(status, CONTROL_GROUPS);

if (!markup.includes('hud-card')) {
  throw new Error('HUD markup missing card container');
}

if (!markup.includes('key-pill')) {
  throw new Error('HUD markup missing key pill elements');
}

if (!markup.includes(status)) {
  throw new Error('HUD markup does not contain status text');
}
