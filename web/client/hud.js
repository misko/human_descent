const SHARED_COMMANDS = [
  { icon: null, keys: ['SPACE'], label: 'New Dims' },
  { icon: null, keys: ['Enter'], label: 'New Batch' },
  { icon: null, keys: ['Del'], label: 'SGD Step' },
  { icon: null, keys: ['Z'], label: '<span class="speed-run-label">SPEED RUN 🔥</span>' },
  { icon: null, keys: ['⇧','🐁'], label: 'Cycle Plane' },
  { icon: null, keys: ['[', ']'], label: 'Step ±' },
  { icon: null, keys: [';'], label: 'Batch-size' },
  { icon: null, keys: ["'"], label: 'FP16/32' },
  { icon: null, keys: ['Y'], label: 'Top 10' },
  { icon: null, keys: ['X'], label: 'Help' },
];

export const CONTROL_GROUPS = [
  { icon: null, keys: ['W', 'A', 'S', 'D', 'Scroll'], label: 'Move' },
  { icon: null, keys: ['⬆️', '⬇️', '⬅️', '➡️', '🐁'], label: 'Rotate' },
  ...SHARED_COMMANDS,
];

export const TEAM_CONTROL_GROUPS = [
  { keys: ['W', 'A', 'S', 'D', 'Scroll'], label: 'Move selected landscape' },
  { keys: ['Shift', 'Click'], label: 'Select own or unassigned landscape' },
  { keys: ['SPACE'], label: 'Request New Dims (everyone)' },
  { keys: ['[', ']'], label: 'Shared Step ±' },
  { keys: ['Enter'], label: 'Shared Batch' },
  { keys: ['⬆️', '⬇️', '⬅️', '➡️', '🐁'], label: 'Rotate' },
  { keys: ['X'], label: 'Team Help' },
];

export const MOBILE_CONTROL_GROUPS = [];

export const MOBILE_HUD_BUTTONS = [
  { action: 'show-top', label: 'Top 10' },
];

export const HUD_TITLE = '🧠 Human Descent: MNIST';

export const escapeHtml = (value) => {
  if (typeof value !== 'string') {
    return '';
  }
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
};

const formatKeyPill = (key) =>
  `<span class="key-pill" role="presentation">${escapeHtml(key)}</span>`;

const formatControlGroup = (group) => {
  const icon = group.icon
    ? `<span class="group-icon" role="presentation">${escapeHtml(group.icon)}</span>`
    : '';
  const keys = (group.keys || []).map(formatKeyPill).join('');
  const label = group.label && group.label.includes('<span')
    ? group.label
    : escapeHtml(group.label || '');
  return `
    <div class="control-group">
      ${icon}
      ${keys}
      <span class="label">${label}</span>
    </div>
  `;
};

const formatHudButtons = (buttons = []) => {
  if (!Array.isArray(buttons) || buttons.length === 0) {
    return '';
  }
  const items = buttons
    .map(({ action, label }) => {
      const safeAction = action ? ` data-hud-action="${escapeHtml(action)}"` : '';
      const safeLabel = escapeHtml(label ?? '');
      return `<button type="button"${safeAction}>${safeLabel}</button>`;
    })
    .join('');
  return `<div class="hud-inline-actions">${items}</div>`;
};

export const formatHudMarkup = (
  statusText = '',
  controlGroups = CONTROL_GROUPS,
  titleOrOptions = {},
) => {
  const options =
    typeof titleOrOptions === 'string'
      ? { title: titleOrOptions }
      : titleOrOptions || {};

  const safeStatus = escapeHtml(statusText ?? '');
  const safeTitle = escapeHtml(options.title ?? '');
  const separator = '<span class="separator">•</span>';
  const controlsMarkup = controlGroups.map(formatControlGroup).join(separator);
  const buttonMarkup = formatHudButtons(options.buttons);
  const statusMarkup = options.statusHtml
    ? `<div class="hud-status">${options.statusHtml}</div>`
    : safeStatus
    ? `<div class="hud-status">${safeStatus}</div>`
    : '';
  const titleMarkup = safeTitle
    ? `<div class="hud-title" aria-hidden="true">${safeTitle}</div>`
    : '';
  return `
    <div class="hud-card">
      ${statusMarkup}
      <div class="hud-controls">
        ${controlsMarkup}
        ${buttonMarkup}
      </div>
      ${titleMarkup}
      <div class="hud-mobile-actions">
        <button type="button" data-hud-action="next-dims">New Dims</button>
        <button type="button" data-hud-action="next-batch">New Batch</button>
      </div>
    </div>
  `;
};
