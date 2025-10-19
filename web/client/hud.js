export const CONTROL_GROUPS = [
  { icon: null, keys: ['W', 'A', 'S', 'D'], label: 'Move' },
  { icon: null, keys: ['⬆️', '⬇️', '⬅️', '➡️'], label: 'Rotate' },
  { icon: null, keys: ['Spacebar'], label: 'New Dims' },
  { icon: null, keys: ['Enter'], label: 'New Batch' },
  { icon: null, keys: ['⇧'], label: 'Cycle Plane' },
  { icon: null, keys: ['[', ']'], label: 'Step ±' },
  { icon: null, keys: [';'], label: 'Batch-size' },
  { icon: null, keys: ["'"], label: 'FP16/32' },
  { icon: null, keys: ['X'], label: 'Help' },
  { icon: null, keys: ['Q'], label: 'Q: Hold to quit' },
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
  const label = escapeHtml(group.label || '');
  return `
    <div class="control-group">
      ${icon}
      ${keys}
      <span class="label">${label}</span>
    </div>
  `;
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
  const statusMarkup = safeStatus
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
      </div>
      ${titleMarkup}
    </div>
  `;
};
