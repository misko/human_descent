// Public setup values only; gameplay receives the authoritative protobuf state.
export function normalizeSettings(value = {}) {
  const batchSize = value.batchSize ?? 32;
  const maxDimensions = value.maxDimensions ?? null;
  if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 512) throw new Error('Batch size must be an integer from 1 to 512.');
  if (maxDimensions !== null && (!Number.isInteger(maxDimensions) || maxDimensions < 6 || maxDimensions > 6006 || maxDimensions % 6 !== 0)) {
    throw new Error('Dimension limit must be a multiple of 6 from 6 to 6006, or unlimited.');
  }
  return { batchSize, maxDimensions };
}

export function settingsLabel(value) {
  const settings = normalizeSettings(value);
  const switches = settings.maxDimensions / 6 - 1;
  return `Batch size ${settings.batchSize} · ${settings.maxDimensions === null ? 'Unlimited dimensions' : `${settings.maxDimensions} dimensions (${switches} ${switches === 1 ? 'switch' : 'switches'})`}`;
}
