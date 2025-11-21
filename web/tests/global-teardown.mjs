const SERVER_HOST = process.env.HUDES_HOST || 'localhost';
const SERVER_PORT = Number(process.env.HUDES_PORT || '10001');
const SPEED_SECONDS = Number(process.env.HUDES_SPEED_RUN_SECONDS || '5');

const proto = process.env.HUDES_APP_ORIGIN?.startsWith('https') ? 'https' : 'http';
const apiBase = `${proto}://${SERVER_HOST}:${SERVER_PORT + 1}/api`;

async function fetchScores(offset = 0, limit = 1000) {
  const resp = await fetch(`${apiBase}/highscores?offset=${offset}&limit=${limit}`);
  if (!resp.ok) throw new Error(`Failed to fetch highscores: ${resp.status} ${resp.statusText}`);
  return resp.json();
}

export default async function globalTeardown() {
  if (!Number.isFinite(SPEED_SECONDS)) return;
  try {
    const limit = 1000;
    let offset = 0;
    let totalDeleted = 0;
    // Iterate until fewer than the limit entries are returned
    for (;;) {
      const rows = await fetchScores(offset, limit);
      if (!Array.isArray(rows) || rows.length === 0) break;
      // Delete matching entries
      const candidates = rows.filter((r) => {
        const duration = Number(r.duration);
        if (duration === 120) return false; // safety: never delete real 120s runs
        return duration === SPEED_SECONDS && Number(r.id) > 0;
      });
      await Promise.all(
        candidates.map(async (r) => {
          try {
            await fetch(`${apiBase}/highscores?id=${r.id}`, { method: 'DELETE' });
          } catch {}
        })
      );
      totalDeleted += candidates.length;
      if (rows.length < limit) break;
      offset += limit;
    }
    if (totalDeleted > 0) {
      // eslint-disable-next-line no-console
      console.log(`[global-teardown] Deleted ${totalDeleted} high-score rows with duration ${SPEED_SECONDS}s`);
    }
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('[global-teardown] Failed to purge test high scores:', err);
  }
}
