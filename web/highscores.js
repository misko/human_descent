async function detectBackend() {
  const params = new URLSearchParams(window.location.search);
  const host = params.get('host') || window.location.hostname || 'localhost';
  const proto = window.location.protocol === 'https:' ? 'https' : 'http';
  const qpPort = params.get('port');
  if (qpPort) return `${proto}://${host}:${Number(qpPort) + 1}/api`;

  const candidates = [10001, 8765];
  for (const wsPort of candidates) {
    try {
      const httpPort = wsPort + 1;
      const res = await fetch(`${proto}://${host}:${httpPort}/health`, { method: 'GET' });
      if (res.ok) return `${proto}://${host}:${httpPort}/api`;
    } catch {}
  }
  // Fallback
  return `${proto}://${host}:${10002}/api`;
}

async function load(offset = 0, limit = 1000) {
  const api = await detectBackend();
  const status = document.getElementById('status');
  status.textContent = 'Loading...';
  const resp = await fetch(`${api}/highscores?offset=${offset}&limit=${limit}`);
  if (!resp.ok) {
    status.textContent = 'Failed to load scores';
    return;
  }
  const rows = await resp.json();
  status.textContent = `Loaded ${rows.length} rows`;
  const tbody = document.querySelector('#scoresTable tbody');
  tbody.innerHTML = '';
  let rank = offset + 1;
  for (const r of rows) {
    const tr = document.createElement('tr');
    const score = typeof r.score === 'number' ? r.score.toFixed(4) : r.score;
    tr.innerHTML = `
      <td>${rank}</td>
      <td>${r.name}</td>
      <td>${score}</td>
      <td>${r.ts}</td>
      <td>${r.duration}</td>
      <td>${r.requestIdx ?? ''}</td>
    `;
    tbody.appendChild(tr);
    rank += 1;
  }
}

(function init() {
  const offsetInput = document.getElementById('offset');
  const limitInput = document.getElementById('limit');
  document.getElementById('loadBtn').addEventListener('click', () => {
    const o = Math.max(0, parseInt(offsetInput.value || '0', 10));
    const l = Math.max(1, Math.min(10000, parseInt(limitInput.value || '1000', 10)));
    load(o, l);
  });
  load(0, 1000);
})();
