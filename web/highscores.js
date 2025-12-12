const DEFAULT_API_PATH = import.meta.env?.VITE_HTTP_API ?? '/api';
const ensureLeadingSlash = (p) => (p?.startsWith('/') ? p : `/${p ?? ''}`);

async function detectBackend() {
  const params = new URLSearchParams(window.location.search);
  const proto = window.location.protocol === 'https:' ? 'https' : 'http';
  const origin = window.location.origin;
  const qpApi = params.get('api') || params.get('apiBase');
  const qpHost = params.get('host');
  const qpPortRaw = params.get('port');
  const qpPort = qpPortRaw ? Number(qpPortRaw) : undefined;

  const toAbsoluteApi = (value) => {
    if (!value) return null;
    if (value.startsWith('http://') || value.startsWith('https://')) return value;
    if (value.startsWith('/')) return `${origin}${value}`;
    return `${origin}/${value.replace(/^\/+/, '')}`;
  };

  if (qpApi) {
    const api = toAbsoluteApi(qpApi);
    if (api) return api;
  }

  if (qpHost || qpPortRaw) {
    const host = qpHost || window.location.hostname || 'localhost';
    const httpPort = Number.isFinite(qpPort) ? qpPort + 1 : undefined;
    if (httpPort) return `${proto}://${host}:${httpPort}${ensureLeadingSlash(DEFAULT_API_PATH)}`;
    return `${proto}://${host}${ensureLeadingSlash(DEFAULT_API_PATH)}`;
  }

  const envApi = import.meta.env?.VITE_HTTP_API;
  if (envApi) {
    const api = toAbsoluteApi(envApi);
    if (api) return api;
  }

  return `${origin}${ensureLeadingSlash(DEFAULT_API_PATH)}`;
}

async function deleteScore(id) {
  if (!confirm('Are you sure you want to delete this score?')) return;
  const api = await detectBackend();
  try {
    const res = await fetch(`${api}/highscores?id=${id}`, { method: 'DELETE' });
    if (res.ok) {
      document.getElementById('loadBtn').click(); // Reload current view
    } else {
      alert('Failed to delete score');
    }
  } catch (e) {
    console.error(e);
    alert('Error deleting score');
  }
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
  status.textContent = `Loaded ${rows.length} rows (Offset: ${offset})`;
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
      <td><button class="delete-btn" data-id="${r.id}">Delete</button></td>
    `;
    tbody.appendChild(tr);
    rank += 1;
  }

  // Bind delete buttons
  document.querySelectorAll('.delete-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      deleteScore(e.target.dataset.id);
    });
  });
}

(function init() {
  const offsetInput = document.getElementById('offset');
  const limitInput = document.getElementById('limit');

  const getParams = () => {
    const o = Math.max(0, parseInt(offsetInput.value || '0', 10));
    const l = Math.max(1, Math.min(10000, parseInt(limitInput.value || '1000', 10)));
    return { o, l };
  };

  document.getElementById('loadBtn').addEventListener('click', () => {
    const { o, l } = getParams();
    load(o, l);
  });

  document.getElementById('prevBtn').addEventListener('click', () => {
    const { o, l } = getParams();
    const newOffset = Math.max(0, o - l);
    offsetInput.value = newOffset;
    load(newOffset, l);
  });

  document.getElementById('nextBtn').addEventListener('click', () => {
    const { o, l } = getParams();
    const newOffset = o + l;
    offsetInput.value = newOffset;
    load(newOffset, l);
  });

  load(0, 1000);
})();
