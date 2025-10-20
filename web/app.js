// import HudesClient from './client/HudesClient.js';

// const client = new HudesClient("localhost", 8765);
// client.runLoop();

//import KeyboardClient from './client/KeyboardClient.js';
import KeyboardClientGL from './client/KeyboardClientGL.js';

async function detectBackendHostPort() {
	const params = new URLSearchParams(window.location.search);
	const qpHost = params.get('host');
	const qpPort = params.get('port');
	const host = qpHost || window.location.hostname || 'localhost';
	if (qpPort) return { host, port: Number(qpPort) };

	// Probe API health endpoints to auto-pick between 10001/10002 and 8765/8766
	const proto = window.location.protocol === 'https:' ? 'https' : 'http';
	const candidates = [
		{ ws: 10001, http: 10002 },
		{ ws: 8765, http: 8766 },
	];

	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), 800);
	try {
		for (const c of candidates) {
			try {
				const res = await fetch(`${proto}://${host}:${c.http}/health`, { signal: controller.signal });
				if (res.ok) {
					clearTimeout(timeout);
					return { host, port: c.ws };
				}
			} catch {}
		}
	} finally {
		clearTimeout(timeout);
	}
	// Fallback
	return { host, port: 8765 };
}

(async function bootstrap() {
	const { host, port } = await detectBackendHostPort();
	const client = new KeyboardClientGL(host, port);
	if (typeof window !== 'undefined') {
		window.__hudesClient = client;
	}
	client.runLoop();
})();
