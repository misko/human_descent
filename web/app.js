import KeyboardClient from './client/KeyboardClient.js';
import KeyboardClientGL from './client/KeyboardClientGL.js';

async function detectBackendHostPort() {
	const params = new URLSearchParams(window.location.search);
	const qpHost = params.get('host');
	const qpPort = params.get('port');
	const host = qpHost || window.location.hostname || 'localhost';
	if (qpPort) return { host, port: Number(qpPort) };

	// Build-time override via Vite env vars (inlined at build)
	// Example: VITE_WS_PORT=10000 VITE_HTTP_PORT=10001 npm run build
	try {
		const wsEnv = Number(import.meta.env?.VITE_WS_PORT);
		const httpEnv = Number(import.meta.env?.VITE_HTTP_PORT);
		if (Number.isFinite(wsEnv) && wsEnv > 0) {
			// If only WS port is provided, assume health is WS+1 unless HTTP override set
			const http = Number.isFinite(httpEnv) && httpEnv > 0 ? httpEnv : (wsEnv + 1);
			return { host, port: wsEnv, httpPort: http };
		}
	} catch {}

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

function installModeToggle(currentMode) {
	if (typeof document === 'undefined') return;
	const header = document.querySelector('.site-header');
	if (!header) return;
	const existing = header.querySelector('.view-toggle');
	if (existing) {
		existing.remove();
	}
	const button = document.createElement('button');
	button.type = 'button';
	button.className = 'view-toggle';
	const targetMode = currentMode === '1d' ? '3d' : '1d';
	button.textContent = currentMode === '1d' ? 'Switch to 3D view' : 'Switch to 1D view';
	button.addEventListener('click', () => {
		const params = new URLSearchParams(window.location.search);
		params.set('mode', targetMode);
		window.location.search = params.toString();
	});
	header.appendChild(button);
}

(async function bootstrap() {
	const { host, port } = await detectBackendHostPort();
	const params = new URLSearchParams(window.location.search);
	const modeParam = (params.get('mode') || '').toLowerCase();
	const renderMode = modeParam === '1d' ? '1d' : '3d';
	const debugParam = params.get('debug');
	const debugEnabled = typeof debugParam === 'string' && /^(1|true|yes|on)$/i.test(debugParam);
	const altKeysEnabled = (() => {
		const flag = params.get('altkeys');
		return typeof flag === 'string' && /^(1|true|yes|on)$/i.test(flag);
	})();
	const gridSizeParam = Number(params.get('gridSize'));
	const gridSize = Number.isFinite(gridSizeParam) && gridSizeParam > 4 ? Math.floor(gridSizeParam) : undefined;
	const rowSpacingParam = params.get('rowSpacing');
	const rowSpacingParsed = rowSpacingParam == null ? undefined : Number(rowSpacingParam);
	const rowSpacingRaw =
		rowSpacingParsed == null || !Number.isFinite(rowSpacingParsed) ? undefined : rowSpacingParsed;
	const depthStepParam = params.get('depthStep');
	const depthStepParsed = depthStepParam == null ? undefined : Number(depthStepParam);
	const depthStep =
		depthStepParsed == null || !Number.isFinite(depthStepParsed) ? undefined : depthStepParsed;
	const distanceParam = params.get('cameraDistance');
	const distanceParsed = distanceParam == null ? undefined : Number(distanceParam);
	const cameraDistanceRaw =
		distanceParsed == null || !Number.isFinite(distanceParsed) || distanceParsed <= 0
			? undefined
			: distanceParsed;
	const rowSpacing = renderMode === '1d' ? (rowSpacingRaw ?? 1) : rowSpacingRaw;
	const cameraDistance = renderMode === '1d' ? (cameraDistanceRaw ?? 50) : cameraDistanceRaw;
	const client =
		renderMode === '1d'
			? new KeyboardClient(host, port, {
					renderMode: '1d',
					lossLines: 6,
					debug: debugEnabled,
					gridSize,
					rowSpacing,
					depthStep,
					cameraDistance,
					altKeys: altKeysEnabled,
			  })
			: new KeyboardClientGL(host, port, {
					renderMode: '3d',
					debug: debugEnabled,
					gridSize,
					rowSpacing,
					depthStep,
					cameraDistance,
			  });
	if (typeof window !== 'undefined') {
		window.__hudesClient = client;
		if (debugEnabled) {
			window.__hudesDebug = true;
		}
	}
	installModeToggle(renderMode);
	client.runLoop();
})();
