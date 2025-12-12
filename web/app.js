import KeyboardClient from './client/KeyboardClient.js';
import KeyboardClientGL from './client/KeyboardClientGL.js';
import { detectMobileMode, setMobileFlag } from './mobile.js';

const DEFAULT_WS_PATH = import.meta.env?.VITE_WS_PATH ?? '/ws';
const DEFAULT_HTTP_API = import.meta.env?.VITE_HTTP_API ?? '/api';
const ensureLeadingSlash = (p) => (!p ? '' : p.startsWith('/') ? p : `/${p}`);

async function detectBackend() {
	const params = new URLSearchParams(window.location.search);
	const origin = typeof window !== 'undefined' ? window.location.origin : '';
	const proto = typeof window !== 'undefined' && window.location?.protocol === 'https:' ? 'https' : 'http';
	const wsOrigin = origin ? origin.replace(/^http/i, 'ws') : `${proto === 'https' ? 'wss' : 'ws'}://localhost`;
	const qpHost = params.get('host');
	const qpPortRaw = params.get('port');
	const qpPort = qpPortRaw ? Number(qpPortRaw) : undefined;
	const hasPort = Number.isFinite(qpPort);
	const qpWs = params.get('ws') || params.get('wsUrl');
	const qpApi = params.get('api') || params.get('apiBase');
	const envWsUrl = import.meta.env?.VITE_WS_URL;
	const envWsPortRaw = import.meta.env?.VITE_WS_PORT;
	const envWsPort = envWsPortRaw ? Number(envWsPortRaw) : undefined;
	const envWsPath = import.meta.env?.VITE_WS_PATH;
	const envHttpApi = import.meta.env?.VITE_HTTP_API;
	const envHttpPortRaw = import.meta.env?.VITE_HTTP_PORT;
	const envHttpPort = envHttpPortRaw ? Number(envHttpPortRaw) : undefined;
	const wsPath = ensureLeadingSlash(envWsPath || DEFAULT_WS_PATH);
	const apiPath = ensureLeadingSlash(envHttpApi || DEFAULT_HTTP_API);
	const wsScheme = proto === 'https' ? 'wss' : 'ws';
	const host = qpHost || (typeof window !== 'undefined' ? window.location.hostname : 'localhost') || 'localhost';

	const normalizeWs = (value) => {
		if (!value) return null;
		if (value.startsWith('ws://') || value.startsWith('wss://')) return value;
		if (value.startsWith('http://') || value.startsWith('https://')) return value.replace(/^http/i, 'ws');
		if (value.startsWith('/')) return `${wsOrigin}${value}`;
		return `${wsOrigin}/${value.replace(/^\/+/, '')}`;
	};

	const normalizeHttp = (value) => {
		if (!value) return null;
		if (value.startsWith('http://') || value.startsWith('https://')) return value;
		if (value.startsWith('/')) return `${origin}${value}`;
		return `${origin}/${value.replace(/^\/+/, '')}`;
	};

	let wsUrl = normalizeWs(qpWs) || null;
	let apiBase = normalizeHttp(qpApi) || null;

	if (!wsUrl && (qpHost || hasPort)) {
		const portPart = hasPort ? `:${qpPort}` : '';
		wsUrl = `${wsScheme}://${host}${portPart}${wsPath}`;
		if (!apiBase) {
			const httpPort = hasPort
				? qpPort + 1
				: envHttpPort ?? (envWsPort ? envWsPort + 1 : undefined);
			if (httpPort) {
				apiBase = `${proto}://${host}:${httpPort}${apiPath}`;
			}
		}
	}

	if (!wsUrl && envWsUrl) {
		wsUrl = normalizeWs(envWsUrl);
	}

	if (!wsUrl && envWsPort) {
		wsUrl = `${wsScheme}://${host}:${envWsPort}${wsPath}`;
		if (!apiBase) {
			const httpPort = envHttpPort ?? envWsPort + 1;
			apiBase = `${proto}://${host}:${httpPort}${apiPath}`;
		}
	}

	if (!wsUrl) {
		wsUrl = `${wsOrigin}${wsPath}`;
	}

	if (!apiBase && envHttpApi) {
		apiBase = normalizeHttp(envHttpApi);
	}

	if (!apiBase) {
		apiBase = `${origin}${apiPath}`;
	}

	return { wsUrl, apiBase, host, port: hasPort ? qpPort : envWsPort };
}

function installModeToggle(currentMode) {
	if (typeof document === 'undefined') return;
	const header = document.querySelector('.site-header');
	if (!header) return;
	let container = header.querySelector('.header-controls');
	if (!container) {
		container = document.createElement('div');
		container.className = 'header-controls';
		header.appendChild(container);
	}
	container.innerHTML = '';

	const makeToggleButton = (label, onClick, extraClass = '') => {
		const btn = document.createElement('button');
		btn.type = 'button';
		btn.className = `header-pill ${extraClass}`.trim();
		btn.textContent = label;
		btn.addEventListener('click', onClick);
		return btn;
	};

	const isMobile = typeof window !== 'undefined' && window.__hudesIsMobile;
	if (!isMobile) {
		const targetMode = currentMode === '1d' ? '3d' : '1d';
		const viewButton = makeToggleButton(currentMode === '1d' ? 'Switch to 3D view' : 'Switch to 1D view', () => {
			const params = new URLSearchParams(window.location.search);
			params.set('mode', targetMode);
			window.location.search = params.toString();
		});
		container.appendChild(viewButton);
	}

	if (currentMode === '1d' && !isMobile) {
		const params = new URLSearchParams(window.location.search);
		const toggleFlag = (key) => {
			const current = params.get(key);
			const enabled = typeof current === 'string' && /^(1|true|yes|on)$/i.test(current);
			if (enabled) {
				params.delete(key);
			} else {
				params.set(key, '1');
			}
			window.location.search = params.toString();
		};

		const makeToggle = (label, key) =>
			makeToggleButton(`${label}: ${params.has(key) ? 'ON' : 'OFF'}`, () => toggleFlag(key), 'mode-toggle');

		container.appendChild(makeToggle('Alt Keys', 'altkeys'));
		container.appendChild(makeToggle('Alt 1D', 'alt1d'));
	}

	let status = header.querySelector('.connection-status');
	if (!status) {
		status = document.createElement('span');
		status.className = 'connection-status';
		header.appendChild(status);
	}
	status.dataset.state = 'connecting';
	status.textContent = 'Connecting…';

	return status;
}

(async function bootstrap() {
	const backend = await detectBackend();
	const { host, port, wsUrl, apiBase } = backend;
	const params = new URLSearchParams(window.location.search);
	const isMobile = detectMobileMode(params);
	setMobileFlag(isMobile);
	const modeParam = (params.get('mode') || '').toLowerCase();
	const renderMode = modeParam === '1d' && !isMobile ? '1d' : '3d';
	const debugParam = params.get('debug');
	const debugEnabled = typeof debugParam === 'string' && /^(1|true|yes|on)$/i.test(debugParam);
	const altKeysEnabled = (() => {
		const flag = params.get('altkeys');
		return typeof flag === 'string' && /^(1|true|yes|on)$/i.test(flag);
	})();
	const alt1dEnabled = (() => {
		const flag = params.get('alt1d');
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
                  alt1d: alt1dEnabled,
                  isMobile,
                  wsUrl,
                  httpBase: apiBase,
              })
          : new KeyboardClientGL(host, port, {
                  renderMode: '3d',
                  debug: debugEnabled,
                  gridSize,
                  rowSpacing,
                  depthStep,
                  cameraDistance,
                  isMobile,
                  wsUrl,
                  httpBase: apiBase,
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
