import { detectMobileMode, setMobileFlag, supportsMultiplayer } from './mobile.js';

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

async function gameOptions(backend, launchDescriptor) {
	const { host, port, wsUrl, apiBase } = backend;
	const params = new URLSearchParams(window.location.search);
	const isMobile = launchDescriptor ? false : detectMobileMode(params);
	setMobileFlag(isMobile);
	const modeParam = (params.get('mode') || '').toLowerCase();
	const renderMode = !launchDescriptor && modeParam === '1d' && !isMobile ? '1d' : '3d';
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
	return { host, port, renderMode, debugEnabled, options: {
        renderMode,
        ...(renderMode === '1d' ? { lossLines: 6, altKeys: altKeysEnabled, alt1d: alt1dEnabled } : {}),
        debug: debugEnabled,
        gridSize,
        rowSpacing,
        depthStep,
        cameraDistance,
        isMobile,
        wsUrl,
        httpBase: apiBase,
    } };
}

async function createGameClient(backend, launchDescriptor, onReturnToLobby, isCurrent = () => true) {
    const [{ default: KeyboardClient }, { default: KeyboardClientGL }] = await Promise.all([
        import('./client/KeyboardClient.js'),
        import('./client/KeyboardClientGL.js'),
    ]);
    const { host, port, renderMode, debugEnabled, options } = await gameOptions(backend, launchDescriptor);
    if (!isCurrent()) return null;
    const clientOptions = { ...options, launchDescriptor, onReturnToLobby };
    const client =
		renderMode === '1d'
          ? new KeyboardClient(host, port, clientOptions)
          : new KeyboardClientGL(host, port, clientOptions);
	if (typeof window !== 'undefined') {
		window.__hudesClient = client;
		if (debugEnabled) {
			window.__hudesDebug = true;
		}
	}
	installModeToggle(renderMode);
    if (launchDescriptor) {
        document.querySelector('.header-controls')?.replaceChildren();
        document.querySelector('.header-lobby').hidden = true;
    }
    try {
        client.runLoop();
    } catch (error) {
        client.destroy?.();
        if (window.__hudesClient === client) window.__hudesClient = null;
        throw error;
    }
    return client;
}

(async function bootstrap() {
    const backend = await detectBackend();
    const params = new URLSearchParams(window.location.search);
    const lobbyEntry = /^(1|true|yes|on)$/i.test(params.get('lobby') || '');
    const gameRoot = document.getElementById('gameRoot');
    const lobbyRoot = document.getElementById('lobbyRoot');
    // The query override selects the solo layout; it cannot grant multiplayer
    // access to a device detected as mobile by its actual capabilities.
    const mobileMultiplayer = !supportsMultiplayer();
    const lobbyLink = document.querySelector('.header-lobby');
    const lobbyEntryUrl = new URL(window.location.href);
    lobbyEntryUrl.searchParams.set('lobby', '1');
    lobbyLink.href = lobbyEntryUrl.href;
    lobbyLink.hidden = mobileMultiplayer;
    if (lobbyEntry && mobileMultiplayer) {
        gameRoot.replaceChildren();
        gameRoot.hidden = true;
        lobbyRoot.hidden = false;
        document.body.classList.add('hudes-lobby');
        const panel = document.createElement('section');
        panel.className = 'desktop-required';
        panel.dataset.testid = 'multiplayer-desktop-required';
        const title = document.createElement('h1');
        title.textContent = 'Multiplayer requires a desktop';
        const description = document.createElement('p');
        description.textContent = 'Open Human Descent on a desktop computer to play with a team. Solo play is available on this device.';
        const solo = document.createElement('a');
        const url = new URL(window.location.href);
        url.searchParams.delete('lobby');
        url.searchParams.delete('lobbyWs');
        solo.href = url.href;
        solo.textContent = 'Play solo';
        solo.dataset.testid = 'desktop-required-solo';
        panel.append(title, description, solo);
        lobbyRoot.replaceChildren(panel);
        window.__hudesClient = null;
        window.__hudesApp = { gameClient: null, lobbyClient: null, mobileRestricted: true };
        return;
    }
    const gameMarkup = gameRoot.innerHTML;
    let activeClient = null;
    let activeSessionId = null;
    let mountRevision = 0;
    let lobby = null;
    let disposed = false;

    function showLobby() {
        mountRevision += 1;
        const client = activeClient;
        activeClient = null;
        activeSessionId = null;
        window.__hudesClient = null;
        if (window.client === client) window.client = null;
        client?.destroy();
        gameRoot.replaceChildren();
        gameRoot.hidden = true;
        lobbyRoot.hidden = false;
        document.body.classList.add('hudes-lobby');
    }

    async function launchGame(descriptor) {
        if (disposed || (activeSessionId && activeSessionId === descriptor.sessionId)) return;
        activeClient?.destroy();
        activeClient = null;
        activeSessionId = descriptor.sessionId;
        const revision = ++mountRevision;
        gameRoot.innerHTML = gameMarkup;
        gameRoot.hidden = false;
        lobbyRoot.hidden = true;
        document.body.classList.remove('hudes-lobby');
        let client;
        try {
            client = await createGameClient(backend, descriptor, () => {
                if (!disposed && revision === mountRevision) showLobby();
            }, () => !disposed && revision === mountRevision);
        } catch (error) {
            if (disposed || revision !== mountRevision) return;
            showLobby();
            const message = document.createElement('p');
            message.className = 'app-error';
            message.setAttribute('role', 'alert');
            message.textContent = `Unable to open game: ${error.message}`;
            lobbyRoot.prepend(message);
            return;
        }
        if (!client) return;
        if (disposed || revision !== mountRevision) {
            client.destroy();
            return;
        }
        activeClient = client;
    }

    if (lobbyEntry) {
        showLobby();
        const { mountLobby } = await import('./lobby/index.js');
        const lobbyUrl = params.get('lobbyWs') || new URL('/lobby/ws', backend.wsUrl).href;
        lobby = mountLobby({
            root: lobbyRoot,
            wsUrl: lobbyUrl,
            onLaunch: launchGame,
            onSolo: () => {
                const url = new URL(window.location.href);
                url.searchParams.delete('lobby');
                url.searchParams.delete('lobbyWs');
                window.location.assign(url.href);
            },
        });
    } else {
        activeClient = await createGameClient(backend);
    }

    const destroy = () => {
        if (disposed) return;
        disposed = true;
        mountRevision += 1;
        activeClient?.destroy?.();
        if (window.client === activeClient) window.client = null;
        lobby?.destroy();
        activeClient = null;
        window.__hudesClient = null;
        window.removeEventListener('pagehide', destroy);
    };
    window.addEventListener('pagehide', destroy);
    window.__hudesApp = {
        get gameClient() { return activeClient; },
        get lobbyClient() { return lobby?.client || null; },
        returnToLobby: showLobby,
        destroy,
    };
})().catch((error) => {
    console.error('Human Descent failed to start', error);
    const message = document.createElement('p');
    message.className = 'app-error';
    message.setAttribute('role', 'alert');
    message.textContent = `Unable to start Human Descent: ${error.message}`;
    document.body.append(message);
});
