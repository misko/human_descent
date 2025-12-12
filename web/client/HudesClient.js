import ClientState from './ClientState.js';
import { loadProto } from './ProtoLoader.js';
import { sleep } from '../utils/sleep.js';
import { log } from '../utils/logger.js';
import View from './View.js';
import { SHARE_TEXT } from './helpTour.js';

export default class HudesClient {
    constructor(addr, port, options = {}) {
        // Allow ws/wss based on environment and accept absolute URL or host/port
        try {
            // Allow overriding via URL: ?host=...&port=... (useful in tests)
            if (typeof window !== 'undefined' && window.location && window.location.search) {
                const params = new URLSearchParams(window.location.search);
                const qpHost = params.get('host');
                const qpPort = params.get('port');
                const qpMode = params.get('mode');
                const qpHideHud = params.get('hideHUD');
                if (qpHost) addr = qpHost;
                if (qpPort) port = Number(qpPort);
                if (qpMode) options.renderMode = qpMode.toLowerCase();
                if (qpHideHud) options.hideHud = /^(1|true|yes|on)$/i.test(qpHideHud);
            }
        } catch { }

        // Remember backend for API calls
        this.backendHost = addr || (typeof window !== 'undefined' ? window.location?.hostname : 'localhost');
        this.backendPort = port;
        this.httpBase = options.httpBase || null;
        const isHttps = (typeof window !== 'undefined' && window.location?.protocol === 'https:');
        this.wsScheme = isHttps ? 'wss' : 'ws';
        const wsPath = options.wsPath ?? '/ws';
        const normalizedWsPath = wsPath?.startsWith('/') ? wsPath : `/${wsPath ?? ''}`;
        this.socketUrl = options.wsUrl
            ?? (addr?.startsWith?.('ws')
                ? addr
                : `${this.wsScheme}://${this.backendHost}${this.backendPort ? `:${this.backendPort}` : ''}${normalizedWsPath}`);
        this.socket = null;
        this.requestIdx = 0;
        this.running = true;
        this._connected = false;
        this._shouldReconnect = true;
        this._reconnectAttempts = 0;
        this._sessionStorageAvailable = typeof window !== 'undefined' && !!window.sessionStorage;
        this._sessionTokenKey = 'hudes.tab.sessionToken';
        this._resumeTokenKey = 'hudes.resume.token';
        this._clientSessionToken = this._loadSessionToken();
        this._pendingSessionToken = null;
        this._resumeToken = this._loadPersistedResumeToken();
        if (this._didReload()) {
            this._clearResumeToken();
        }
        this._resumeInFlight = false;
        this._readyForInput = false;
        this._serverAckRequestIdx = 0;
        this._protoReadyPromise = this.loadProto();
        this.apiBase = this.httpBase || this._apiBase(this.backendHost, this.backendPort);

        const renderMode = (options.renderMode || '3d').toLowerCase();
        this.isMobile = Boolean(options.isMobile);
        this.debug = Boolean(options.debug);
        this.renderMode = renderMode === '1d' ? '1d' : '3d';
        this.hideHud = Boolean(options.hideHud);
        this.meshEnabled = this.renderMode !== '1d';
        this.lossLines = this.renderMode === '1d' ? (options.lossLines ?? 6) : 0;
        this.alt1d = this.renderMode === '1d' && Boolean(options.alt1d);
        this.altKeys = Boolean(options.altKeys);

        this.state = new ClientState(-0.05, 12);
        // Optional: allow skipping help screens via URL query, e.g., ?help=off
        let disableInitialTour = false;
        try {
            if (typeof window !== 'undefined' && window.location && window.location.search) {
                const params = new URLSearchParams(window.location.search);
                const help = params.get('help');
                if (help && /^(off|0|false|no)$/i.test(help)) {
                    this.state.helpScreenIdx = -1;
                    disableInitialTour = true;
                }
            }
        } catch { }
        disableInitialTour = disableInitialTour || this.state.helpScreenIdx === -1;
        this.grids = this.meshEnabled ? (options.meshGrids ?? 3) : 0;
        this.grid_size = options.gridSize ?? 31;
        this.rowSpacing = options.rowSpacing;
        this.depthStep = options.depthStep;
        this.cameraDistance = options.cameraDistance;
        this.dimsOffset = 0;
        this.view = new View(this.grid_size, this.grids, this.state, {
            mode: this.renderMode,
            lossLines: this.lossLines,
            debug: this.debug,
            rowSpacing: this.rowSpacing,
            depthStep: this.depthStep,
            cameraDistance: this.cameraDistance,
            alt1d: this.alt1d,
            altKeys: this.altKeys,
            mobile: this.isMobile,
            hideHud: this.hideHud,
            disableInitialTour,
        });
        if (this.hideHud && typeof document !== 'undefined') {
            document.body.classList.add('hudes-hide-hud');
        }
        this.view.initializeCharts(); // Initialize charts
        this.Control = null;
        this.ControlType = null;
        this.lossLineLabels = [];
        this.keyHolds = {};
        this.cooldownTimeMs = 10;
        this.keySecondsPressed = 200; // ms
        this.stepSizeMultiplier = 1;
        this._textCaptureActive = false;

        // Speed run state (frontend)
        this.speedRunActive = false;
        this.speedRunSecondsRemaining = 0;
        this._srsInterval = null; // local countdown ticker (started on first server reply)
        this._srsEndAt = null; // timestamp (ms) when local countdown should end
        this.highScoreSubmitted = false;
        this._speedRunRequestPending = false;


        // Initialize loss and step tracking arrays
        this.trainLosses = [];
        this.valLosses = [];
        this.trainSteps = [];
        this.valSteps = [];

        this._handleHudAction = this._handleHudAction.bind(this);
        this._speedRunCountdownOverlay = null;
        this._speedRunCountdownText = null;
        this.initKeys();
        this._bindHudActions();
        this.levels = [];
        this.bestSessionLoss = Infinity;
        this._loadLevels();
        this._connectSocket();
    }

    async _loadLevels() {
        try {
            const res = await fetch('./levels.json');
            if (res.ok) {
                this.levels = await res.json();
                // Sort descending by loss (e.g. 2.3, 2.1, ... 1.0)
                this.levels.sort((a, b) => b.loss - a.loss);
                // Assign level numbers: Level 1 is best (lowest loss), Level N is worst
                // Since we are sorted worst (high loss) to best (low loss),
                // index 0 is Level N, index length-1 is Level 1.
                this.levels.forEach((level, index) => {
                    level.levelNumber = this.levels.length - index;
                });
                console.log('[HudesClient] Levels loaded:', this.levels.length);
            }
        } catch (e) {
            log('[HudesClient] Failed to load levels', e);
        }
    }

    _bindHudActions() {
        if (typeof document === 'undefined') return;
        document.addEventListener('click', this._handleHudAction);
    }

    _handleHudAction(event) {
        const target = event.target?.closest?.('[data-hud-action]');
        if (!target) return;
        const action = target.dataset.hudAction;
        const refreshHud = () => {
            try {
                this.view?.annotateBottomScreen?.(this.state.toString());
            } catch { }
        };
        switch (action) {
            case 'next-dims':
                event.preventDefault();
                this.getNextDims();
                this._notifyTutorialStep('new_dims');
                break;
            case 'next-batch':
                event.preventDefault();
                this.getNextBatch();
                break;
            case 'step-plus':
                event.preventDefault();
                if (typeof this.state?.increaseStepSize === 'function') {
                    this.state.increaseStepSize(this.stepSizeMultiplier || 1);
                    try { this.sendConfig(); } catch { }
                    refreshHud();
                }
                this._notifyTutorialStep('step');
                break;
            case 'step-minus':
                event.preventDefault();
                if (typeof this.state?.decreaseStepSize === 'function') {
                    this.state.decreaseStepSize(this.stepSizeMultiplier || 1);
                    try { this.sendConfig(); } catch { }
                    refreshHud();
                }
                this._notifyTutorialStep('step');
                break;
            case 'toggle-fp':
                event.preventDefault();
                if (typeof this.state?.toggleDtype === 'function') {
                    this.state.toggleDtype();
                    try { this.sendConfig(); } catch { }
                    refreshHud();
                }
                this._notifyTutorialStep('precision');
                break;
            case 'show-top':
                event.preventDefault();
                try { this.requestLeaderboard?.(); } catch { }
                break;
            case 'toggle-help':
                event.preventDefault();
                this.view?.openHelpOverlay?.('controls');
                break;
            case 'show-tutorial':
                event.preventDefault();
                this.view?.openTutorialOverlay?.();
                break;
            default:
        }
    }

    async loadProto() {
        const { Control, ControlType, root } = await loadProto('hudes.proto');
        this.Control = Control;
        this.ControlType = ControlType;
        try {
            const resumeEnum = root.lookupEnum('hudes.Control.Resume.Status');
            this.resumeStatus = resumeEnum?.values ?? {
                RESUME_OK: 0,
                RESUME_NOT_FOUND: 1,
                RESUME_EXPIRED: 2,
                RESUME_REJECTED: 3,
            };
        } catch {
            this.resumeStatus = {
                RESUME_OK: 0,
                RESUME_NOT_FOUND: 1,
                RESUME_EXPIRED: 2,
                RESUME_REJECTED: 3,
            };
        }
        log("[HudesClient] Proto file and enums loaded successfully.");
    }
    _handleMessage(event) {
        try {
            const buffer = new Uint8Array(event.data);
            const message = this.Control.decode(buffer); // Decode the protobuf message
            // Instrument message counters for tests/debugging
            try {
                if (typeof window !== 'undefined') {
                    window.__hudesMsgCounts = window.__hudesMsgCounts || {};
                    const t = Object.keys(this.ControlType.values).find(
                        k => this.ControlType.values[k] === message.type
                    ) || `type_${message.type}`;
                    window.__hudesMsgCounts[t] = (window.__hudesMsgCounts[t] || 0) + 1;
                }
            } catch { }
            if (typeof message.resumeToken === 'string' && message.resumeToken.length > 0) {
                this._persistResumeToken(message.resumeToken);
            }
            if (typeof message.requestIdx === 'number') {
                this._serverAckRequestIdx = Math.max(
                    this._serverAckRequestIdx,
                    message.requestIdx,
                );
            }

            //log("Decoded Message:", message);

            switch (message.type) {
                case this.ControlType.values.CONTROL_TRAIN_LOSS_AND_PREDS:
                    if (!message.trainLossAndPreds) {
                        throw new Error("trainLossAndPreds field missing");
                    }

                    log(`[HudesClient] recv TRAIN_LOSS_AND_PREDS reqIdx=${message.requestIdx}`);

                    // Update local countdown but do not end until server signals finish
                    this._updateSpeedRunFromMessage(message);

                    // Parse train predictions and reshape
                    const trainPredsFlattened = message.trainLossAndPreds.preds;
                    const trainPredsShape = message.trainLossAndPreds.predsShape;
                    const trainPreds = this._reshapeArray(trainPredsFlattened, trainPredsShape);
                    //log(`Train Predictions Length: ${trainPredsFlattened.length}`);

                    // Parse confusion matrix and reshape
                    const confusionMatrixFlattened = message.trainLossAndPreds.confusionMatrix;
                    const confusionMatrixShape = message.trainLossAndPreds.confusionMatrixShape;
                    const confusionMatrix = this._reshapeArray(confusionMatrixFlattened, confusionMatrixShape);
                    //log("Confusion Matrix received");

                    // Update client state with SGD steps and losses
                    this.state.sgdSteps = message.totalSgdSteps;

                    if (
                        this.trainSteps.length === 0 ||
                        this.trainSteps[this.trainSteps.length - 1] < message.requestIdx
                    ) {
                        this.trainLosses.push(message.trainLossAndPreds.trainLoss);
                        this.trainSteps.push(message.requestIdx);
                        //log(`Train Loss: ${message.trainLossAndPreds.trainLoss}`);
                    }

                    // Update the view with training and validation loss history
                    this.view.updateLossChart(
                        this.trainLosses,
                        this.valLosses,
                        this.trainSteps
                    );

                    this.view.updateLastStepsChart(this.trainSteps.slice(-9), this.trainLosses.slice(-9));

                    // Update the view with predictions and confusion matrix
                    this.view.updateConfusionMatrix(confusionMatrix);
                    this.view.updateExamplePreds(trainPreds);

                    //log("hudes_client: receive message: loss and preds: done");

                    this._updateEvalStepsFromMessage(message);
                    break;
                case this.ControlType.values.CONTROL_RESUME:
                    this._handleResumeControl(message);
                    break;

                case this.ControlType.values.CONTROL_VAL_LOSS:
                    if (!message.valLoss) {
                        throw new Error("valLoss field missing");
                    }
                    // VAL may carry speed_run_finished which is authoritative for ending
                    this._updateSpeedRunFromMessage(message);

                    log(`[HudesClient] recv VAL_LOSS reqIdx=${message.requestIdx} val=${message.valLoss.valLoss}`);

                    // Update validation loss
                    while (this.valLosses.length < this.trainSteps.length) {
                        this.valLosses.push(message.valLoss.valLoss);
                        this.valSteps.push(message.requestIdx);
                    }
                    // if (this.valLosses.length>1) {
                    //     var diffLoss = this.valLosses[this.valLosses.length-1]-this.valLosses[this.valLosses.length-2];
                    //     var diffSteps = this.valSteps[this.valSteps.length-1]-this.valSteps[this.valSteps.length-2];
                    //     for (let i = 0; i < diffSteps; i++) {
                    //         console.log(i); // Will print numbers from 0 to 9
                    //     }
                    //     while (this.valLosses.length)<message.requestIdx:

                    // }


                    this.state.updateBestScoreOrNot(this.valLosses[this.valLosses.length - 1]);

                    // Check for level up
                    const currentValLoss = this.valLosses[this.valLosses.length - 1];
                    if (this.levels && this.levels.length > 0) {
                        // Find the best level we have achieved (loss <= level.loss)
                        // Since levels are sorted descending (2.3, 2.1, ...), we want the LAST one that matches (lowest loss)
                        // Or we can just find the one with the lowest loss that is >= currentValLoss?
                        // No, we want the level corresponding to the milestone we passed.
                        // If we are at 1.6, we passed 1.7. We want to show 1.7.
                        // If we are at 1.4, we passed 1.7 and 1.5. We want to show 1.5 (the best one).

                        // So we want the level with the smallest loss that is still >= currentValLoss.
                        // Wait, if level is 1.5, and we are at 1.4. 1.5 >= 1.4.
                        // If level is 1.0, and we are at 1.4. 1.0 < 1.4.
                        // So we want the level with smallest loss such that level.loss >= currentValLoss.
                        // Actually, usually "Level 1.5" means you beat 1.5. So loss <= 1.5.
                        // So if we are at 1.4, we achieved the 1.5 level.
                        // If we are at 0.9, we achieved the 1.0 level.

                        // We want to find the best level we've beaten (currentValLoss <= level.loss)
                        // that is BETTER than the last one we notified.
                        let bestNewLevel = null;
                        for (const level of this.levels) {
                            if (level.hidden) continue; // Skip hidden levels for in-game notifications
                            if (currentValLoss <= level.loss) {
                                // Since levels are sorted worst to best (descending loss),
                                // later levels in the array are "better" (lower loss).
                                // We want the *best* one we qualify for.
                                if (!bestNewLevel || level.loss < bestNewLevel.loss) {
                                    bestNewLevel = level;
                                }
                            }
                        }

                        if (bestNewLevel) {
                            // Only trigger if this is a new best for this session
                            // OR if we want to show it because we haven't shown THIS level yet?
                            // Requirement: "drops below these leves for the first time in a session"
                            // "If the modal is not dismissed and a new level is achieved the new level should take precedence"

                            // So we track bestSessionLoss.
                            // If currentValLoss < bestSessionLoss, we might have hit a new level.
                            // But wait, if I was at 1.4 (Level 1.5 shown), and I go to 1.3 (Still Level 1.5).
                            // I shouldn't show it again.
                            // I should only show if the *achieved level* is better (lower loss) than the previously achieved level?
                            // Or simply: if we crossed a threshold.

                            // Let's track `this.lastNotifiedLevelLoss`.
                            // If bestAchieved.loss < (this.lastNotifiedLevelLoss ?? Infinity), then show it.

                            if (bestNewLevel.loss < (this.lastNotifiedLevelLoss ?? Infinity)) {
                                console.log('[HudesClient] Level Up! New:', bestNewLevel.loss, 'Old:', this.lastNotifiedLevelLoss);
                                this.lastNotifiedLevelLoss = bestNewLevel.loss;
                                this.view.showLevelUp(bestNewLevel, this.speedRunActive);
                            } else {
                                console.log('[HudesClient] Level not better. New:', bestNewLevel.loss, 'Old:', this.lastNotifiedLevelLoss);
                            }
                        } else {
                            console.log('[HudesClient] No bestNewLevel found for loss:', currentValLoss);
                        }
                    }

            this._updateEvalStepsFromMessage(message);

            // Update the view with training and validation loss history
            this.view.updateLossChart(
                this.trainLosses,
                this.valLosses,
                this.trainSteps
            );

            //og("hudes_client: receive message: val loss: done");
            break;
                case this.ControlType.values.CONTROL_BATCH_EXAMPLES:
            if (message.batchExamples) {
                // log(
                //    `Batch received. Index: ${message.batchExamples.batchIdx}, N: ${message.batchExamples.n}, Train Data Length: ${message.batchExamples.trainData.length}`
                //);

                // Extract train data and labels from the message
                const trainDataFlattened = message.batchExamples.trainData;
                const trainDataShape = message.batchExamples.trainDataShape;

                const trainLabelsFlattened = message.batchExamples.trainLabels;
                const trainLabelsShape = message.batchExamples.trainLabelsShape;

                // Function to reshape a flat array into the given shape
                function reshape(flatArray, shape) {
                    let result = flatArray;
                    for (let i = shape.length - 1; i > 0; i--) {
                        result = chunkArray(result, shape[i]);
                    }
                    return result;
                }

                // Helper function to chunk an array
                function chunkArray(array, size) {
                    const chunkedArray = [];
                    for (let i = 0; i < array.length; i += size) {
                        chunkedArray.push(array.slice(i, i + size));
                    }
                    return chunkedArray;
                }

                // Reshape train data and labels
                const trainData = reshape(trainDataFlattened, trainDataShape);
                const trainLabels = reshape(trainLabelsFlattened, trainLabelsShape);

                // Update the view with the new train data and labels
                this.view.updateExamples(trainData, trainLabels);
                log(`[HudesClient] recv BATCH_EXAMPLES idx=${message.batchExamples.batchIdx}`);
            } else {
                //log("Batch example missing in message.");
            }
            break;


                case this.ControlType.values.CONTROL_MESHGRID_RESULTS:
            if (!message.meshGridResults || !message.meshGridShape) {
                throw new Error("meshGridResults or meshGridShape field missing");
            }

            //log("hudes_client: received message: mesh grid results");

            // Parse the mesh grid results and reshape to the original dimensions
            const meshGridResultsFlattened = message.meshGridResults;
            const meshGridShape = message.meshGridShape;

            // Reshape the flattened results into a 3D array using meshGridShape
            const meshGridResults = this._reshapeArray(
                meshGridResultsFlattened,
                meshGridShape
            );

            //log(`Mesh grid results received:`, meshGridResults);

            // Update the view with the reshaped mesh grid results
            this.view.updateMeshGrids(meshGridResults);

            // Update local countdown but do not end until server signals finish
            this._updateSpeedRunFromMessage(message);
            this._updateEvalStepsFromMessage(message);
            break;

                case this.ControlType.values.CONTROL_LOSS_LINE_RESULTS: {
                if (!message.lossLineResults || !message.lossLineShape) {
                    throw new Error("lossLineResults or lossLineShape field missing");
                }
                const lossLines = this._reshapeArray(
                    message.lossLineResults,
                    message.lossLineShape
                );
                const labels = this._buildLossLineLabels(lossLines.length);
                const stepSpacing = (this.state?.stepSize ?? 1) / 2;
                this.view.updateLossLines?.(lossLines, { stepSpacing, labels });
                this.view.annotateBottomScreen?.(this.state.toString());
                this._updateSpeedRunFromMessage(message);
                this._updateEvalStepsFromMessage(message);
                break;
            }

                // No CONTROL_FULL_LOSS handling (removed)


                case this.ControlType.values.CONTROL_SGD_STEP:
            log(`[HudesClient] recv SGD_STEP ack steps=${message.sgdSteps}`);
            break;
                case this.ControlType.values.CONTROL_LEADERBOARD_RESPONSE:
            try {
                const names = message.leaderboardNames || [];
                const scores = message.leaderboardScores || [];
                const rows = names.map((n, i) => ({ name: n, score: scores[i] }))
                    .filter(r => r && typeof r.name === 'string');
                this._showTop100Modal(rows);
            } catch (e) {
                log('[HudesClient] Failed to render leaderboard modal', e);
            }
            break;
                default:
        log(`[HudesClient] Unknown message type: ${message.type}`);
            }
        } catch (error) {
    log(`[HudesClient] Failed to handle message: ${error}`);
    log(error);
}
    }
_updateSpeedRunFromMessage(message) {
    try {
        const hasFinished = Object.prototype.hasOwnProperty.call(
            message,
            'speedRunFinished',
        );
        const serverActiveFlag = Object.prototype.hasOwnProperty.call(
            message,
            'speedRunActive',
        )
            ? Boolean(message.speedRunActive)
            : null;
        log(
            `[HudesClient] speed-run update: finishFlag=${hasFinished ? message.speedRunFinished : 'n/a'} ` +
            `srs=${Object.prototype.hasOwnProperty.call(message, 'speedRunSecondsRemaining') ? message.speedRunSecondsRemaining : 'n/a'} ` +
            `reqIdx=${message.requestIdx ?? 'n/a'}`,
        );
        if (serverActiveFlag !== null) {
            if (serverActiveFlag && !this.speedRunActive) {
                this.speedRunActive = true;
                this.state.speedRunActive = true;
                document.body.classList.add('speedrun-active');
                this.lastNotifiedLevelLoss = undefined;
            } else if (!serverActiveFlag && this.speedRunActive && !(hasFinished && message.speedRunFinished)) {
                this.speedRunActive = false;
                this.state.speedRunActive = false;
                this.speedRunSecondsRemaining = 0;
                this.state.speedRunSecondsRemaining = 0;
                document.body.classList.remove('speedrun-active');
            }
        }
        if (hasFinished && message.speedRunFinished) {
            // Authoritative end-of-run signal from server
            if (this._srsInterval) { try { clearInterval(this._srsInterval); } catch { } this._srsInterval = null; }
            this._srsEndAt = null;
            this.speedRunSecondsRemaining = 0;
            this.state.speedRunSecondsRemaining = 0;
            this.speedRunActive = false;
            this.state.speedRunActive = false;
            this._speedRunRequestPending = false;
            // Capture achieved final val loss if present
            if (message.valLoss && typeof message.valLoss.valLoss === 'number') {
                this._lastFinalValLoss = message.valLoss.valLoss;
                this._lastRank = message.valLoss.rank;
                this._lastTotalScores = message.valLoss.totalScores;
                log(`[HudesClient] final val loss recorded: ${this._lastFinalValLoss}`);
            }
            if (!this.highScoreSubmitted) {
                log('[HudesClient] prompting for high score submission');
                this._promptAndSubmitHighScore(this._lastFinalValLoss, this._lastRank, this._lastTotalScores);
            }
            document.body.classList.remove('speedrun-active');
            return;
        }

        const hasSrs = Object.prototype.hasOwnProperty.call(
            message,
            'speedRunSecondsRemaining',
        );
        if (!hasSrs) return;
        const srs = message.speedRunSecondsRemaining ?? 0;

        // Start local timer on first srs > 0 after Speed Run starts
        if (srs > 0) {
            this._speedRunRequestPending = false;
            this.speedRunActive = true;
            this.state.speedRunActive = true;
            if (!this._srsInterval) {
                log(`[HudesClient] starting local speed-run countdown at ${srs}s`);
                this._srsEndAt = Date.now() + srs * 1000;
                this.speedRunSecondsRemaining = srs;
                this.state.speedRunSecondsRemaining = srs;
                let lastShown = srs;
                this._srsInterval = setInterval(() => {
                    const now = Date.now();
                    const remainMs = Math.max(0, (this._srsEndAt ?? now) - now);
                    const remain = Math.max(0, Math.floor(remainMs / 1000));
                    const remainFrac = Math.max(0, Math.round(remainMs / 100) / 10); // 0.1s resolution
                    if (remain !== lastShown) {
                        lastShown = remain;
                        this.speedRunSecondsRemaining = remainFrac;
                        this.state.speedRunSecondsRemaining = remainFrac;
                        try { this.view?.annotateBottomScreen?.(this.state.toString()); } catch { }
                        log(`[HudesClient] speed-run countdown tick: ${remainFrac.toFixed?.(1) ?? remainFrac}s remaining`);
                    }
                    if (remain === 0) {
                        // Stop ticking locally, but keep run active until
                        // server sends VAL with speed_run_finished=true
                        try { clearInterval(this._srsInterval); } catch { }
                        this._srsInterval = null;
                        this._srsEndAt = null;
                        this.speedRunSecondsRemaining = 0;
                        this.state.speedRunSecondsRemaining = 0;
                        try { this.view?.annotateBottomScreen?.(this.state.toString()); } catch { }
                        log('[HudesClient] local countdown reached zero waiting for server finish');
                    }
                }, 300);
            }
        } else {
            // srs==0: if we haven't started local timer yet, reflect zero
            if (!this._srsInterval) {
                this.speedRunSecondsRemaining = 0;
                this.state.speedRunSecondsRemaining = 0;
                // Do not flip active here; end only on server finish
                try { this.view?.annotateBottomScreen?.(this.state.toString()); } catch { }
            }
        }
    } catch { }
}
_promptAndSubmitHighScore(finalValLoss, rank, totalScores) {
    log('[HudesClient] _promptAndSubmitHighScore invoked');
    if (this.highScoreSubmitted) return;
    try {
        this._showSpeedRunResults({ finalValLoss, rank, totalScores });
    } catch { }
}

requestLeaderboard() {
    try {
        if (!this.Control || !this.ControlType) return;
        const payload = {
            type: this.ControlType.values.CONTROL_LEADERBOARD_REQUEST,
        };
        this.sendQ(payload);
    } catch { }
}

_createOverlay() {
    let overlay = document.getElementById('modalOverlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'modalOverlay';
        document.body.appendChild(overlay);
    }
    return overlay;
}

_showSpeedRunResults({ finalValLoss, rank, totalScores }) {
    this._setTextCaptureActive(true);
    const overlay = this._createOverlay();
    overlay.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'glass-card results-card';
    const title = document.createElement('h2');
    title.textContent = 'Congratulations, Speed Run Complete!';

    // Calculate Level
    let achievedLevel = null;
    if (this.levels && this.levels.length > 0 && typeof finalValLoss === 'number') {
        // Levels are sorted descending by loss (worst to best)
        // We want the best level achieved (lowest loss threshold met)
        for (const level of this.levels) {
            if (finalValLoss <= level.loss) {
                if (!achievedLevel || level.loss < achievedLevel.loss) {
                    achievedLevel = level;
                }
            }
        }
    }

    const metrics = document.createElement('div');
    metrics.className = 'results-metrics-row';

    const lossVal = (typeof finalValLoss === 'number' && Number.isFinite(finalValLoss))
        ? finalValLoss.toFixed(4)
        : '—';
    const evalSteps = this.state?.evalSteps ?? 0;
    const planes = Math.max(1, Math.floor((this.state?.dimsUsed || 0) / (this.state?.n || 1)));
    const rankText = (rank && totalScores) ? `${rank}/${totalScores}` : '-/-';

    metrics.innerHTML = `
            <div class="metric-item"><span>Loss</span><strong>${lossVal}</strong></div>
            <div class="metric-sep"></div>
            <div class="metric-item"><span>Steps</span><strong>${evalSteps}</strong></div>
            <div class="metric-sep"></div>
            <div class="metric-item"><span>Subspaces</span><strong>${planes}</strong></div>
            <div class="metric-sep"></div>
            <div class="metric-item"><span>Rank</span><strong>${rankText}</strong></div>
        `;

    const shareText = SHARE_TEXT(
        finalValLoss,
        achievedLevel?.levelNumber || '-',
        achievedLevel?.title || 'Unranked'
    );

    const share = document.createElement('p');
    share.className = 'results-share';
    share.textContent = shareText;

    // Social Actions
    const shareActions = document.createElement('div');
    shareActions.className = 'share-actions';

    // X (Twitter) Button
    const xBtn = document.createElement('a');
    xBtn.className = 'share-btn x-btn';
    xBtn.target = '_blank';
    xBtn.rel = 'noopener noreferrer';
    xBtn.textContent = 'Share on 𝕏';
    xBtn.href = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(window.location.origin)}`;

    // Copy Button
    const copyBtn = document.createElement('button');
    copyBtn.className = 'share-btn copy-btn';
    copyBtn.textContent = 'Copy Text';
    copyBtn.onclick = async () => {
        try {
            await navigator.clipboard.writeText(shareText);
            const original = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            setTimeout(() => copyBtn.textContent = original, 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    };

    shareActions.append(xBtn, copyBtn);

    const form = document.createElement('form');
    form.className = 'name-form';
    form.innerHTML = `
            <label for="nameInput">Enter a 4-character name to submit</label>
            <input id="nameInput" maxlength="4" placeholder="USER" autocomplete="off" />
            <div class="actions">
                <button type="submit">Submit score</button>
            </div>
        `;
    const enableActions = () => { };
    form.addEventListener('submit', (event) => {
        event.preventDefault();
    });
    form.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            event.stopPropagation();
        }
    });
    const submitBtn = form.querySelector('button[type="submit"]');
    if (!submitBtn) {
        log('[HudesClient] Missing submit button in speed-run modal');
        return;
    }
    submitBtn.addEventListener('click', (event) => {
        event.preventDefault();
        const input = form.querySelector('#nameInput');
        const clean = (input.value || 'USER').toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 4);
        if (clean.length !== 4) {
            input.classList.add('invalid');
            return;
        }
        input.classList.remove('invalid');
        this.submitHighScore(clean);
        enableActions();
        this.highScoreSubmitted = true;
        try { this.requestLeaderboard(); } catch { }
        overlay.classList.remove('open');
        this._setTextCaptureActive(false);
    });
    card.append(title, metrics, share, shareActions, form);
    overlay.appendChild(card);
    overlay.classList.add('open');
    const input = form.querySelector('#nameInput');
    input?.focus();
    log('[HudesClient] speed-run results modal shown');
}

_showLeaderboardModal(top10, me) {
    this._setTextCaptureActive(false);
    const overlay = this._createOverlay();
    overlay.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'glass-card';
    const title = document.createElement('h2');
    title.textContent = 'Leaderboard';
    const you = document.createElement('p');
    you.className = 'muted';
    const score = (typeof me.score === 'number' && isFinite(me.score)) ? me.score.toFixed(4) : '—';
    const rankText = (me.rank && me.total) ? `#${me.rank} of ${me.total}` : '#—';
    you.textContent = `${me.name ?? 'YOU'} • Score: ${score} • Rank: ${rankText}`;

    const list = document.createElement('ol');
    list.className = 'top10-list';
    (top10 || []).forEach((row, idx) => {
        const li = document.createElement('li');
        const rank = (idx + 1);
        const s = (typeof row.score === 'number') ? row.score.toFixed(4) : row.score;
        li.textContent = `${rank} ${row.name} — ${s}`;
        list.appendChild(li);
    });

    const actions = document.createElement('div');
    actions.className = 'actions';
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.addEventListener('click', () => {
        overlay.classList.remove('open');
        this._setTextCaptureActive(false);
    });
    const viewBtn = document.createElement('button');
    viewBtn.textContent = 'View Top 10';
    viewBtn.className = 'link-btn';
    viewBtn.addEventListener('click', () => {
        try { this.requestLeaderboard?.(); } catch { }
    });
    actions.appendChild(viewBtn);
    actions.appendChild(closeBtn);

    card.appendChild(title);
    card.appendChild(you);
    card.appendChild(list);
    card.appendChild(actions);
    overlay.appendChild(card);
    overlay.classList.add('open');
}

_detectBackend() {
    const params = new URLSearchParams(window.location.search);
    const host = params.get('host') || this.backendHost || window.location.hostname || 'localhost';
    const port = params.get('port') ? Number(params.get('port')) : this.backendPort;
    return { host, port, httpBase: this.apiBase };
}

_apiBase(host, port) {
    if (this.httpBase) return this.httpBase;
    const proto = typeof window !== 'undefined' && window.location?.protocol === 'https:' ? 'https' : 'http';
    if (Number.isFinite(Number(port))) {
        const httpPort = Number(port) + 1;
        return `${proto}://${host}:${httpPort}/api`;
    }
    if (typeof window !== 'undefined' && window.location?.origin) {
        return `${window.location.origin}/api`;
    }
    return `${proto}://${host || 'localhost'}/api`;
}
_buildLossLineLabels(count) {
    this.lossLineLabels = Array.from({ length: count }, (_, idx) => {
        const dimNumber = this.dimsOffset + idx + 1;
        return `dim ${dimNumber}`;
    });
    return this.lossLineLabels;
}
_reshapeArray(flatArray, shape) {
    // Helper function to recursively reshape the flattened array
    function recursiveReshape(array, dims) {
        if (dims.length === 0) {
            return array[0];
        }
        const size = dims[0];
        const rest = dims.slice(1);
        const reshaped = [];
        for (let i = 0; i < size; i++) {
            reshaped.push(recursiveReshape(array.splice(0, rest.reduce((a, b) => a * b, 1)), rest));
        }
        return reshaped;
    }

    return recursiveReshape([...flatArray], shape);
}
// // Helper function to deserialize bytes
// _deserializeBytes(bytes) {
//     try {
//         // Assuming protobuf bytes are serialized with JSON
//         return JSON.parse(new TextDecoder().decode(bytes));
//     } catch (error) {
//         console.error("Failed to deserialize bytes:", error);
//         return [];
//     }
// }


_connectSocket() {
    if (!this._shouldReconnect) {
        return;
    }
    try {
        log(`[HudesClient] Connecting WebSocket to ${this.socketUrl}`);
        this.socket = new WebSocket(this.socketUrl);
        this.socket.binaryType = 'arraybuffer';
        this.socket.onopen = () => this._handleSocketOpen();
        this.socket.onclose = (event) => this._handleSocketClose(event);
        this.socket.onmessage = (event) => this._handleMessage(event);
        this.socket.onerror = (error) =>
            log(`[HudesClient] WebSocket error: ${error?.message ?? error}`, null, 'error');
    } catch (err) {
        log(`[HudesClient] Failed to open WebSocket: ${err?.message ?? err}`, null, 'error');
        this._scheduleReconnect();
    }
}

_handleSocketOpen() {
    log('[HudesClient] WebSocket connected.');
    this._connected = true;
    try {
        if (typeof window !== 'undefined') {
            const el = document.querySelector('.connection-status');
            if (el) {
                el.dataset.state = 'connected';
                el.textContent = 'Connected';
            }
        }
    } catch { }
    this._reconnectAttempts = 0;
    if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = null;
    }
    this._readyForInput = false;
    (async () => {
        try {
            await this._protoReadyPromise;
            if (this._resumeToken) {
                this._resumeInFlight = true;
                this._sendResumeRequest();
            } else {
                this._resumeInFlight = false;
                await this.sendConfig();
                this._readyForInput = true;
            }
        } catch (err) {
            log(`[HudesClient] Failed during socket open init: ${err?.message ?? err}`, null, 'error');
        }
    })();
}

_handleSocketClose(event) {
    log(`[HudesClient] WebSocket disconnected (code=${event?.code ?? 'n/a'})`);
    this._connected = false;
    try {
        if (typeof window !== 'undefined') {
            const el = document.querySelector('.connection-status');
            if (el) {
                el.dataset.state = 'disconnected';
                el.textContent = 'Disconnected';
            }
        }
    } catch { }
    this._readyForInput = false;
    this._resumeInFlight = false;
    this._pendingSessionToken = null;
    if (!this._shouldReconnect) {
        this.running = false;
        return;
    }
    this._scheduleReconnect();
}

_scheduleReconnect() {
    this._reconnectAttempts += 1;
    const delay = Math.min(1000 * this._reconnectAttempts, 5000);
    if (this._reconnectTimer) {
        clearTimeout(this._reconnectTimer);
    }
    this._reconnectTimer = setTimeout(() => this._connectSocket(), delay);
}

_sendResumeRequest() {
    if (!this.Control || !this.ControlType) {
        return;
    }
    if (!this._resumeToken) {
        this._pendingSessionToken = null;
        this._readyForInput = false;
        this.sendConfig();
        return;
    }
    const nextSessionToken = this._generateSessionToken();
    this._pendingSessionToken = nextSessionToken;
    const payload = {
        type: this.ControlType.values.CONTROL_RESUME,
        resume: {
            token: this._resumeToken,
            lastRequestIdx: this._serverAckRequestIdx,
            clientSessionToken: this._clientSessionToken,
            newClientSessionToken: nextSessionToken,
        },
    };
    this.sendQ(payload);
}

_handleResumeControl(message) {
    const resumePayload = message?.resume;
    const status = resumePayload?.status;
    const token = resumePayload?.token || message?.resumeToken;
    if (token) {
        this._persistResumeToken(token);
    }
    if (typeof resumePayload?.lastRequestIdx === 'number') {
        this._serverAckRequestIdx = Math.max(
            this._serverAckRequestIdx,
            resumePayload.lastRequestIdx,
        );
    }
    if (status === this.resumeStatus?.RESUME_OK || status === 0) {
        const nextSession =
            resumePayload?.clientSessionToken || this._pendingSessionToken;
        if (nextSession) {
            this._setSessionToken(nextSession);
        }
        this._pendingSessionToken = null;
        this._resumeInFlight = false;
        this._readyForInput = true;
        try {
            this.sendConfig();
        } catch (err) {
            log(`[HudesClient] Failed to resend config after resume: ${err?.message ?? err}`, null, 'error');
        }
        return;
    }
    this._pendingSessionToken = null;
    this._resumeInFlight = false;
    this._readyForInput = false;
    this._clearResumeToken();
    this.sendConfig();
}

_persistResumeToken(token) {
    if (!token || typeof token !== 'string') {
        return;
    }
    this._resumeToken = token;
    try {
        if (this._sessionStorageAvailable) {
            window.sessionStorage?.setItem(this._resumeTokenKey, token);
        }
    } catch { }
}

_loadPersistedResumeToken() {
    try {
        if (this._sessionStorageAvailable) {
            return window.sessionStorage?.getItem(this._resumeTokenKey) || null;
        }
    } catch { }
    return null;
}

_didReload() {
    try {
        if (typeof performance !== 'undefined') {
            const entries = performance.getEntriesByType?.('navigation');
            if (entries && entries.length > 0) {
                return entries[0].type === 'reload';
            }
            if (performance.navigation) {
                const TYPE_RELOAD =
                    performance.navigation.TYPE_RELOAD ??
                    (typeof PerformanceNavigation !== 'undefined'
                        ? PerformanceNavigation.TYPE_RELOAD
                        : 1);
                return performance.navigation.type === TYPE_RELOAD;
            }
        }
        } catch { }
        return false;
    }

    _getLevelForScore(score, { includeHidden = false } = {}) {
        if (!this.levels || !Array.isArray(this.levels) || !Number.isFinite(score)) {
            return null;
        }
        let best = null;
        for (const level of this.levels) {
            if (!level || (!includeHidden && level.hidden) || typeof level.loss !== 'number') {
                continue;
            }
            if (score <= level.loss) {
                if (!best || level.loss < best.loss) {
                    best = level;
                }
            }
        }
        return best;
    }

    _clearResumeToken() {
        this._resumeToken = null;
        try {
            if (this._sessionStorageAvailable) {
                window.sessionStorage?.removeItem(this._resumeTokenKey);
        }
    } catch { }
}

_notifyTutorialStep(stepId) {
    if (!stepId) {
        return;
    }
    try {
        this.view?.notifyTutorialEvent?.(stepId);
    } catch { }
}

toggleKeyOverlay(mode) {
    const view = this.view;
    if (!view) {
        return;
    }
    if (view.isManualKeyOverlayVisible?.()) {
        view.hideImage?.();
        return;
    }
    view.openHelpOverlay?.('controls');
}

_ensureCountdownOverlay() {
    if (this._speedRunCountdownOverlay) {
        return this._speedRunCountdownOverlay;
    }
    if (typeof document === 'undefined') {
        return null;
    }
    const overlay = document.createElement('div');
    overlay.id = 'speedrunCountdownOverlay';
    const box = document.createElement('div');
    box.className = 'speedrun-countdown__box';
    const text = document.createElement('span');
    text.className = 'speedrun-countdown__text';
    box.appendChild(text);
    overlay.appendChild(box);
    document.body.appendChild(overlay);
    this._speedRunCountdownOverlay = overlay;
    this._speedRunCountdownText = text;
    return overlay;
}

_showCountdownFrame(text, variant = 'ready') {
    const overlay = this._ensureCountdownOverlay();
    if (!overlay) {
        return;
    }
    if (this._speedRunCountdownText) {
        this._speedRunCountdownText.textContent = text.toUpperCase();
    } else {
        overlay.textContent = text.toUpperCase();
    }
    overlay.classList.remove('ready', 'count');
    overlay.classList.add('visible', variant);
    document.body.classList.add('speedrun-countdown');
}

_hideCountdownOverlay() {
    const overlay = this._speedRunCountdownOverlay;
    if (overlay) {
        overlay.classList.remove('visible', 'ready', 'count');
        if (this._speedRunCountdownText) {
            this._speedRunCountdownText.textContent = '';
        } else {
            overlay.textContent = '';
        }
    }
    document.body.classList.remove('speedrun-countdown');
}

_clearCountdownTimer() {
    if (this._speedRunCountdownTimer) {
        clearTimeout(this._speedRunCountdownTimer);
        this._speedRunCountdownTimer = null;
    }
}

_runSpeedRunCountdown() {
    const frames = [
        { text: 'Get ready', variant: 'ready', duration: 900 },
        { text: '3', variant: 'count', duration: 700 },
        { text: '2', variant: 'count', duration: 700 },
        { text: '1', variant: 'count', duration: 700 },
    ];
    let idx = 0;
    const step = () => {
        if (idx >= frames.length) {
            this._clearCountdownTimer();
            this._hideCountdownOverlay();
            this._sendSpeedRunStart();
            return;
        }
        const frame = frames[idx++];
        this._showCountdownFrame(frame.text, frame.variant);
        this._speedRunCountdownTimer = setTimeout(step, frame.duration);
    };
    step();
}

_cancelSpeedRunCountdown() {
    this._clearCountdownTimer();
    this._hideCountdownOverlay();
    document.body.classList.remove('speedrun-active');
    this._speedRunRequestPending = false;
}

_updateEvalStepsFromMessage(message) {
    const steps = message?.totalEvalSteps;
    if (typeof steps === 'number' && Number.isFinite(steps)) {
        if (typeof this.state.setEvalSteps === 'function') {
            this.state.setEvalSteps(steps);
        } else {
            this.state.evalSteps = steps;
        }
    }
}

_loadSessionToken() {
    if (!this._sessionStorageAvailable) {
        return this._generateSessionToken();
    }
    try {
        const existing = window.sessionStorage?.getItem(this._sessionTokenKey);
        if (existing) {
            return existing;
        }
    } catch { }
    const token = this._generateSessionToken();
    this._persistSessionToken(token);
    return token;
}

_persistSessionToken(token) {
    if (!token || typeof token !== 'string' || !this._sessionStorageAvailable) {
        return;
    }
    try {
        window.sessionStorage?.setItem(this._sessionTokenKey, token);
    } catch { }
}

_setSessionToken(token) {
    if (!token || typeof token !== 'string') {
        return;
    }
    this._clientSessionToken = token;
    this._persistSessionToken(token);
}

_generateSessionToken() {
    try {
        if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
            const bytes = new Uint32Array(4);
            crypto.getRandomValues(bytes);
            return Array.from(bytes, (b) => b.toString(16).padStart(8, '0')).join('');
        }
    } catch { }
    return `sess-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
}

_canSendUserCommands() {
    return (
        this.socket &&
        this.socket.readyState === WebSocket.OPEN &&
        !this._resumeInFlight &&
        this._readyForInput
    );
}

initKeys() {
    this._handleKeyDown = this._handleKeyDown.bind(this);
    this._handleKeyUp = this._handleKeyUp.bind(this);

    window.addEventListener("keydown", this._handleKeyDown);
    window.addEventListener("keyup", this._handleKeyUp);
}

_setTextCaptureActive(active) {
    this._textCaptureActive = Boolean(active);
}

_shouldIgnoreKeyEvent(event) {
    if (this._textCaptureActive) {
        return true;
    }
    const target = event?.target;
    if (target && target instanceof HTMLElement) {
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
            return true;
        }
        if (target.isContentEditable) {
            return true;
        }
    }
    return false;
}


// Utility to track key press timing
updateKeyHolds(currentlyPressedKeys) {
    const currentTime = performance.now();
    for (const key in this.keyHolds) {
        if (currentlyPressedKeys[key]) {
            if (this.keyHolds[key].firstPress === -1) {
                this.keyHolds[key].firstPress = currentTime;
            }
        } else {
            this.keyHolds[key].firstPress = -1;
        }
    }
}

// Checks if the cooldown time has elapsed for a key
checkCooldownTime(event, key, currentTime, cooldownTimeSec) {
    const cooldownTimeMs = cooldownTimeSec * 1000; // Convert to ms
    if (event.key !== key) {
        return false;
    }
    if (currentTime - (this.keyHolds[key]?.lastExec || 0) > cooldownTimeMs) {
        this.keyHolds[key].lastExec = currentTime;
        return true;
    }
    return false;
}

// Checks if the key has been held down for a specific duration
checkKeyHoldTime(event, key, currentTime, holdTimeSec) {
    const holdTimeMs = holdTimeSec * 1000; // Convert to ms
    if (event.key !== key) {
        return false;
    }
    if (
        this.keyHolds[key]?.firstPress >= 0 &&
        currentTime - this.keyHolds[key].firstPress > holdTimeMs
    ) {
        return true;
    }
    return false;
}

setN(n) {
    n = Math.max(6, n); // Ensure minimum value of 6
    this.state.n = n;
    this.zeroDimsAndStepsOnCurrentDims();
}
_handleKeyDown(event) {
    if (this._shouldIgnoreKeyEvent(event)) {
        return;
    }
    const currentTime = performance.now();
    if (!this.keyHolds[event.code]) {
        this.keyHolds[event.code] = { firstPress: currentTime, lastExec: 0 };
    } else if (this.keyHolds[event.code].firstPress === -1) {
        this.keyHolds[event.code].firstPress = currentTime;
    }
    this.processKeyPress(event); // Call the processKeyPress method
}

_handleKeyUp(event) {
    if (this._shouldIgnoreKeyEvent(event)) {
        return;
    }
    if (this.keyHolds[event.code]) {
        this.keyHolds[event.code].firstPress = -1;
    }
}

processKeyPress(event) {
    // Default key handling logic, can be overridden by derived classes
    this.processCommonKeys(event);
}

addKeyToWatch(key) {
    // Initialize tracking for the key if not already present
    if (!this.keyHolds[key]) {
        this.keyHolds[key] = { firstPress: -1, lastExec: 0 };
    }
}

processCommonKeys(event) {
    if (event.metaKey || event.ctrlKey) {
        return;
    }
    const currentTime = Date.now();

    // Check cooldown
    const keyInfo = this.keyHolds[event.code];
    if (keyInfo && currentTime - keyInfo.lastExec < this.cooldownTimeMs) {
        return;
    }

    if (event.type === "keydown") {
        switch (event.code) {
            case "KeyZ":
                if (this.speedRunActive || this._speedRunRequestPending) {
                    log('[HudesClient] KeyZ pressed: cancelling speed run');
                    this.cancelSpeedRun();
                } else {
                    log('[HudesClient] KeyZ pressed: starting speed run');
                    this.startSpeedRun();
                }
                break;
            case "KeyY":
                // Request Top 10 leaderboard over WebSocket
                this.requestLeaderboard();
                break;
            case "KeyX":
                if (this.view?.isManualKeyOverlayVisible?.()) {
                    this.view.hideImage?.();
                } else if (this.state.helpScreenIdx !== -1) {
                    if (typeof this.state.closeHelpScreens === 'function') {
                        this.state.closeHelpScreens();
                    } else {
                        this.state.helpScreenIdx = -1;
                    }
                    this.view.hideImage?.();
                } else {
                    this.toggleKeyOverlay();
                }
                break;
            case "Slash":
                if (event.shiftKey) {
                    this.toggleKeyOverlay();
                }
                break;
            // default:
            //       log(`Unhandled key: ${event.code}`);
        }
    }

    if (this.state.helpScreenIdx != -1) {
        return;
    }

    if (event.type === "keydown") {

        switch (event.code) {
            case "KeyQ":
                if (currentTime - keyInfo.firstPress > 1000) {
                    log("Quitting...");
                    this.running = false;
                }
                break;

            case "Backspace":
            case "Delete":
                if (this.speedRunActive || this.state.speedRunActive) {
                    log("SGD disabled during Speed Run.");
                } else {
                    log("[HudesClient] Performing SGD step.");
                    this.getSGDStep();
                }
                break;

            case "Space":
                if (currentTime - keyInfo.lastExec > this.keySecondsPressed) {
                    log("[HudesClient] Getting next dimensions.");
                    this.getNextDims();
                    this._notifyTutorialStep('new_dims');
                }
                break;

            case "BracketLeft":
                log("[HudesClient] Increasing step size.");
                this.state.increaseStepSize(this.stepSizeMultiplier);
                this.sendConfig();
                this._notifyTutorialStep('step');
                break;

            case "BracketRight":
                log("[HudesClient] Decreasing step size.");
                this.state.decreaseStepSize(this.stepSizeMultiplier);
                this.sendConfig();
                this._notifyTutorialStep('step');
                break;

            case "Quote":
                log("[HudesClient] Toggling dtype.");
                this.state.toggleDtype();
                this.sendConfig();
                this._notifyTutorialStep('precision');
                break;

            case "Semicolon":
                log("[HudesClient] Toggling batch size.");
                this.state.toggleBatchSize();
                this.sendConfig();
                this._notifyTutorialStep('batch');
                break;

            case "Enter":
                log("[HudesClient] Getting next batch.");
                this.getNextBatch();
                break;

        }

        // Update last execution time
        if (this.keyHolds[event.code]) {
            this.keyHolds[event.code].lastExec = currentTime;
        }
    }
}
    _showTop100Modal(rows) {
        const overlay = this._createOverlay();
        overlay.innerHTML = '';
        const card = document.createElement('div');
        card.className = 'glass-card';
    const title = document.createElement('h2');
    title.textContent = 'Top 10 Leaderboard';
    const listWrap = document.createElement('div');
        listWrap.className = 'scroll-wrap';
        const list = document.createElement('ol');
        list.className = 'top10-list';
        (rows || []).forEach((r, idx) => {
            const li = document.createElement('li');
            const rank = idx + 1;
            const score = Number.isFinite(r.score) ? r.score : null;
            const lossStr = score == null ? '---' : score.toFixed(3);
            const levelInfo = this._getLevelForScore(score, { includeHidden: true });
            const levelLabel = levelInfo ? `L${levelInfo.levelNumber ?? '—'}` : 'L—';
            const nameRaw = (typeof r.name === 'string' && r.name.trim()) ? r.name.trim() : '—';
            const maxNameWidth = 5;
            const nameFixed = nameRaw.length > maxNameWidth ? nameRaw.slice(0, maxNameWidth) : nameRaw;
            const mono = document.createElement('span');
            mono.className = 'top10-mono';
            const monoParts = [
                String(rank).padStart(2, ' '),
                nameFixed.padEnd(maxNameWidth, ' '),
                lossStr.padStart(6, ' '),
                levelLabel.padEnd(4, ' ')
            ];
            mono.textContent = monoParts.join(' ');
            const titleSpan = document.createElement('span');
            titleSpan.className = 'top10-title';
            const levelTitle = levelInfo?.title;
            titleSpan.textContent = levelTitle ? ` ${levelTitle}` : '';
            li.appendChild(mono);
            if (titleSpan.textContent.trim()) {
                li.appendChild(titleSpan);
            }
            list.appendChild(li);
        });
        listWrap.appendChild(list);
        const actions = document.createElement('div');
        actions.className = 'actions';
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.addEventListener('click', () => overlay.classList.remove('open'));
    actions.appendChild(closeBtn);
    card.appendChild(title);
    card.appendChild(listWrap);
    card.appendChild(actions);
    overlay.appendChild(card);
    overlay.classList.add('open');
}

waitForWebSocket(callback) {
    const checkInterval = 50; // Check every 50ms
    const maxWaitTime = 5000; // Maximum wait time of 5 seconds
    let elapsedTime = 0;

    // Quick check if the WebSocket is already open
    if (!this.socket) {
        console.error("WebSocket is not initialized.");
        return;
    }
    if (this.socket.readyState === WebSocket.OPEN) {
        callback();
        return;
    }

    const interval = setInterval(() => {
        if (!this.socket) {
            clearInterval(interval);
            console.error("WebSocket instance missing during wait.");
            return;
        }
        if (this.socket.readyState === WebSocket.OPEN) {
            clearInterval(interval);
            callback(); // Invoke the callback when the WebSocket is ready
        } else if (elapsedTime >= maxWaitTime) {
            clearInterval(interval);
            console.error("WebSocket did not open within the timeout period.");
        }
        elapsedTime += checkInterval;
    }, checkInterval);
}
sendQ(payload) {
    if (!this.Control || !this.ControlType) {
        console.error("Proto definitions not loaded. Cannot send message.");
        return;
    }

    // Add requestIdx to the payload if not already present
    payload.requestIdx = this.requestIdx;

    // Verify the payload
    const errMsg = this.Control.verify(payload);
    if (errMsg) {
        console.error("Message verification failed:", errMsg);
        return;
    }

    // Create and encode the message
    const message = this.Control.create(payload);
    const buffer = this.Control.encode(message).finish();

    // Send the message over WebSocket
    this.waitForWebSocket(() => {
        try {
            const t = Object.keys(this.ControlType.values).find(
                k => this.ControlType.values[k] === payload.type
            );
            log(`[HudesClient] send ${t ?? payload.type} reqIdx=${payload.requestIdx ?? this.requestIdx}`);
        } catch { }
        this.socket.send(buffer);
        this.requestIdx++;
    });
    //console.log("Message sent:", payload);
}


nextDimsMessage(requestIdx) {
    if (!this.Control || !this.ControlType) {
        console.error("Proto definitions not loaded. Cannot create nextDimsMessage.");
        return null;
    }

    return {
        type: this.ControlType.values.CONTROL_NEXT_DIMS,
        requestIdx,
    };
}

getSGDStep() {
    if (!this.Control || !this.ControlType) {
        console.error("Proto definitions not loaded. Cannot send SGD step message.");
        return;
    }

    if (this.speedRunActive || this.state.speedRunActive) {
        // Client-side guard; server also ignores
        return;
    }
    if (!this._canSendUserCommands()) {
        log("[HudesClient] Ignoring SGD request while socket unavailable", null, 'warn');
        return;
    }

    const payload = {
        type: this.ControlType.values.CONTROL_SGD_STEP,
        sgdSteps: 1,
    };

    this.sendQ(payload);
    console.log("[HudesClient] SGD step message sent.");
}


getNextBatch() {
    if (!this.Control || !this.ControlType) {
        console.error("Proto definitions not loaded. Cannot request next batch.");
        return;
    }
    if (!this._canSendUserCommands()) {
        log("[HudesClient] Ignoring next batch while socket unavailable", null, 'warn');
        return;
    }

    const payload = {
        type: this.ControlType.values.CONTROL_NEXT_BATCH,
    };

    this.sendQ(payload);
    console.log("[HudesClient] Next batch message sent.");
}


    async getNextDims() {
    if (!this.Control || !this.ControlType) {
        console.error("Proto definitions not loaded. Cannot request next dims.");
        return;
    }
    if (!this._canSendUserCommands()) {
        log("[HudesClient] Ignoring next dims while socket unavailable", null, 'warn');
        return;
    }

    // Create payload
    const payload = {
        type: this.ControlType.values.CONTROL_NEXT_DIMS,
    };

    // Use sendQ to handle sending
    this.sendQ(payload);

    // Perform the reset and update operations
    this.zeroDimsAndStepsOnCurrentDims();
    this.dimsAndStepsUpdated();
    this.state.dimsUsed += this.state.n;
    this.dimsOffset += this.state.n;

    //console.log("Next dimensions requested and steps reset.");
}

startSpeedRun() {
    if (!this.Control || !this.ControlType) {
        console.error("Proto definitions not loaded. Cannot start Speed Run.");
        return;
    }
    if (!this._canSendUserCommands()) {
        log("[HudesClient] Cannot start Speed Run while disconnected", null, 'warn');
        return;
    }
    if (this._speedRunCountdownTimer) {
        this._cancelSpeedRunCountdown();
        return;
    }
    if (this.speedRunActive || this._speedRunRequestPending) {
        this.cancelSpeedRun();
        return;
    }
    this._speedRunRequestPending = true;
    this._runSpeedRunCountdown();
}

_sendSpeedRunStart() {
    this.highScoreSubmitted = false;
    if (this._srsInterval) {
        try {
            clearInterval(this._srsInterval);
        } catch { }
        this._srsInterval = null;
    }
    this._srsEndAt = null;
    this.speedRunActive = true;
    this.state.speedRunActive = true;
    this.state.speedRunSecondsRemaining = 0;
    this.state.bestScore = Infinity;
    document.body.classList.remove('speedrun-countdown');
    document.body.classList.add('speedrun-active');
    const payload = {
        type: this.ControlType.values.CONTROL_SPEED_RUN_START,
    };
    this.sendQ(payload);
    log("[HudesClient] Speed Run started.");
}

cancelSpeedRun() {
    if (!this.Control || !this.ControlType) {
        console.error("Proto definitions not loaded. Cannot cancel Speed Run.");
        return;
    }
    if (!this._canSendUserCommands()) {
        log("[HudesClient] Cannot cancel Speed Run while disconnected", null, 'warn');
        return;
    }
    this._cancelSpeedRunCountdown();
    if (!this.speedRunActive && !this._speedRunRequestPending) {
        return;
    }
    this._speedRunRequestPending = false;
    this.speedRunActive = false;
    this.state.speedRunActive = false;
    this.speedRunSecondsRemaining = 0;
    this.state.speedRunSecondsRemaining = 0;
    if (this._srsInterval) {
        try { clearInterval(this._srsInterval); } catch { }
        this._srsInterval = null;
    }
    this._srsEndAt = null;
    document.body.classList.remove('speedrun-active', 'speedrun-countdown');
    const payload = {
        type: this.ControlType.values.CONTROL_SPEED_RUN_CANCEL,
    };
    this.sendQ(payload);
    log('[HudesClient] Speed Run cancelled.');
}

submitHighScore(name) {
    if (!this.Control || !this.ControlType) {
        console.error("Proto definitions not loaded. Cannot submit High Score.");
        return;
    }
    if (this.highScoreSubmitted) return;
    if (!this._canSendUserCommands()) {
        log("[HudesClient] Cannot submit High Score while disconnected", null, 'warn');
        return;
    }
    const clean = (name || '').toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 4);
    if (clean.length !== 4) return;
    const payload = {
        type: this.ControlType.values.CONTROL_HIGH_SCORE_LOG,
        highScore: {
            name: clean,
        },
    };
    this.sendQ(payload);
    this.highScoreSubmitted = true;
    log(`[HudesClient] High Score submitted for ${clean}`);
}


zeroDimsAndStepsOnCurrentDims() {
    this.dimsAndStepsOnCurrentDims = new Array(this.state.n).fill(0);
    //console.log("Dims and steps reset to zero.");
}
dimsAndStepsUpdated() {
    // if (this.view && typeof this.view.updateDimsSinceLastUpdate === 'function') {
    //     this.view.updateDimsSinceLastUpdate(this.dimsAndStepsOnCurrentDims || []);
    // } else {
    //     console.warn("View or updateDimsSinceLastUpdate is not defined.");
    // }
}
    async sendDimsAndSteps(dimsAndSteps) {
    try {
        // Debug: Log control type status
        if (!this.Control || !this.ControlType) {
            console.error("Proto definitions not loaded. Cannot send dims and steps.");
            return;
        }
        if (!this._canSendUserCommands()) {
            log("[HudesClient] Ignoring dims send while socket unavailable", null, 'warn');
            return;
        }

        // Construct the payload
        const payload = {
            type: this.ControlType.values.CONTROL_DIMS,
            dimsAndSteps: Object.entries(dimsAndSteps).map(([dim, step]) => ({
                dim: parseInt(dim, 10),
                step,
            })),
        };

        // Debug: Log the payload before sending
        log(`[HudesClient] send DIMS dims=${payload.dimsAndSteps.map(x => `(${x.dim}:${x.step.toFixed?.(3) ?? x.step})`).join(', ')}`);

        // Use sendQ to handle message creation and sending
        this.sendQ(payload);

        // Simulate delay for synchronization
        await sleep(10);

        // Update local state
        for (const [dim, step] of Object.entries(dimsAndSteps)) {
            this.dimsAndStepsOnCurrentDims[dim] =
                (this.dimsAndStepsOnCurrentDims[dim] || 0) + step;
        }

        // Debug: Log the updated dimsAndStepsOnCurrentDims state
        //console.log("Updated dimsAndStepsOnCurrentDims:", this.dimsAndStepsOnCurrentDims);

        // Update the UI or internal state to reflect the changes
        this.dimsAndStepsUpdated();

    } catch (error) {
        console.error("Error in sendDimsAndSteps:", error);
    }
}


    async sendConfig() {
    if (!this.Control || !this.ControlType) {
        console.error("Proto definitions not loaded. Cannot send config.");
        return;
    }

    this.dimsOffset = 0;
    this.lossLineLabels = [];

    const payload = {
        type: this.ControlType.values.CONTROL_CONFIG,
        config: {
            seed: Math.floor(Math.random() * 1000),
            dimsAtATime: this.state.n,
            meshGridSize: this.grid_size,
            meshStepSize: this.state.stepSize,
            meshGrids: this.meshEnabled ? this.grids : 0,
            batchSize: this.state.batchSize,
            dtype: this.state.dtype,
            meshEnabled: this.meshEnabled,
            lossLines: this.renderMode === '1d' ? this.lossLines : 0,
            resumeSupported: true,
            clientSessionToken: this._clientSessionToken,
        },
    };

    // Use sendQ to send the message
    this.sendQ(payload);
    this._readyForInput = true;

    log(`[HudesClient] Config sent: mode=${this.renderMode} grids=${this.grids} lossLines=${this.lossLines} size=${this.grid_size} step=${this.state.stepSize} batch=${this.state.batchSize} dtype=${this.state.dtype} meshEnabled=${this.meshEnabled}`);
}



    async runLoop() {
        await this._protoReadyPromise;
        while (this.running) {
            await sleep(10);
        }
        log("[HudesClient] Exiting run loop...");
    }

}
