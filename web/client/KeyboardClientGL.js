import KeyboardClient from './KeyboardClient.js';
import { log } from '../utils/logger.js';
import { installMouseControls, computeStepVector } from './mouseControls.js';
import { installTouchControls } from './touchControls.js';

const DEBUG_MOUSE = false;

const debugMouse = (message) => {
    if (DEBUG_MOUSE) {
        log(message);
    }
};

export default class KeyboardClientGL extends KeyboardClient {
    constructor(addr, port, options = {}) {
        const mergedOptions = { ...options, renderMode: '3d' };
        if (options.isMobile) {
            mergedOptions.meshGrids = 1;
        }
        super(addr, port, mergedOptions);
        this.isMobile = Boolean(options.isMobile);
        this.initInput();

    if (this.state.helpScreenIdx != -1) {
        this.view.showImage(this.state.helpScreenFns[this.state.helpScreenIdx]);
    }
    }

    initInput() {
        super.initInput();

        // Disable 1D paired key controls (e/d, r/f, etc.) in GL mode.
        // GL uses its own controls (WASD, arrows, mouse drag/scroll),
        // so we clear the base mapping to avoid accidental dimension steps.
        this.pairedKeys = [];
        this.keyToParamAndSign = {};

        // Additional state setup for GL
        this.stepSizeKeyboardMultiplier = 2.5;
        this.lastSelectPress = 0;
        this.keyCooldownMs = 200;
        this.directionalKeys = new Set(['w', 's', 'a', 'd']);
        this.angleKeys = new Set(['arrowleft', 'arrowright', 'arrowup', 'arrowdown']);

        this.dragSensitivity = 0.18;
        this.verticalTiltDragFraction = 0.75; // portion of window height mapping to full tilt range
        this._mouseControls = null;
        this._touchControls = null;
        this._touchRotateCleanup = null;
        this._boundTouchRotatePageHide = null;
        this._touchRotateCleanup = null;

        this.view.resetAngle(); // Set initial angles

        const canvas = this.view.getCanvasElement?.();
        if (canvas) {
            if (this.isMobile) {
                this._touchControls = installTouchControls(canvas, {
                    onVector: (vector) => this._applyTouchVector(vector),
                });
                this.__applyTouchVector = (vector) => this._applyTouchVector(vector);
                this._installMobileUi();
                this._installTouchRotationControls(canvas);
                this._installTouchRotationControls(canvas);
            } else {
                debugMouse('[KeyboardClientGL] installing mouse controls');
                this._mouseControls = installMouseControls(canvas, {
                    moveTarget: typeof window !== 'undefined' ? window : canvas,
                    onDrag: ({ deltaX, deltaY }) => {
                    if (this.state.helpScreenIdx !== -1) {
                        debugMouse('[KeyboardClientGL] drag ignored while help screen active');
                        return;
                    }
                    const horizontal = deltaX * this.dragSensitivity;
                    const verticalSensitivity = this._getVerticalDragSensitivity();
                    const vertical = -deltaY * verticalSensitivity;

                    const before = this.view.getAngles();
                    debugMouse(
                        `[KeyboardClientGL] drag deltaX=${deltaX} deltaY=${deltaY} horizontal=${horizontal.toFixed(2)} vertical=${vertical.toFixed(2)} angleH=${before.angleH.toFixed(2)} angleV=${before.angleV.toFixed(2)}`,
                    );

                    if (horizontal) {
                        this.view.adjustAngles(horizontal, 0);
                    }
                    if (vertical) {
                        this.view.adjustAngles(0, vertical);
                    }

                    const after = this.view.getAngles();
                    debugMouse(
                        `[KeyboardClientGL] updated angles angleH=${after.angleH.toFixed(2)} angleV=${after.angleV.toFixed(2)}`,
                    );
                },
                onClick: ({ event }) => {
                    if (this.state.helpScreenIdx !== -1) {
                        debugMouse('[KeyboardClientGL] click ignored while help screen active');
                        return;
                    }
                    const index = this.view.selectGridAt?.(event.clientX, event.clientY);
                    if (typeof index === 'number') {
                        debugMouse(`[KeyboardClientGL] click selected grid ${index}`);
                    }
                },
                onScroll: ({ deltaY }) => {
                    if (this.state.helpScreenIdx !== -1) {
                        debugMouse('[KeyboardClientGL] scroll ignored while help screen active');
                        return;
                    }
                    if (!deltaY) {
                        debugMouse('[KeyboardClientGL] scroll with zero delta ignored');
                        return;
                    }

                    const direction = deltaY < 0 ? 'backward' : 'forward';
                    const { angleH } = this.view.getAngles();
                    const rotated = computeStepVector(angleH, direction);
                    debugMouse(
                        `[KeyboardClientGL] scroll deltaY=${deltaY} direction=${direction} angleH=${angleH.toFixed(2)} stepX=${rotated.x.toFixed(3)} stepY=${rotated.y.toFixed(3)}`,
                    );
                    this.stepInSelectedGrid(rotated.x, rotated.y);
                },
                onContext: ({ event }) => {
                    if (this.state.helpScreenIdx !== -1) {
                        debugMouse('[KeyboardClientGL] context click ignored while help screen active');
                        return;
                    }
                    debugMouse('[KeyboardClientGL] context click cycling grid');
                    this.view.incrementSelectedGrid();
                    this.view.updateMeshGrids();
                    event.preventDefault?.();
                },
                });
                debugMouse('[KeyboardClientGL] mouse controls installed');
            }
        } else {
            log('[KeyboardClientGL] renderer canvas missing; mouse controls disabled');
        }

        // Event listeners for keypresses
        //window.addEventListener('keydown', (event) => this.processKeyPress(event));
        //window.addEventListener('keyup', (event) => this.updateKeyHolds());

                this.state.setHelpScreenFns([
            "help_screens/hudes_help_start.png",
            "help_screens/hudes_1.png",
            "help_screens/hudes_2.png",
            "help_screens/hudes_3.png",
            "help_screens/hudes_4.png",
            "help_screens/hudes_5.png",
            "help_screens/hudes_6.png",
            "help_screens/hudes_7.png",
            "help_screens/hudes_8.png",
            "help_screens/hudes_9.png",
            "help_screens/hudes_2d_keyboard_controls.png",
            "help_screens/hudes_2d_xbox_controls.png",
          ]);

    }

      // Step in the selected grid with given s0 and s1 values
      stepInSelectedGrid(s0, s1) {
        try {
            const selectedGrid = this.view.getSelectedGrid();
            const stepSize = this.state.stepSize;

            // Prepare the steps dictionary similar to the Python version
            const dimsAndSteps = {
                [1 + selectedGrid * 2]: s1 * stepSize * 0.25,
                [0 + selectedGrid * 2]: s0 * stepSize * 0.25,
            };


            // Send the dimensions and steps
            this.sendDimsAndSteps(dimsAndSteps);
        } catch (error) {
            console.error("Error in stepInSelectedGrid:", error);
        }
    }


    _rotateVector(x, y, angleDeg) {
        const angleRad = (Math.PI / 180) * angleDeg; // Convert angle to radians
        const cos = Math.cos(angleRad);
        const sin = Math.sin(angleRad);

        // Rotate (x, y) by angleDeg
        return {
            x: x * cos - y * sin,
            y: x * sin + y * cos,
        };
    }

    _performDirectionalStep(baseX, baseY) {
        if (this.state.helpScreenIdx !== -1) {
            return;
        }
        const { angleH } = this.view.getAngles();
        const rotated = this._rotateVector(baseX, baseY, -angleH);
        this.stepInSelectedGrid(rotated.x, rotated.y);
    }

    _startDirectionalRepeat(key, baseX, baseY) {
        this._startRepeat(key, () => this._performDirectionalStep(baseX, baseY));
    }

    _performAngleAdjust(deltaH, deltaV) {
        if (this.state.helpScreenIdx !== -1) {
            return;
        }
        this.view.adjustAngles(deltaH, deltaV);
    }

    _startAngleRepeat(key, deltaH, deltaV) {
        this._startRepeat(key, () => this._performAngleAdjust(deltaH, deltaV));
    }

    processKeyPress(event) {
        if (event.metaKey || event.ctrlKey) {
            return this.processCommonKeys(event);
        }
        const currentTime = performance.now();
        let redraw = this.processCommonKeys(event);

        if (this.state.helpScreenIdx !== -1) {
            this.view.showImage(this.state.helpScreenFns[this.state.helpScreenIdx]);
            return redraw; // Skip further processing if help screen is active
          }
          this.view.hideImage();

        if (event.type === 'keydown') {
            const key = event.key.toLowerCase();
            if ((this.directionalKeys.has(key) || this.angleKeys.has(key)) && event.repeat) {
                event.preventDefault?.();
                return true;
            }


            switch (event.code) {
                case 'KeyW': {
                    this._performDirectionalStep(-1, 0);
                    this._startDirectionalRepeat(key, -1, 0);
                    redraw = true;
                    break;
                }
                case 'KeyS': {
                    this._performDirectionalStep(1, 0);
                    this._startDirectionalRepeat(key, 1, 0);
                    redraw = true;
                    break;
                }
                case 'KeyA': {
                    this._performDirectionalStep(0, -1);
                    this._startDirectionalRepeat(key, 0, -1);
                    redraw = true;
                    break;
                }
                case 'KeyD': {
                    this._performDirectionalStep(0, 1);
                    this._startDirectionalRepeat(key, 0, 1);
                    redraw = true;
                    break;
                }
                case 'ArrowLeft':
                    this._performAngleAdjust(-5, 0);
                    this._startAngleRepeat(key, -5, 0);
                    redraw = true;
                    break;
                case 'ArrowRight':
                    this._performAngleAdjust(5, 0);
                    this._startAngleRepeat(key, 5, 0);
                    redraw = true;
                    break;
                case 'ArrowUp':
                    this._performAngleAdjust(0, 2.5);
                    this._startAngleRepeat(key, 0, 2.5);
                    redraw = true;
                    break;
                case 'ArrowDown':
                    this._performAngleAdjust(0, -2.5);
                    this._startAngleRepeat(key, 0, -2.5);
                    redraw = true;
                    break;
                case 'ShiftRight':
                        this.view.incrementSelectedGrid();
                        this.view.updateMeshGrids()
                        //this.view.updatePointsAndColors();
                        redraw = true;
                    break;
                case 'ShiftLeft':
                        this.view.decrementSelectedGrid();
                        this.view.updateMeshGrids()
                        //this.view.updatePointsAndColors();
                        redraw = true;
                    break;
                case ' ': // Space key for new random projection
                    this.view.incrementSelectedGrid();
                    this.view.updatePointsAndColors();
                    redraw = true;
                    break;
                default:
                    // Handle paired keys from the base class
                    if (this.keyToParamAndSign[key]) {
                        const { dim, sign } = this.keyToParamAndSign[key];
                        if (
                            this.checkCooldownTime(
                                event,
                                key,
                                currentTime,
                                this.cooldownTimeMs / 1000
                            )
                        ) {
                            this.sendDimsAndSteps({ [dim]: this.state.stepSize * sign });
                        }
                    }
                    break;
            }
        }
        return redraw;
    }

    _getVerticalDragSensitivity() {
        const impl = this.view?.impl;
        const maxAngleV = typeof impl?.maxAngleV === 'number' ? impl.maxAngleV : 25;
        const height = typeof window !== 'undefined' && window.innerHeight > 0 ? window.innerHeight : 800;
        const dragSpan = Math.max(1, height * (this.verticalTiltDragFraction ?? 0.5));
        // adjustAngles multiplies deltas by 2, so use half the desired slope here
        return (2 * maxAngleV) / dragSpan;
    }

    _applyTouchVector(vector) {
        if (!vector || this.state.helpScreenIdx !== -1) {
            return;
        }
        const rawX = Number(vector.x) || 0;
        const rawY = Number(vector.y) || 0;
        if (!rawX && !rawY) {
            return;
        }
        const clamp = (v) => Math.max(-1, Math.min(1, v));
        const baseX = clamp(-rawY);
        const baseY = clamp(-rawX);
        const magnitude = Math.min(1, Math.hypot(baseX, baseY));
        if (magnitude < 0.05) {
            return;
        }
        const { angleH } = this.view.getAngles();
        const rotated = this._rotateVector(baseX, baseY, -angleH);
        this.stepInSelectedGrid(-rotated.x, -rotated.y);
    }

    _installTouchRotationControls(canvas) {
        if (!canvas || typeof window === 'undefined') {
            return;
        }
        this._cleanupTouchRotation();
        const state = { active: false, prevX: 0, prevY: 0 };
        const getMidpoint = (touchList) => {
            if (!touchList || touchList.length < 2) {
                return null;
            }
            const t1 = touchList[0];
            const t2 = touchList[1];
            return {
                x: (t1.clientX + t2.clientX) / 2,
                y: (t1.clientY + t2.clientY) / 2,
            };
        };
        const handleStart = (event) => {
            if ((event.touches?.length || 0) >= 2) {
                const mid = getMidpoint(event.touches);
                if (mid) {
                    state.active = true;
                    state.prevX = mid.x;
                    state.prevY = mid.y;
                    event.preventDefault?.();
                }
            }
        };
        const handleMove = (event) => {
            if (!state.active) {
                return;
            }
            if ((event.touches?.length || 0) < 2) {
                state.active = false;
                return;
            }
            const mid = getMidpoint(event.touches);
            if (!mid) {
                return;
            }
            const dx = mid.x - state.prevX;
            const dy = mid.y - state.prevY;
            state.prevX = mid.x;
            state.prevY = mid.y;
            if (dx || dy) {
                this._handleTouchRotationDrag(dx, dy);
            }
            event.preventDefault?.();
        };
        const endGesture = () => {
            state.active = false;
        };
        const handleEnd = () => {
            endGesture();
        };
        const handleCancel = () => {
            endGesture();
        };
        canvas.addEventListener('touchstart', handleStart, { passive: false });
        canvas.addEventListener('touchmove', handleMove, { passive: false });
        canvas.addEventListener('touchend', handleEnd, { passive: false });
        canvas.addEventListener('touchcancel', handleCancel, { passive: false });
        this._touchRotateCleanup = () => {
            canvas.removeEventListener('touchstart', handleStart);
            canvas.removeEventListener('touchmove', handleMove);
            canvas.removeEventListener('touchend', handleEnd);
            canvas.removeEventListener('touchcancel', handleCancel);
        };
        if (typeof window !== 'undefined' && !this._boundTouchRotatePageHide) {
            this._boundTouchRotatePageHide = () => this._cleanupTouchRotation();
            window.addEventListener('pagehide', this._boundTouchRotatePageHide, { passive: true });
        }
    }

    _handleTouchRotationDrag(deltaX, deltaY) {
        if (this.state.helpScreenIdx !== -1) {
            return;
        }
        const horizontal = deltaX * this.dragSensitivity;
        const verticalSensitivity = this._getVerticalDragSensitivity();
        const vertical = -deltaY * verticalSensitivity;
        if (horizontal) {
            this.view.adjustAngles(horizontal, 0);
        }
        if (vertical) {
            this.view.adjustAngles(0, vertical);
        }
    }

    _cleanupTouchRotation() {
        if (typeof window === 'undefined') {
            return;
        }
        if (this._touchRotateCleanup) {
            try {
                this._touchRotateCleanup();
            } catch {}
            this._touchRotateCleanup = null;
        }
        if (this._boundTouchRotatePageHide) {
            window.removeEventListener('pagehide', this._boundTouchRotatePageHide);
            this._boundTouchRotatePageHide = null;
        }
    }

    updateKeyHolds() {
        // Handle key up events here if needed
    }

    _installMobileUi() {
        if (this._mobileUi) {
            return;
        }
        const panel = document.createElement('div');
        panel.id = 'mobileActionPanel';
        const grid = document.createElement('div');
        grid.className = 'mobile-panel__grid';
        panel.appendChild(grid);

        const buttons = {};
        const addButton = (key, handler) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.dataset.mobileAction = key;
            btn.addEventListener('click', handler);
            grid.appendChild(btn);
            buttons[key] = btn;
            return btn;
        };

        addButton('speed', () => {
            if (this.speedRunActive || this.state.speedRunActive) {
                this.cancelSpeedRun();
            } else {
                this.startSpeedRun();
            }
            this._updateMobileUi();
        });

        addButton('batch', () => {
            this.state.toggleBatchSize();
            this.sendConfig();
            this._updateMobileUi();
        });

        addButton('sgd', () => {
            this.getSGDStep();
        });

        const matrixContainer = document.getElementById('confusionMatrixChart');
        if (matrixContainer?.parentElement) {
            matrixContainer.parentElement.appendChild(panel);
        } else {
            document.body.appendChild(panel);
        }
        this._mobileUi = { panel, buttons };
        this._updateMobileUi();
        this._mobileUiTicker = setInterval(() => this._updateMobileUi(), 800);
    }

    _cycleHelpScreens() {
        this.state.nextHelpScreen();
        if (this.state.helpScreenIdx !== -1) {
            const idx = Math.min(this.state.helpScreenIdx, (this.state.helpScreenFns?.length ?? 1) - 1);
            const fn = this.state.helpScreenFns?.[idx];
            if (fn) {
                this.view.showImage(fn);
            }
        } else {
            this.view.hideImage();
        }
    }

    _updateMobileUi() {
        if (!this._mobileUi) {
            return;
        }
        const { buttons } = this._mobileUi;
        const setText = (btn, text) => {
            if (!btn || btn.textContent === text) {
                return;
            }
            btn.textContent = text;
        };
        setText(
            buttons.speed,
            this.state.speedRunActive
                ? `🔥 ${Number(this.state.speedRunSecondsRemaining ?? 0).toFixed(1)}s`
                : 'SPEED 🔥'
        );
        setText(buttons.batch, `Batch ${this.state.batchSize ?? ''}`);
        setText(buttons.sgd, 'SGD');
        if (buttons.sgd) {
            buttons.sgd.disabled = Boolean(this.state.speedRunActive);
        }
    }

}
