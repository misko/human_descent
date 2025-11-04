import KeyboardClient from './KeyboardClient.js';
import { log } from '../utils/logger.js';
import { installMouseControls, computeStepVector } from './mouseControls.js';

const DEBUG_MOUSE = false;

const debugMouse = (message) => {
    if (DEBUG_MOUSE) {
        log(message);
    }
};

export default class KeyboardClientGL extends KeyboardClient {
    constructor(addr, port, options = {}) {
        super(addr, port, { ...options, renderMode: '3d' });
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

        this.dragSensitivity = 0.3;
        this._mouseControls = null;

        this.view.resetAngle(); // Set initial angles

        const canvas = this.view.getCanvasElement?.();
        if (canvas) {
            debugMouse('[KeyboardClientGL] installing mouse controls');
            this._mouseControls = installMouseControls(canvas, {
                moveTarget: typeof window !== 'undefined' ? window : canvas,
                onDrag: ({ deltaX, deltaY }) => {
                    if (this.state.helpScreenIdx !== -1) {
                        debugMouse('[KeyboardClientGL] drag ignored while help screen active');
                        return;
                    }
                    const horizontal = deltaX * this.dragSensitivity;
                    const vertical = -deltaY * this.dragSensitivity;

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

    updateKeyHolds() {
        // Handle key up events here if needed
    }
}
