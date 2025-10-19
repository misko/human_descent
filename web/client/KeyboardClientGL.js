import KeyboardClient from './KeyboardClient.js';
import { log } from '../utils/logger.js';
import View from './View.js';
import { installMouseControls, computeStepVector } from './mouseControls.js';

const DEBUG_MOUSE = false;

const debugMouse = (message) => {
    if (DEBUG_MOUSE) {
        log(message);
    }
};

export default class KeyboardClientGL extends KeyboardClient {
    constructor(addr, port) {
        super(addr, port);
        this.initInput();

    if (this.state.helpScreenIdx != -1) {
        this.view.showImage(this.state.helpScreenFns[this.state.helpScreenIdx]);
    }
    }

    initInput() {
        super.initInput();

        // Additional state setup for GL
        this.stepSizeKeyboardMultiplier = 2.5;
        this.lastSelectPress = 0;
        this.keyCooldownMs = 200;

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

    processKeyPress(event) {
        const currentTime = performance.now();
        let redraw = this.processCommonKeys(event);

        if (this.state.helpScreenIdx !== -1) {
            this.view.showImage(this.state.helpScreenFns[this.state.helpScreenIdx]);
            return redraw; // Skip further processing if help screen is active
          }
          this.view.hideImage();


        if (event.type === 'keydown') {
            const key = event.key.toLowerCase();
            const { angleH } = this.view.getAngles(); // Get the current horizontal and vertical angles


            switch (event.code) {
                case 'KeyW': {
                    // Rotate vector (0, -1) by angleH
                    const rotated = this._rotateVector( -1,0, -angleH);
                    this.stepInSelectedGrid(rotated.x, rotated.y);
                    redraw = true;
                    break;
                }
                case 'KeyS': {
                    // Rotate vector (0, 1) by angleH
                    const rotated = this._rotateVector(1,0, -angleH);
                    this.stepInSelectedGrid(rotated.x, rotated.y);
                    redraw = true;
                    break;
                }
                case 'KeyA': {
                    // Rotate vector (-1, 0) by angleH
                    const rotated = this._rotateVector(0,-1, -angleH);
                    this.stepInSelectedGrid(rotated.x, rotated.y);
                    redraw = true;
                    break;
                }
                case 'KeyD': {
                    // Rotate vector (1, 0) by angleH
                    const rotated = this._rotateVector(0,1,-angleH);
                    this.stepInSelectedGrid(rotated.x, rotated.y);
                    redraw = true;
                    break;
                }
                case 'ArrowLeft':
                    this.view.adjustAngles(-5, 0);
                    redraw = true;
                    break;
                case 'ArrowRight':
                    this.view.adjustAngles(5, 0);
                    redraw = true;
                    break;
                case 'ArrowUp':
                    this.view.adjustAngles(0, 2.5);
                    redraw = true;
                    break;
                case 'ArrowDown':
                    this.view.adjustAngles(0, -2.5);
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
