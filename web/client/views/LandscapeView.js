import {
    Scene,
    PerspectiveCamera,
    MathUtils,
    WebGLRenderer,
    Color,
    PlaneGeometry,
    BufferAttribute,
    BufferGeometry,
    ShaderMaterial,
    Mesh,
    SphereGeometry,
    MeshBasicMaterial,
    Raycaster,
    Vector2,
    Line,
    LineBasicMaterial,
    Group,
    LineSegments,
    LineDashedMaterial,
} from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { OutlinePass } from 'three/examples/jsm/postprocessing/OutlinePass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; // Import OrbitControls
import annotationPlugin from 'chartjs-plugin-annotation';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js'; // Import CSS2DRenderer for annotations
import { CONTROL_GROUPS, formatHudMarkup, MOBILE_CONTROL_GROUPS, MOBILE_HUD_BUTTONS, escapeHtml } from '../hud.js';
import {
    BRAND_TAGLINE,
    SPLASH_LOGO_URL,
    SPLASH_MEDIA_URL,
    WELCOME_BULLETS,
    HOW_IT_WORKS_POINTS,
    MODE_CARDS,
    DESKTOP_CONTROLS_GRID,
    MOBILE_CONTROLS_GRID,
    TUTORIAL_TASKS,
    SPEEDRUN_RULES,
    SPEEDRUN_COMPACT_KEYS,
    SPEEDRUN_RULES_MOBILE,
    SPEEDRUN_COMPACT_KEYS_MOBILE,
    TOUR_FLOWS,
    TOUR_SCREENS,
    SHARE_TEXT,
} from '../helpTour.js';
import { log } from '../../utils/logger.js'; // Import your logging utility

const DEBUG_SELECTION = false;

const debugSelection = (message) => {
    if (DEBUG_SELECTION) {
        log(message);
    }
};

const CONFUSION_LABELS = Array.from({ length: 10 }, (_, i) => i.toString());
const HEATMAP_STOPS = [
    { stop: 0, color: [33, 11, 68] },
    { stop: 0.5, color: [0, 168, 168] },
    { stop: 1, color: [255, 221, 87] },
];

const lerp = (a, b, t) => a + (b - a) * t;
const colorToRgba = ([r, g, b], alpha = 0.95) =>
    `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`;

const interpolateHeatmapColor = (valueRatio) => {
    const t = Math.min(1, Math.max(0, valueRatio));
    for (let i = 0; i < HEATMAP_STOPS.length - 1; i += 1) {
        const left = HEATMAP_STOPS[i];
        const right = HEATMAP_STOPS[i + 1];
        if (t >= left.stop && t <= right.stop) {
            const localT = (t - left.stop) / (right.stop - left.stop || 1);
            return colorToRgba([
                lerp(left.color[0], right.color[0], localT),
                lerp(left.color[1], right.color[1], localT),
                lerp(left.color[2], right.color[2], localT),
            ]);
        }
    }
    return colorToRgba(HEATMAP_STOPS[HEATMAP_STOPS.length - 1].color);
};

import {
    Chart,
    CategoryScale,    // Register the category scale
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    LineController,
    BarController,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

import { MatrixController, MatrixElement } from 'chartjs-chart-matrix';

// Register all necessary components
Chart.register(
    CategoryScale,
    LinearScale,
    LineController,
    BarController,
    MatrixController,  // Register MatrixController
    MatrixElement,     // Register MatrixElement
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend,
    annotationPlugin
);


export default class LandscapeView {
    constructor(gridSize, numGrids, clientState, options = {}) {
        this.mode = (options.mode || '3d').toLowerCase();
        this.gridSize = gridSize;
        this.numGrids = numGrids;
        this.effectiveGrids = numGrids;
        this.debug = Boolean(options.debug);
        this.alt1dMode = this.mode === '1d' && Boolean(options.alt1d);
        this.altKeysMode = Boolean(options.altKeys);

        this.state = clientState;
        this.mobileMode = Boolean(options.mobile);
        this.hideHud = Boolean(options.hideHud);
        this.initialTourDisabled = Boolean(options.disableInitialTour) || this.state?.helpScreenIdx === -1;
        if (typeof document !== 'undefined') {
            const bottomHud = document.getElementById('bottomTextContainer');
            if ((this.mobileMode || this.hideHud) && bottomHud) {
                bottomHud.remove();
            }
        }
        if (this.mobileMode && this.numGrids > 1) {
            this.numGrids = 1;
            this.effectiveGrids = 1;
        }

        this.lossChart = null;
        this.lastStepsChart = null;
        this.stepSizeChart = null;
        this.dimsAndStepsChart = null;
        this.exampleImagesContainer = document.getElementById('exampleImages');
        this.confusionMatrixContainer = document.getElementById('confusionMatrixChart');
        this.confusionMatrixChart = null;
        this.confusionMatrixMaxValue = 1;
        this.confusionRows = 10;
        this.confusionCols = 10;
        this.exampleCharts = [];
        this.helpOverlay = null;
        this.helpOverlayContent = null;
        this.helpTabsBar = null;
        this.helpTabs = {};
        this.helpTourSelection = 'explore';
        this.selectedTourMode = 'explore';
        this.manualKeyOverlay = false;
        this._helpKeyListenerAttached = false;
        this._boundHelpKeydown = (event) => this._handleHelpKeydown(event);
        this.helpContentMode = 'tour';
        this.activeHelpTab = 'controls';
        this.currentTourFlow = 'initial';
        this.tutorialProgress = new Map();
        this._resetTutorialProgress();
        this.tutorialToastEl = null;
        this.tutorialComplete = false;
        if (!this.initialTourDisabled && typeof window !== 'undefined') {
            window.requestAnimationFrame(() => this._setTourFlow('initial'));
        }
        this.levelToastTimeout = null;
        this.levelModalTimeout = null;
        this._levelDismissHandler = null;


        // Grid properties
        this.gridObjects = [];
        this.lineGroup = null;
        this.lineObjects = [];
        this.lineFrames = [];
        this.lineContainers = [];
        this.centerLines = [];
        this.horizontalLines = [];
        this.lineScaleCache = [];
        this.lineBaseColors = [];
        this.frameGlowState = [];
        this.alt1dContainers = [];
        this.alt1dFrames = [];
        this.alt1dCenterLines = [];
        this.alt1dHorizontalLines = [];
        this.alt1dGlowStates = [];
        this.alt1dFrameBaseColors = [];
        this.alt1dLineToContainer = [];
        this.outlineSelection = new Set();
        this.glowDuration = 150;
        this.glowExpand = 0.05;
        this.glowEdgeStrength = 3.5;
        this.tempColor = new Color();
        this.highlightColor = new Color(1, 1, 1);
        this.lineHeightBase = this.gridSize * 0.4;
        this.lineHeightMin = Math.max(1, this.gridSize * 0.05);
        this.lineHeightMax = this.gridSize * 0.35;
        this.rowSpacing = typeof options.rowSpacing === 'number' ? options.rowSpacing : null;
        this.depthStep = typeof options.depthStep === 'number' ? options.depthStep : null;
        this.customCameraDistance = typeof options.cameraDistance === 'number' ? options.cameraDistance : null;
        this.spacing = 20; // Spacing between grids
        this.mobileCameraScalar = this.mobileMode ? this._computeMobileCameraScalar() : 1;
        this.selectedGridIndex = 0; // Index of the selected grid
        this.selectedGridScale = 2 * 1.5; //1.5; // Scale multiplier for the selected grid
        // Three.js setup
        const glContainer = document.getElementById('glContainer');
        if (glContainer) {
            this.scene = new Scene();
            this.camera = new PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            this.renderer = new WebGLRenderer({ antialias: true });
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.renderer.domElement.style.position = 'absolute';
            this.renderer.domElement.style.top = '0';
            this.renderer.domElement.style.left = '0';
            this.renderer.domElement.style.width = '100%';
            this.renderer.domElement.style.height = '100%';
            this.renderer.domElement.style.pointerEvents = 'auto';
            glContainer.appendChild(this.renderer.domElement);
            // Calculate camera position to fit all grids
            const fovRadians = (this.camera.fov * Math.PI) / 180;
            const totalWidth = (this.numGrids + this.selectedGridScale - 1.0) * this.gridSize + (this.numGrids - 1) * this.spacing;
            this.camera_distance = (totalWidth / 2) / Math.tan(fovRadians / 2);
            if (this.customCameraDistance != null) {
                this.camera_distance = this.customCameraDistance;
            }
            if (this.alt1dMode) {
                this.camera_distance *= 1.25;
            }
            if (this.mobileMode) {
                this.camera_distance *= this.mobileCameraScalar;
            }
            if (this.mobileMode) {
                this.camera_distance *= this.mobileCameraScalar;
            }

            this.raycaster = new Raycaster();
            this.pointer = new Vector2();

            // Set the y-offset for the camera to position the grids 2/3 of the way down the view
            // The y offset is calculated as (1/3 of the totalHeight) since we want the grids to be 2/3 of the way down
            this.camera_yOffset = this.mobileMode ? (totalWidth / 6) * this.mobileCameraScalar * 0.85 : totalWidth / 8;

            // Update camera position
            this.camera.position.set(0, this.camera_yOffset, this.camera_distance);
            // Orbit controls setup
            //this.controls = new OrbitControls(this.camera, this.renderer.domElement);


            // CSS2DRenderer setup for annotations
            this.labelRenderer = new CSS2DRenderer();
            this.labelRenderer.setSize(window.innerWidth, window.innerHeight);
            this.labelRenderer.domElement.style.position = 'absolute';
            this.labelRenderer.domElement.style.top = '0px';
            this.labelRenderer.domElement.style.pointerEvents = 'none';
            document.body.appendChild(this.labelRenderer.domElement);

            this.composer = new EffectComposer(this.renderer);
            this.renderPass = new RenderPass(this.scene, this.camera);
            this.composer.addPass(this.renderPass);
            this.outlinePass = new OutlinePass(new Vector2(window.innerWidth, window.innerHeight), this.scene, this.camera);
            this.outlinePass.edgeStrength = 0;
            this.outlinePass.edgeGlow = 0.6;
            this.outlinePass.edgeThickness = 1.0;
            this.outlinePass.visibleEdgeColor.set(1, 1, 1);
            this.outlinePass.hiddenEdgeColor.set(0, 0, 0);
            this.outlinePass.pulsePeriod = 0;
            this.composer.addPass(this.outlinePass);

            // Colors for the grids
            this.gridColors = [
                new Color(0.0, 1.0, 1.0), // Cyan
                new Color(1.0, 0.0, 1.0), // Magenta
                new Color(1.0, 1.0, 0.0), // Yellow
                new Color(0.0, 1.0, 0.0), // Green
                new Color(1.0, 0.5, 0.0), // Orange
                new Color(1.0, 0.0, 0.0), // Red
                new Color(0.0, 0.0, 1.0), // Blue
                new Color(0.5, 0.0, 1.0), // Purple
                new Color(0.0, 0.5, 1.0), // Sky Blue
                new Color(1.0, 0.0, 0.5), // Pink
                new Color(0.5, 1.0, 0.0), // Lime
                new Color(1.0, 0.75, 0.8), // Light Pink
            ];

            this.alpha = 0.8; // Transparency

            // Angle properties
            this.defaultAngleV = 20;
            this.maxAngleV = 25;
            this.angleH = 0.0;
            this.angleV = this.defaultAngleV;
            // Start rendering loop
            this.render();
            window.addEventListener('resize', this.onWindowResize.bind(this), false);
        }
    }

    _computeMobileCameraScalar() {
        if (typeof window === 'undefined') {
            return 1.35;
        }
        const shortRef = 760;
        const longRef = 1280;
        const width = window.innerWidth || shortRef;
        const height = window.innerHeight || longRef;
        const shortSide = Math.min(width, height) || shortRef;
        const longSide = Math.max(width, height) || longRef;
        const shortRatio = shortRef / shortSide;
        const longRatio = longRef / longSide;
        const scalar = Math.min(1.6, Math.max(1.2, Math.max(shortRatio, longRatio)));
        return scalar;
    }

    _updateAlt1dLossLines(lines, { stepSpacing, labels } = {}) {
        if (!this.alt1dMode || !this.scene || !Array.isArray(lines) || !lines.length) {
            return;
        }
        if (!this.lineGroup) {
            this.lineGroup = new Group();
            this.scene.add(this.lineGroup);
        }

        const count = lines.length;
        const linesPerPlot = 3;
        const containerCount = Math.max(1, Math.ceil(count / linesPerPlot));
        this.effectiveGrids = containerCount;
        const scaleFactor = 2;
        const cellWidth = this.gridSize * 0.9 * scaleFactor;
        const cellHeight = this.gridSize * 0.6 * scaleFactor;
        const halfWidth = cellWidth / 2;
        const halfHeight = cellHeight / 2;
        const gapX = 1;

        for (let idx = 0; idx < containerCount; idx += 1) {
            this._ensureAlt1dScaffold(idx, {
                cellWidth,
                cellHeight,
                halfWidth,
                halfHeight,
            });
        }

        while (this.alt1dContainers.length > containerCount) {
            const container = this.alt1dContainers.pop();
            const frame = this.alt1dFrames.pop();
            const centerLine = this.alt1dCenterLines.pop();
            const horizontalLine = this.alt1dHorizontalLines.pop();
            this.alt1dFrameBaseColors.pop();
            if (frame) {
                this.outlineSelection.delete(frame);
                frame.geometry?.dispose?.();
                frame.material?.dispose?.();
            }
            centerLine?.geometry?.dispose?.();
            centerLine?.material?.dispose?.();
            horizontalLine?.geometry?.dispose?.();
            horizontalLine?.material?.dispose?.();
            if (container) {
                container.children.slice().forEach((child) => {
                    if (child instanceof Line || child instanceof LineSegments) {
                        if (child !== frame && child !== centerLine && child !== horizontalLine) {
                            container.remove(child);
                        }
                    }
                });
                this.lineGroup.remove(container);
            }
        }

        for (let idx = 0; idx < this.alt1dContainers.length; idx += 1) {
            const container = this.alt1dContainers[idx];
            if (container) {
                const offset = (idx - (this.alt1dContainers.length - 1) / 2) * (cellWidth + gapX);
                container.position.set(offset, 0, 0);
            }
        }

        while (this.lineObjects.length > count) {
            const line = this.lineObjects.pop();
            this.lineScaleCache.pop();
            this.lineBaseColors.pop();
            this.alt1dGlowStates.pop();
            this.alt1dLineToContainer.pop();
            if (line) {
                line.parent?.remove(line);
                line.geometry?.dispose?.();
                line.material?.dispose?.();
            }
        }

        while (this.lineObjects.length < count) {
            const material = new LineBasicMaterial({
                color: 0xffffff,
                linewidth: 2.5,
                transparent: true,
                opacity: 1,
            });
            material.depthTest = false;
            material.depthWrite = false;
            const geometry = new BufferGeometry();
            const line = new Line(geometry, material);
            line.renderOrder = 2;
            this.lineObjects.push(line);
            this.lineScaleCache.push(0);
            this.lineBaseColors.push(new Color(1, 1, 1));
            this.alt1dGlowStates.push(null);
            this.alt1dLineToContainer.push(0);
        }

        this.lineScaleCache.length = count;
        this.lineBaseColors.length = count;
        this.alt1dGlowStates.length = count;
        this.alt1dLineToContainer.length = count;

        const eps = 1e-6;
        const lerpAlpha = 0.35;

        for (let i = 0; i < count; i += 1) {
            const data = lines[i];
            const line = this.lineObjects[i];
            if (!Array.isArray(data) || !line) {
                continue;
            }
            if (this.alt1dGlowStates[i] === undefined) {
                this.alt1dGlowStates[i] = null;
            }

            const containerIdx = Math.min(
                this.alt1dContainers.length - 1,
                Math.floor(i / linesPerPlot),
            );
            const container = this.alt1dContainers[containerIdx];
            if (!container) {
                continue;
            }
            if (line.parent !== container) {
                line.parent?.remove(line);
                container.add(line);
            }
            this.alt1dLineToContainer[i] = containerIdx;

            const length = data.length;
            if (!length) {
                continue;
            }

            const mid = (length - 1) / 2;
            const baseline = data[Math.max(0, Math.floor(mid))];
            let maxAbs = 1e-6;
            for (let j = 0; j < length; j += 1) {
                maxAbs = Math.max(maxAbs, Math.abs(data[j] - baseline));
            }

            const scaleX = length > 1 ? cellWidth / (length - 1) : cellWidth;
            let scaleY = 0;
            if (maxAbs > eps) {
                const targetScale = halfHeight / maxAbs;
                const previous = this.lineScaleCache[i] ?? targetScale;
                const lerped = MathUtils.lerp(previous, targetScale, lerpAlpha);
                scaleY = Math.min(lerped, targetScale);
                this.lineScaleCache[i] = scaleY;
            } else {
                this.lineScaleCache[i] = 0;
            }

            let geometry = line.geometry;
            if (!(geometry instanceof BufferGeometry)) {
                geometry = new BufferGeometry();
                line.geometry = geometry;
            }

            let positionAttr = geometry.getAttribute('position');
            if (!positionAttr || positionAttr.array.length !== length * 3) {
                const positions = new Float32Array(length * 3);
                positionAttr = new BufferAttribute(positions, 3);
                geometry.setAttribute('position', positionAttr);
            }

            const positions = positionAttr.array;
            for (let j = 0; j < length; j += 1) {
                const idx = j * 3;
                positions[idx] = (j - mid) * scaleX;
                positions[idx + 1] = scaleY > 0 ? (data[j] - baseline) * scaleY : 0;
                positions[idx + 2] = 0;
            }

            positionAttr.needsUpdate = true;
            geometry.computeBoundingSphere();
            geometry.setDrawRange(0, length);

            const color = this.gridColors[i % this.gridColors.length];
            const baseColor = color.clone();
            this.lineBaseColors[i] = baseColor;
            line.material.color.copy(color);
            line.material.linewidth = 2.5;
            line.material.opacity = 1;
            line.material.needsUpdate = true;
            line.material.depthTest = false;
            line.material.depthWrite = false;
        }

        this.frameGlowState = [];
        this.lineFrames = [];
        this.centerLines = [];
        this.horizontalLines = [];
        this.lineContainers = [...this.alt1dContainers];
    }

    _ensureAlt1dScaffold(index, dims) {
        const { cellWidth, cellHeight, halfWidth, halfHeight } = dims;
        while (this.alt1dContainers.length <= index) {
            this.alt1dContainers.push(null);
            this.alt1dFrames.push(null);
            this.alt1dCenterLines.push(null);
            this.alt1dHorizontalLines.push(null);
            this.alt1dFrameBaseColors.push(null);
        }

        let container = this.alt1dContainers[index];
        if (!container) {
            container = new Group();
            container.position.set(0, 0, 0);
            this.lineGroup.add(container);
            this.alt1dContainers[index] = container;

            const frameMaterial = new LineBasicMaterial({
                color: 0xffffff,
                linewidth: 1.6,
                transparent: true,
                opacity: 0.9,
            });
            frameMaterial.depthTest = false;
            frameMaterial.depthWrite = false;
            const frame = new LineSegments(new BufferGeometry(), frameMaterial);
            frame.renderOrder = 1;
            container.add(frame);
            this.alt1dFrames[index] = frame;
            this.alt1dFrameBaseColors[index] = frameMaterial.color.clone();

            const centerMaterial = new LineDashedMaterial({
                color: 0xffffff,
                linewidth: 1,
                dashSize: 1,
                gapSize: 1,
                transparent: true,
                opacity: 0.5,
                depthWrite: false,
            });
            centerMaterial.depthTest = false;
            const centerLine = new Line(new BufferGeometry(), centerMaterial);
            centerLine.renderOrder = 0;
            container.add(centerLine);
            this.alt1dCenterLines[index] = centerLine;

            const horizontalMaterial = new LineDashedMaterial({
                color: 0xffffff,
                linewidth: 1,
                dashSize: 1,
                gapSize: 1,
                transparent: true,
                opacity: 0.5,
                depthWrite: false,
            });
            horizontalMaterial.depthTest = false;
            const horizontalLine = new Line(new BufferGeometry(), horizontalMaterial);
            horizontalLine.renderOrder = 0;
            container.add(horizontalLine);
            this.alt1dHorizontalLines[index] = horizontalLine;
        }

        const frame = this.alt1dFrames[index];
        if (frame) {
            const framePositions = new Float32Array([
                -halfWidth, halfHeight, 0,
                halfWidth, halfHeight, 0,
                halfWidth, halfHeight, 0,
                halfWidth, -halfHeight, 0,
                halfWidth, -halfHeight, 0,
                -halfWidth, -halfHeight, 0,
                -halfWidth, -halfHeight, 0,
                -halfWidth, halfHeight, 0,
            ]);
            frame.geometry.setAttribute('position', new BufferAttribute(framePositions, 3));
            frame.geometry.attributes.position.needsUpdate = true;
            frame.geometry.computeBoundingSphere();
            frame.geometry.setDrawRange(0, 8);
        }

        const centerLine = this.alt1dCenterLines[index];
        if (centerLine) {
            const centerPositions = new Float32Array([
                0, halfHeight, 0,
                0, -halfHeight, 0,
            ]);
            centerLine.geometry.setAttribute('position', new BufferAttribute(centerPositions, 3));
            centerLine.geometry.attributes.position.needsUpdate = true;
            centerLine.computeLineDistances();
            const dashSize = Math.max(0.02 * cellWidth, 0.5);
            centerLine.material.dashSize = dashSize;
            centerLine.material.gapSize = Math.max(0.04 * cellWidth, dashSize * 2);
            centerLine.material.needsUpdate = true;
        }

        const horizontalLine = this.alt1dHorizontalLines[index];
        if (horizontalLine) {
            const horizontalPositions = new Float32Array([
                -halfWidth, 0, 0,
                halfWidth, 0, 0,
            ]);
            horizontalLine.geometry.setAttribute(
                'position',
                new BufferAttribute(horizontalPositions, 3),
            );
            horizontalLine.geometry.attributes.position.needsUpdate = true;
            horizontalLine.computeLineDistances();
            const dashSize = Math.max(0.02 * cellWidth, 0.5);
            horizontalLine.material.dashSize = dashSize;
            horizontalLine.material.gapSize = Math.max(0.04 * cellWidth, dashSize * 2);
            horizontalLine.material.needsUpdate = true;
        }
    }

    _startAlt1dLineGlow(index, durationMs = this.glowDuration) {
        if (!this.alt1dMode) {
            return;
        }
        if (index == null || index < 0 || index >= this.lineObjects.length) {
            return;
        }
        const line = this.lineObjects[index];
        if (!line) {
            return;
        }
        const duration = Math.max(30, durationMs || this.glowDuration);
        const now = performance.now();
        this.alt1dGlowStates[index] = {
            startTime: now,
            duration,
        };
    }

    _renderAlt1dGlow(now) {
        const containerIntensity = new Array(this.alt1dFrames.length).fill(0);

        for (let i = 0; i < this.lineObjects.length; i += 1) {
            const line = this.lineObjects[i];
            const baseColor = this.lineBaseColors[i];
            if (!line || !line.material || !baseColor) {
                continue;
            }
            const state = this.alt1dGlowStates[i];
            const containerIdx = this.alt1dLineToContainer[i] ?? 0;

            if (!state) {
                this._restoreAlt1dLineAppearance(line, baseColor);
                continue;
            }

            const elapsed = now - state.startTime;
            if (elapsed >= state.duration) {
                this.alt1dGlowStates[i] = null;
                this._restoreAlt1dLineAppearance(line, baseColor);
                continue;
            }

            const intensity = 1 - elapsed / state.duration;
            this._applyAlt1dLineAppearance(line, baseColor, intensity);
            if (containerIdx >= 0 && containerIdx < containerIntensity.length) {
                containerIntensity[containerIdx] = Math.max(
                    containerIntensity[containerIdx],
                    intensity,
                );
            }
        }

        let maxIntensity = 0;
        let selectionDirty = false;

        for (let idx = 0; idx < this.alt1dFrames.length; idx += 1) {
            const frame = this.alt1dFrames[idx];
            const base = this.alt1dFrameBaseColors[idx];
            const intensity = containerIntensity[idx] ?? 0;
            maxIntensity = Math.max(maxIntensity, intensity);

            if (!frame || !frame.material || !base) {
                continue;
            }

            if (intensity > 0) {
                const blended = this.tempColor
                    .copy(base)
                    .lerp(this.highlightColor, Math.min(1, intensity * 0.5));
                frame.material.color.copy(blended);
                frame.material.needsUpdate = true;
                if (!this.outlineSelection.has(frame)) {
                    this.outlineSelection.add(frame);
                    selectionDirty = true;
                }
            } else {
                frame.material.color.copy(base);
                frame.material.needsUpdate = true;
                if (this.outlineSelection.delete(frame)) {
                    selectionDirty = true;
                }
            }
        }

        if (selectionDirty) {
            this._refreshOutlineSelection();
        }

        return maxIntensity;
    }

    _applyAlt1dLineAppearance(line, baseColor, intensity) {
        if (!line || !line.material || !baseColor) {
            return;
        }
        const blended = this.tempColor
            .copy(baseColor)
            .lerp(this.highlightColor, Math.min(1, intensity * 0.85));
        line.material.color.copy(blended);
        line.material.linewidth = 2.5 + (intensity * 3.5);
        line.material.opacity = 1;
        line.material.needsUpdate = true;
    }

    _restoreAlt1dLineAppearance(line, baseColor) {
        if (!line || !line.material || !baseColor) {
            return;
        }
        line.material.color.copy(baseColor);
        line.material.linewidth = 2.5;
        line.material.opacity = 1;
        line.material.needsUpdate = true;
    }

    _getDimColorHex(dimIndex) {
        const color = this.gridColors?.[dimIndex % this.gridColors.length];
        if (!color) {
            return '#ffd666';
        }
        return `#${color.getHexString()}`;
    }

    _build1DControlGroups() {
        const useAltKeys = Boolean(this.altKeysMode);
        const positiveLetters = useAltKeys
            ? ['W', 'E', 'R', 'S', 'D', 'F']
            : ['W', 'E', 'R', 'U', 'I', 'O'];
        const negativeLetters = useAltKeys
            ? ['U', 'I', 'O', 'J', 'K', 'L']
            : ['S', 'D', 'F', 'J', 'K', 'L'];

        const buildLabel = (letters, sign) => {
            const markup = letters
                .map((letter, idx) => {
                    const color = this._getDimColorHex(idx);
                    return `<span class="hud-dim-key" style="color:${color};">${letter}</span>`;
                })
                .join('<span class="hud-dim-key-sep"></span>');
            return `<span class="hud-dim-combo">${markup}<span class="hud-dim-sign">${sign}</span></span>`;
        };

        const combos = [
            { icon: null, keys: [], label: buildLabel(positiveLetters, '(+)') },
            { icon: null, keys: [], label: buildLabel(negativeLetters, '(-)') },
        ];

        const sharedControls = [
            { icon: null, keys: ['Z'], label: '<span class="speed-run-label">SPEED RUN 🔥</span>' },
            { icon: null, keys: ['Enter'], label: 'New Batch' },
            { icon: null, keys: ['SPACE'], label: 'New Dims' },
            { icon: null, keys: ['Del'], label: 'SGD Step' },
            { icon: null, keys: ['[', ']'], label: 'Step ±' },
            { icon: null, keys: [';'], label: 'Batch-size' },
            { icon: null, keys: ["'"], label: 'FP16/32' },
            { icon: null, keys: ['Y'], label: 'Top 10' },
            { icon: null, keys: ['X'], label: 'Help' },
        ];

        return [...combos, ...sharedControls];
    }

    _buildMobileHudButtons() {
        if (!this.mobileMode) {
            return [];
        }
        const dtype = (this.state?.dtype || '').replace('float', '').toUpperCase();
        const dtypeLabel = dtype ? `FP${dtype}` : 'FP';
        return MOBILE_HUD_BUTTONS.map((button) => {
            if (button.action === 'toggle-fp') {
                return { ...button, label: dtypeLabel };
            }
            return button;
        });
    }

    annotateBottomScreen(text, size = 20) {
        const bottomTextContainer = document.getElementById('bottomTextContainer');
        if (!bottomTextContainer) {
            return;
        }
        let controls;
        let hudButtons = [];
        if (this.mode === '1d') {
            controls = this._build1DControlGroups();
        } else if (this.mobileMode) {
            controls = MOBILE_CONTROL_GROUPS;
            hudButtons = this._buildMobileHudButtons();
        } else {
            controls = CONTROL_GROUPS;
        }
        const statusText = this.mobileMode ? '' : typeof text === 'string' ? text : '';
        const options = hudButtons.length ? { buttons: hudButtons } : {};
        const statusHtml = this.mobileMode ? null : this._buildStatusHtml(statusText);
        if (statusHtml) {
            options.statusHtml = statusHtml;
        }
        const displayStatus = this.mobileMode ? '' : (statusText || text || '');
        bottomTextContainer.innerHTML = formatHudMarkup(displayStatus, controls, options);
    }

    _buildStatusHtml(text) {
        if (!this.state?.speedRunActive || typeof text !== 'string' || !text.length) {
            return null;
        }
        const marker = 'SPEED RUN:';
        const idx = text.toUpperCase().indexOf(marker);
        if (idx === -1) {
            return null;
        }
        const before = escapeHtml(text.slice(0, idx));
        const highlight = escapeHtml(text.slice(idx));
        if (!highlight.length) {
            return null;
        }
        return `${before}<span class="hud-timer">${highlight}</span>`;
    }
    highlightLossLine(index, durationMs = this.glowDuration) {
        if (this.mode !== '1d') {
            return;
        }
        if (this.alt1dMode) {
            this._startAlt1dLineGlow(index, durationMs);
            return;
        }
        if (!this.lineFrames || !this.lineFrames[index]) {
            return;
        }
        if (index >= this.frameGlowState.length) {
            this.frameGlowState.length = index + 1;
        }
        const duration = Math.max(30, durationMs || this.glowDuration);
        const now = performance.now();
        this.frameGlowState[index] = {
            startTime: now,
            duration,
        };
        const frame = this.lineFrames[index];
        if (this.outlinePass && frame) {
            this.outlineSelection.add(frame);
            this._refreshOutlineSelection();
        }
    }
    // Function to handle window resizing
    onWindowResize() {
        if (this.mobileMode) {
            this.mobileCameraScalar = this._computeMobileCameraScalar();
        }

        // Calculate camera position to fit all grids
        const fovRadians = (this.camera.fov * Math.PI) / 180;
        const totalWidth = (this.numGrids + this.selectedGridScale - 1.0) * this.gridSize + (this.numGrids - 1) * this.spacing;
        let distance = (totalWidth / 2) / Math.tan(fovRadians / 2);
        if (this.customCameraDistance != null) {
            distance = this.customCameraDistance;
        }
        if (this.alt1dMode) {
            distance *= 1.25;
        }
        if (this.mobileMode) {
            distance *= this.mobileCameraScalar;
        }
        this.camera_distance = distance;
        this.camera_yOffset = this.mobileMode ? (totalWidth / 6) * this.mobileCameraScalar * 0.85 : totalWidth / 8;
        this.camera.position.set(0, this.camera_yOffset, distance);

        // Update camera aspect ratio and projection matrix
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();

        // Update renderer size
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.labelRenderer.setSize(window.innerWidth, window.innerHeight);
        this.composer?.setSize(window.innerWidth, window.innerHeight);
        this.outlinePass?.setSize(window.innerWidth, window.innerHeight);
    }

    // Function to adjust angles
    adjustAngles(angleH, angleV) {
        this.angleH += 2 * angleH;
        this.angleV += 2 * angleV;
        //this.angleV = this.normalizeDegree(this.angleV);
        this.angleH = this.normalizeDegree(this.angleH);
        this.angleV = Math.sign(this.angleV) * Math.min(Math.abs(this.angleV), this.maxAngleV);
    }

    // Function to reset angles
    resetAngle() {
        this.angleH = 0.0;
        this.angleV = this.defaultAngleV;
    }

    // Helper function to normalize angles between 0 and 360 degrees
    normalizeDegree(angle) {
        return ((angle % 360) + 360) % 360;
    }




    // Initialize all charts (including confusion matrix heatmap)
    initializeCharts() {
        this.initializeLossChart();
        this.initializeLastStepsChart();
        this.initializeStepSizeChart();
        this.initializeDimsAndStepsChart();
        this.initializeConfusionMatrixHeatmap();
        log("Charts initialized.");
    }
    // Optional: Initialize Confusion Matrix as an empty Heatmap
    initializeConfusionMatrixHeatmap() {
        const container = this.confusionMatrixContainer;
        if (!container) {
            return;
        }
        container.innerHTML = '';
        const canvas = document.createElement('canvas');
        canvas.className = 'confusion-matrix-canvas';
        container.appendChild(canvas);
        const ctx = canvas.getContext('2d');
        const dataset = {
            label: 'Confusion Matrix',
            data: [],
            borderWidth: 0.5,
            borderColor: 'rgba(255, 255, 255, 0.05)',
            backgroundColor: (context) => {
                const value = context.raw?.v ?? 0;
                return this._getHeatmapColor(value);
            },
            width: (ctxArg) => this._matrixCellWidth(ctxArg?.chart),
            height: (ctxArg) => this._matrixCellHeight(ctxArg?.chart),
        };
        const xAxisConfig = {
            type: 'category',
            labels: CONFUSION_LABELS,
            title: {
                display: true,
                text: 'Predicted Label',
                color: '#ffffff',
            },
            ticks: { color: '#ffffff' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
        };
        const yAxisConfig = {
            type: 'category',
            labels: CONFUSION_LABELS,
            reverse: true,
            title: {
                display: true,
                text: 'Actual Label',
                color: '#ffffff',
            },
            ticks: { color: '#ffffff' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
        };
        if (this.mobileMode) {
            xAxisConfig.ticks = { display: false };
            yAxisConfig.ticks = { display: false };
            xAxisConfig.title.display = false;
            yAxisConfig.title.display = false;
            xAxisConfig.grid = { display: false };
            yAxisConfig.grid = { display: false };
        }
        this.confusionMatrixChart = new Chart(ctx, {
            type: 'matrix',
            data: {
                datasets: [dataset],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (items) => {
                                const item = items[0];
                                const raw = item.raw || {};
                                return `Predicted: ${raw.x} • Actual: ${raw.y}`;
                            },
                            label: (item) => {
                                const value = item.raw?.v ?? 0;
                                return `Count: ${value}`;
                            },
                        },
                    },
                },
                layout: {
                    padding: {
                        top: 8,
                        left: 12,
                        right: 12,
                        bottom: 32,
                    },
                },
                scales: {
                    x: xAxisConfig,
                    y: yAxisConfig,
                },
            },
        });
    }

    initializeLossChart() {
        if (document.getElementById('lossChart')) {
            const lineWidth = this.mobileMode ? 1.2 : 2.4;
            this.lossChart = new Chart(document.getElementById('lossChart'), {
                type: 'line',
                data: {
                    labels: [], // Step numbers
                    datasets: [
                        { label: 'Train', data: [], borderColor: 'blue', fill: false },
                        { label: 'Val', data: [], borderColor: 'red', fill: false },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top', // Ensure legend is at the top of the chart area
                            labels: {
                                boxWidth: 2,
                                padding: 10,
                            },
                        },
                    },
                    layout: {
                        padding: {
                            top: 0, // Add padding to avoid overlap between legend and plot data
                        },
                    },
                    elements: {
                        line: {
                            borderWidth: lineWidth,
                        },
                        point: {
                            radius: 0,
                            hoverRadius: 0,
                            hitRadius: 4,
                        },
                    },
                    scales: {
                        x: {
                            title: {
                                display: !this.mobileMode,
                                text: 'Step',
                            },
                            ticks: {
                                display: !this.mobileMode,
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Loss',
                            },
                        },
                    },
                },
            });
        }
    }

    // Initialize Last Steps Chart
    initializeLastStepsChart() {
        if (!this.mobileMode && document.getElementById('lastStepsChart')) {
            const lineWidth = this.mobileMode ? 1.2 : 2.4;
            this.lastStepsChart = new Chart(document.getElementById('lastStepsChart'), {
                type: 'line',
                data: {
                    labels: [], // Recent steps
                    datasets: [{ label: 'Train (recent)', data: [], borderColor: 'green', fill: false }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top', // Ensure legend is at the top of the chart area
                            labels: {
                                boxWidth: 2,
                                padding: 10,
                            },
                        },
                    },
                    layout: {
                        padding: {
                            top: 0, // Add padding to avoid overlap between legend and plot data
                        },
                    },
                    elements: {
                        line: {
                            borderWidth: lineWidth,
                        },
                        point: {
                            radius: 0,
                            hoverRadius: 0,
                            hitRadius: 4,
                        },
                    },
                    scales: {
                        x: {
                            title: {
                                display: !this.mobileMode,
                                text: 'Step',
                            },
                            ticks: {
                                display: !this.mobileMode,
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Loss',
                            },
                        },
                    },
                },
            });
        }
    }


    // Initialize Step Size Chart
    initializeStepSizeChart() {
        if (document.getElementById('stepSizeChart')) {
            this.stepSizeChart = new Chart(document.getElementById('stepSizeChart'), {
                type: 'bar',
                data: {
                    labels: ['Step Size'],
                    datasets: [{ label: 'Log Step Size', data: [], backgroundColor: 'orange' }],
                },
                options: { responsive: true, indexAxis: 'y' },
            });
        }
    }

    // Initialize Dims and Steps Chart
    initializeDimsAndStepsChart() {
        if (document.getElementById('dimsAndStepsChart')) {
            this.dimsAndStepsChart = new Chart(document.getElementById('dimsAndStepsChart'), {
                type: 'bar',
                data: {
                    labels: [], // Dimensions
                    datasets: [{ label: 'Cumulative Step', data: [], backgroundColor: 'teal' }],
                },
                options: { responsive: true, scales: { x: { title: { display: true, text: 'Dimension' } } } },
            });
        }
    }


    // Update Loss Chart
    updateLossChart(trainLoss, valLoss, steps) {
        //var lastLoss = trainLoss[trainLoss.length-1];
        //this.state.updateBestScoreOrNot(lastLoss);
        this.annotateBottomScreen(this.state.toString());
        if (this.lossChart) {
            this.lossChart.data.labels = steps;
            this.lossChart.data.datasets[0].data = trainLoss;
            this.lossChart.data.datasets[1].data = valLoss;
            this.lossChart.update();
        }
    }

    // Update Last Steps Chart
    updateLastStepsChart(lastSteps, losses) {
        if (!this.mobileMode && this.lastStepsChart) {
            this.lastStepsChart.data.labels = lastSteps;
            this.lastStepsChart.data.datasets[0].data = losses;
            this.lastStepsChart.update();
        }
    }

    // Update Step Size Chart
    updateStepSizeChart(logStepSize) {
        if (this.stepSizeChart) {
            this.stepSizeChart.data.datasets[0].data = [logStepSize];
            this.stepSizeChart.update();
        }
    }

    // Update Dims and Steps Chart
    updateDimsAndStepsChart(dimLabels, steps) {
        if (this.dimsAndStepsChart) {
            this.dimsAndStepsChart.data.labels = dimLabels;
            this.dimsAndStepsChart.data.datasets[0].data = steps;
            this.dimsAndStepsChart.update();
        }
    }
    // Update Confusion Matrix Heatmap with new data
    updateConfusionMatrix(confusionMatrix) {
        if (!this.confusionMatrixChart) {
            this.initializeConfusionMatrixHeatmap();
        }
        const chart = this.confusionMatrixChart;
        if (!chart) {
            return;
        }
        try {
            if (!Array.isArray(confusionMatrix) || confusionMatrix.length === 0) {
                throw new Error("Confusion matrix data must be a non-empty array.");
            }
            const numRows = confusionMatrix.length;
            const numCols = Array.isArray(confusionMatrix[0]) ? confusionMatrix[0].length : 0;
            if (
                numCols === 0 ||
                !confusionMatrix.every((row) => Array.isArray(row) && row.length === numCols)
            ) {
                throw new Error("Confusion matrix data is empty or improperly formatted.");
            }
            const labels = Array.from({ length: numCols }, (_, idx) => idx.toString());
            const flatValues = confusionMatrix.flat();
            const maxVal = Math.max(1, ...flatValues);
            this.confusionMatrixMaxValue = maxVal;
            this.confusionRows = numRows;
            this.confusionCols = numCols;
            const data = [];
            for (let actual = 0; actual < numRows; actual += 1) {
                for (let predicted = 0; predicted < numCols; predicted += 1) {
                    data.push({
                        x: labels[predicted],
                        y: labels[actual],
                        v: confusionMatrix[actual][predicted],
                    });
                }
            }
            chart.data.datasets[0].data = data;
            chart.options.scales.x.labels = labels;
            chart.options.scales.y.labels = labels;
            chart.update('none');
        } catch (error) {
            console.error("Failed to update confusion matrix heatmap:", error);
        }
    }
    _matrixCellWidth(chart) {
        if (!chart || !chart.chartArea) {
            return 18;
        }
        const { width } = chart.chartArea;
        const cols = Math.max(1, this.confusionCols || CONFUSION_LABELS.length);
        const gap = Math.max(0, cols - 1) * 0.4;
        return Math.max(6, (width - gap) / cols);
    }

    _matrixCellHeight(chart) {
        if (!chart || !chart.chartArea) {
            return 18;
        }
        const { height } = chart.chartArea;
        const rows = Math.max(1, this.confusionRows || CONFUSION_LABELS.length);
        const gap = Math.max(0, rows - 1) * 0.4;
        return Math.max(6, (height - gap) / rows);
    }

    _getHeatmapColor(value) {
        const ratio = value / Math.max(1, this.confusionMatrixMaxValue);
        return interpolateHeatmapColor(ratio);
    }

    _createExampleChart(ctx, labels) {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Probability',
                        data: [],
                        backgroundColor: 'rgba(255, 165, 0, 0.8)',
                        borderWidth: 0,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Prob: ${(ctx.raw ?? 0).toFixed(2)}`,
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: { color: '#ffffff', font: { size: 9 } },
                        grid: { display: false },
                        title: { display: true, text: 'Class', color: '#ffffff', font: { size: 10 } },
                    },
                    y: {
                        beginAtZero: true,
                        suggestedMax: 1.2,
                        ticks: { color: '#ffffff', maxTicksLimit: 4 },
                        grid: { color: 'rgba(255, 255, 255, 0.08)' },
                        title: { display: true, text: 'Probability', color: '#ffffff', font: { size: 10 } },
                    },
                },
            },
        });
    }


    updateMeshGrids(meshGrids = null) {
        if (this.mode === '1d') {
            return;
        }
        if (this.scene == null) {
            return;
        }
        if (this.lineGroup) {
            this.lineContainers.forEach((container) => {
                container.children.forEach((child) => child.geometry?.dispose?.());
                this.lineGroup.remove(container);
            });
            this.scene.remove(this.lineGroup);
            this.lineGroup = null;
            this.lineObjects = [];
            this.lineFrames = [];
            this.lineContainers = [];
            this.centerLines = [];
            this.horizontalLines = [];
            this.lineScaleCache = [];
            this.lineBaseColors = [];
            this._resetAllFrameGlow();
        }
        // Use previous meshGrids if no parameter is provided
        if (meshGrids === null && this.previousMeshGrids) {
            meshGrids = this.previousMeshGrids;
        } else {
            this.previousMeshGrids = meshGrids; // Save current meshGrids for reuse
        }

        if (!meshGrids) {
            return;
        }

        // Remove old grids and spheres if re-initializing
        if (this.gridObjects.length > 0) {
            this.gridObjects.forEach(grid => {
                this.scene.remove(grid);
            });
            this.gridObjects = [];
        }

        if (this.sphereObjects && this.sphereObjects.length > 0) {
            this.sphereObjects.forEach(sphere => {
                this.scene.remove(sphere);
            });
            this.sphereObjects = [];
        } else {
            this.sphereObjects = [];
        }

        // Subtract center value from each grid
        const originValue = meshGrids[0][Math.floor(this.gridSize / 2)][Math.floor(this.gridSize / 2)];
        for (let i = 0; i < meshGrids.length; i++) {
            for (let j = 0; j < meshGrids[i].length; j++) {
                for (let k = 0; k < meshGrids[i][j].length; k++) {
                    meshGrids[i][j][k] -= originValue;
                }
            }
        }

        // Find the maximum absolute value for scaling
        let maxAbsValue = 0;
        meshGrids.forEach(grid => {
            grid.forEach(row => {
                row.forEach(value => {
                    maxAbsValue = Math.max(maxAbsValue, Math.abs(value));
                });
            });
        });

        const eps = 1e-3;
        const scale = 1.5 * this.gridSize / (maxAbsValue + eps);

        // Scale meshGrids
        meshGrids = meshGrids.map(grid =>
            grid.map(row =>
                row.map(value => value * scale)
            )
        );

        // Calculate total width of all grids to properly center them
        const numGrids = meshGrids.length;
        const totalWidth = (numGrids + this.selectedGridScale - 1.0) * this.gridSize + (numGrids - 1) * this.spacing;

        // Create or update each grid and sphere
        for (let i = 0; i < numGrids; i++) {
            // Create new geometry for each grid
            const geometry = new PlaneGeometry(this.gridSize, this.gridSize, this.gridSize - 1, this.gridSize - 1);
            const color = this.gridColors[i % this.gridColors.length];
            let baseOpacity = 1.0;//this.alpha;
            let secondaryOpacity = baseOpacity * 0.3;

            // Apply lower opacity for non-selected grids
            if (i !== this.selectedGridIndex) {
                secondaryOpacity = 0.05;
                baseOpacity *= 0.4;
            } else {
                secondaryOpacity *= 0.8; // Make non-selected grids more transparent
            }

            // Update the geometry of the mesh to reflect the new heights and apply transparency for Z >= 0
            const positions = geometry.attributes.position.array;
            const alphas = new Float32Array(positions.length / 3); // Create an array to store alpha values for each vertex

            for (let j = 0; j < this.gridSize; j++) {
                for (let k = 0; k < this.gridSize; k++) {
                    const index = 3 * (j * this.gridSize + k);
                    const zValue = meshGrids[i][j][k];
                    positions[index + 2] = zValue; // Update Z value (height) of the grid

                    // Set alpha value lower for points where Z >= 0
                    alphas[j * this.gridSize + k] = zValue >= 0 ? secondaryOpacity : baseOpacity;
                }
            }

            geometry.setAttribute('alpha', new BufferAttribute(alphas, 1)); // Add alpha as a vertex attribute
            geometry.attributes.position.needsUpdate = true;

            // Use ShaderMaterial to apply alpha for each vertex
            const material = new ShaderMaterial({
                uniforms: {
                    color: { value: color },
                },
                vertexShader: `
                    attribute float alpha;
                    varying float vAlpha;

                    void main() {
                        vAlpha = alpha;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform vec3 color;
                    varying float vAlpha;

                    void main() {
                        gl_FragColor = vec4(color, vAlpha);
                    }
                `,
                transparent: true,
                wireframe: true,
            });

            const mesh = new Mesh(geometry, material);
            mesh.rotation.x = -Math.PI / 2; // Rotate to lay flat

            // Calculate position to center grids
            let xOffset = -totalWidth / 2 + i * (this.gridSize + this.spacing) + this.gridSize / 2;

            // Apply scaling if it's the selected grid
            if (i > this.selectedGridIndex) {
                xOffset += this.gridSize * (this.selectedGridScale - 1); // Adjust position for scaled grid
            }
            if (i === this.selectedGridIndex) {
                xOffset += this.gridSize * (this.selectedGridScale - 1) / 2
                mesh.scale.set(this.selectedGridScale, this.selectedGridScale, 1); // Scale selected grid on X, Z axes
            } else {
                mesh.scale.set(1, 1, 1); // Reset scale for non-selected grids
            }

            mesh.position.set(xOffset, 0, 0);
            mesh.userData.gridIndex = i;

            // Add new grid to scene and store it for later reference
            this.scene.add(mesh);
            this.gridObjects.push(mesh);

            // Create a red sphere to represent the center point of each grid
            const sphereGeometry = new SphereGeometry(1, 16, 16); // Radius = 1, segments for smoother look
            const sphereMaterial = new MeshBasicMaterial({ color: 0xff0000 }); // Red color
            const sphere = new Mesh(sphereGeometry, sphereMaterial);

            // Position the sphere at the center of the current grid
            sphere.position.set(xOffset, 0, 0); // Set it on the center of the grid

            // Adjust Y position (height) to match the central height of the grid
            const centerHeight = meshGrids[i][Math.floor(this.gridSize / 2)][Math.floor(this.gridSize / 2)];
            sphere.position.y = centerHeight;

            // Add the sphere to the scene and store it for later reference
            this.scene.add(sphere);
            this.sphereObjects.push(sphere);
        }
    }




    // Function to get current horizontal and vertical angles
    getAngles() {
        return { angleH: this.angleH, angleV: this.angleV };
    }

    getCanvasElement() {
        return this.renderer?.domElement ?? null;
    }

    // Function to get the selected grid index
    getSelectedGrid() {
        return this.selectedGridIndex;
    }

    // Function to increase zoom level
    increaseZoom() {
        this.scaleFactor = Math.max(0.7, this.scaleFactor - 0.05);
    }

    // Function to decrease zoom level
    decreaseZoom() {
        this.scaleFactor = Math.min(5.0, this.scaleFactor + 0.05);
    }

    // Function to increment selected grid
    incrementSelectedGrid() {
        this.selectedGridIndex = (this.selectedGridIndex + 1) % this.effectiveGrids;
    }

    // Function to decrement selected grid
    decrementSelectedGrid() {
        this.selectedGridIndex = (this.selectedGridIndex - 1 + this.effectiveGrids) % this.effectiveGrids; // Adding effectiveGrids to ensure non-negative index
    }

    // Function to set the selected grid
    setSelectedGrid(gridIdx) {
        this.selectedGridIndex = gridIdx;
    }

    selectGridAt(clientX, clientY) {
        const canvas = this.getCanvasElement();
        if (!canvas || !this.camera || !this.raycaster || !this.pointer) {
            return null;
        }

        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) {
            return null;
        }

        const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
        const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;

        this.pointer.set(ndcX, ndcY);
        this.raycaster.setFromCamera(this.pointer, this.camera);

        const intersections = this.raycaster.intersectObjects(this.gridObjects, false);
        if (!intersections.length) {
            debugSelection('[View] selectGridAt: no intersection');
            return null;
        }

        const target = intersections[0].object;
        const gridIndex =
            typeof target.userData?.gridIndex === 'number'
                ? target.userData.gridIndex
                : this.gridObjects.indexOf(target);

        if (gridIndex < 0) {
            debugSelection('[View] selectGridAt: intersection missing grid index');
            return null;
        }

        if (gridIndex !== this.selectedGridIndex) {
            debugSelection(`[View] selectGridAt: selecting grid ${gridIndex}`);
            this.setSelectedGrid(gridIndex);
            this.updateMeshGrids();
        } else {
            debugSelection(`[View] selectGridAt: grid ${gridIndex} already selected`);
        }

        return gridIndex;
    }

    updateLossLines(lines, { stepSpacing, labels } = {}) {
        if (this.mode !== '1d' || !this.scene) {
            return;
        }
        if (!Array.isArray(lines) || !lines.length) {
            return;
        }

        if (this.alt1dMode) {
            this._updateAlt1dLossLines(lines, { stepSpacing, labels });
            return;
        }

        if (!this.lineGroup) {
            this.lineGroup = new Group();
            this.scene.add(this.lineGroup);
        }

        const count = lines.length;
        this.effectiveGrids = count || 1;
        const columns = Math.max(1, Math.min(3, count));
        const rows = Math.max(1, Math.ceil(count / columns));
        const cellWidth = this.gridSize * 0.9;
        const cellHeight = this.gridSize * 0.6;
        const gapX = this.gridSize * 0.05;
        const gapY = this.rowSpacing != null ? this.rowSpacing : this.gridSize * 0.3;
        const depthStep = this.depthStep != null ? this.depthStep : 0.01;

        while (this.lineObjects.length < count) {
            const material = new LineBasicMaterial({ color: 0xffffff, linewidth: 2.5 });
            const geometry = new BufferGeometry();
            const line = new Line(geometry, material);

            const frameMaterial = new LineBasicMaterial({ color: 0xffffff, linewidth: 1.5 });
            const frameGeometry = new BufferGeometry();
            const frame = new LineSegments(frameGeometry, frameMaterial);
            frameMaterial.transparent = true;
            frameMaterial.opacity = 0.9;
            frame.scale.set(1, 1, 1);

            const centerMaterial = new LineDashedMaterial({
                color: 0xffffff,
                linewidth: 1,
                dashSize: 1,
                gapSize: 1,
                transparent: true,
                opacity: 0.5,
                depthWrite: false,
            });
            centerMaterial.depthTest = false;
            const centerGeometry = new BufferGeometry();
            const centerLine = new Line(centerGeometry, centerMaterial);

            const hMaterial = new LineDashedMaterial({
                color: 0xffffff,
                linewidth: 1,
                dashSize: 1,
                gapSize: 1,
                transparent: true,
                opacity: 0.5,
                depthWrite: false,
            });
            hMaterial.depthTest = false;
            const hGeometry = new BufferGeometry();
            const hLine = new Line(hGeometry, hMaterial);

            const container = new Group();
            container.add(frame);
            container.add(centerLine);
            container.add(hLine);
            container.add(line);
            this.lineGroup.add(container);

            frame.renderOrder = 1;
            centerLine.renderOrder = 0;
            hLine.renderOrder = 0;
            line.renderOrder = 2;

            this.lineContainers.push(container);
            this.lineFrames.push(frame);
            this.centerLines.push(centerLine);
            this.horizontalLines.push(hLine);
            this.lineObjects.push(line);
        }

        while (this.lineObjects.length > count) {
            const line = this.lineObjects.pop();
            const idx = this.lineObjects.length;
            const frame = this.lineFrames.pop();
            const centerLine = this.centerLines.pop();
            const hLine = this.horizontalLines.pop();
            this.lineBaseColors.pop();
            if (frame) {
                frame.scale.set(1, 1, 1);
                this.outlineSelection.delete(frame);
            }
            if (this.frameGlowState[idx]) {
                this.frameGlowState[idx] = null;
            }
            const container = this.lineContainers.pop();
            if (container) {
                container.remove(line);
                if (frame) {
                    container.remove(frame);
                }
                if (centerLine) {
                    container.remove(centerLine);
                }
                if (hLine) {
                    container.remove(hLine);
                }
                this.lineGroup.remove(container);
            }
            frame?.geometry?.dispose?.();
            centerLine?.geometry?.dispose?.();
            centerLine?.material?.dispose?.();
            hLine?.geometry?.dispose?.();
            hLine?.material?.dispose?.();
            line.geometry.dispose?.();
        }

        this._refreshOutlineSelection();
        this.frameGlowState.length = count;
        this.lineScaleCache.length = count;
        const halfWidth = cellWidth / 2;
        const halfHeight = cellHeight / 2;
        const lerpAlpha = 0.35;
        const eps = 1e-6;

        for (let i = 0; i < count; i += 1) {
            const data = lines[i];
            const line = this.lineObjects[i];
            const frame = this.lineFrames[i];
            const container = this.lineContainers[i];
            const hLine = this.horizontalLines[i];
            if (!Array.isArray(data) || !line || !container || !frame) {
                continue;
            }
            if (this.frameGlowState[i] === undefined) {
                this.frameGlowState[i] = null;
            }

            const length = data.length;
            if (!length) {
                continue;
            }

            const mid = (length - 1) / 2;
            const baseline = data[Math.max(0, Math.floor(mid))];
            let maxAbs = 1e-6;
            for (let j = 0; j < length; j += 1) {
                maxAbs = Math.max(maxAbs, Math.abs(data[j] - baseline));
            }
            const preMaxAbs = maxAbs;

            const scaleX = length > 1 ? cellWidth / (length - 1) : cellWidth;
            let scaleY = 0;
            if (maxAbs > eps) {
                const targetScale = halfHeight / maxAbs;
                const previous = this.lineScaleCache[i] ?? targetScale;
                const lerped = MathUtils.lerp(previous, targetScale, lerpAlpha);
                scaleY = Math.min(lerped, targetScale);
                this.lineScaleCache[i] = scaleY;
            } else {
                this.lineScaleCache[i] = 0;
            }
            let postMaxAbs = 0;

            let geometry = line.geometry;
            if (!(geometry instanceof BufferGeometry)) {
                geometry = new BufferGeometry();
                line.geometry = geometry;
            }

            let positionAttr = geometry.getAttribute('position');
            if (!positionAttr || positionAttr.array.length !== length * 3) {
                const positions = new Float32Array(length * 3);
                positionAttr = new BufferAttribute(positions, 3);
                geometry.setAttribute('position', positionAttr);
            }

            const positions = positionAttr.array;
            for (let j = 0; j < length; j += 1) {
                const idx = j * 3;
                positions[idx] = (j - mid) * scaleX;
                const normalizedY = scaleY > 0 ? (data[j] - baseline) * scaleY : 0;
                positions[idx + 1] = normalizedY;
                positions[idx + 2] = 0;
                postMaxAbs = Math.max(postMaxAbs, Math.abs(normalizedY));
            }

            positionAttr.needsUpdate = true;
            geometry.computeBoundingSphere();
            geometry.setDrawRange(0, length);

            const color = this.gridColors[i % this.gridColors.length];
            const baseColor = color.clone();
            this.lineBaseColors[i] = baseColor;
            line.material.color.copy(color);
            line.material.linewidth = 2.5;
            line.material.needsUpdate = true;
            line.material.depthTest = false;
            line.material.depthWrite = false;

            frame.material.color.copy(color);
            frame.material.linewidth = 1.5;
            frame.material.needsUpdate = true;
            frame.material.depthTest = false;
            frame.material.depthWrite = false;
            frame.scale.set(1, 1, 1);

            const framePositions = new Float32Array([
                -halfWidth, halfHeight, 0,
                halfWidth, halfHeight, 0,
                halfWidth, halfHeight, 0,
                halfWidth, -halfHeight, 0,
                halfWidth, -halfHeight, 0,
                -halfWidth, -halfHeight, 0,
                -halfWidth, -halfHeight, 0,
                -halfWidth, halfHeight, 0,
            ]);
            frame.geometry.setAttribute('position', new BufferAttribute(framePositions, 3));
            frame.geometry.attributes.position.needsUpdate = true;
            frame.geometry.computeBoundingSphere();
            frame.geometry.setDrawRange(0, 8);

            const centerLine = this.centerLines[i];
            const centerPositions = new Float32Array([
                0, halfHeight, -0.01,
                0, -halfHeight, -0.01,
            ]);
            centerLine.geometry.setAttribute('position', new BufferAttribute(centerPositions, 3));
            centerLine.geometry.attributes.position.needsUpdate = true;
            centerLine.material.dashSize = Math.max(0.02 * cellHeight, 0.5);
            centerLine.material.gapSize = Math.max(0.04 * cellHeight, 0.8);
            centerLine.material.needsUpdate = true;
            centerLine.computeLineDistances();
            centerLine.material.opacity = 0.8;

            if (hLine) {
                const hPositions = new Float32Array([
                    -halfWidth, 0, -0.01,
                    halfWidth, 0, -0.01,
                ]);
                hLine.geometry.setAttribute('position', new BufferAttribute(hPositions, 3));
                hLine.geometry.attributes.position.needsUpdate = true;
                hLine.material.dashSize = Math.max(0.02 * cellWidth, 0.5);
                hLine.material.gapSize = Math.max(0.04 * cellWidth, 0.8);
                hLine.material.needsUpdate = true;
                hLine.computeLineDistances();
                hLine.material.opacity = 0.6;
                hLine.material.depthTest = false;
                hLine.material.depthWrite = false;
            }

            const col = i % columns;
            const row = Math.floor(i / columns);
            const xOffset = (col - (columns - 1) / 2) * (cellWidth + gapX);
            const yOffset = rows > 1 ? -(row - (rows - 1) / 2) * (cellHeight + gapY) : 0;
            const zOffset = -(row - (rows - 1) / 2) * depthStep;
            container.position.set(xOffset, yOffset, zOffset);

            if (this.debug) {
                const info = {
                    line: i,
                    preMaxAbs,
                    postMaxAbs,
                    halfHeight,
                    scaleY,
                    stepSize: this.state?.stepSize,
                };
                log(`[LandscapeView] loss-line bounds ${JSON.stringify(info)}`);
            }
        }
    }
    render() {
        requestAnimationFrame(() => this.render());

        const now = performance.now();
        let maxIntensity = 0;

        // Rotate each grid according to the current angles
        this.gridObjects.forEach((grid) => {
            grid.rotation.z = MathUtils.degToRad(this.angleH);
            grid.rotation.x = MathUtils.degToRad(this.angleV - 90);
        });

        if (this.lineGroup) {
            if (this.mode === '1d') {
                this.lineGroup.rotation.set(0, 0, 0);
            } else {
                this.lineGroup.rotation.y = MathUtils.degToRad(this.angleH);
                this.lineGroup.rotation.x = MathUtils.degToRad(this.angleV - 90);
            }
        }

        if (this.alt1dMode) {
            maxIntensity = this._renderAlt1dGlow(now);
        } else {
            for (let i = 0; i < this.frameGlowState.length; i += 1) {
                const state = this.frameGlowState[i];
                const frame = this.lineFrames[i];
                if (!state || !frame) {
                    continue;
                }
                const elapsed = now - state.startTime;
                if (elapsed >= state.duration) {
                    this.frameGlowState[i] = null;
                    frame.scale.set(1, 1, 1);
                    this._restoreFrameMaterial(i);
                    if (this.outlineSelection.delete(frame)) {
                        this._refreshOutlineSelection();
                    }
                    continue;
                }
                const intensity = 1 - elapsed / state.duration;
                const expandScale = 1 + this.glowExpand * intensity;
                frame.scale.set(expandScale, expandScale, 1);
                this._applyFrameGlowAppearance(i, intensity);
                maxIntensity = Math.max(maxIntensity, intensity);
            }
        }

        if (this.outlinePass) {
            if (maxIntensity > 0) {
                this.outlinePass.edgeStrength = this.glowEdgeStrength * maxIntensity;
                this.outlinePass.edgeThickness = 1 + (this.glowExpand * 10 * maxIntensity);
            } else {
                this.outlinePass.edgeStrength = 0;
            }
        }

        if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
        this.labelRenderer?.render(this.scene, this.camera);
    }

    _applyFrameGlowAppearance(index, intensity) {
        const frame = this.lineFrames?.[index];
        const baseColor = this.lineBaseColors?.[index];
        if (!frame || !frame.material || !baseColor) {
            return;
        }
        const blended = this.tempColor.copy(baseColor).lerp(this.highlightColor, Math.min(1, intensity * 0.6));
        frame.material.color.copy(blended);
        frame.material.opacity = 0.9 + 0.3 * intensity;
        frame.material.needsUpdate = true;
    }

    _restoreFrameMaterial(index) {
        const frame = this.lineFrames?.[index];
        const baseColor = this.lineBaseColors?.[index];
        if (!frame || !frame.material || !baseColor) {
            return;
        }
        frame.material.color.copy(baseColor);
        frame.material.opacity = 0.9;
        frame.material.needsUpdate = true;
    }

    _refreshOutlineSelection() {
        if (this.outlinePass) {
            this.outlinePass.selectedObjects = Array.from(this.outlineSelection);
        }
    }

    _resetAllFrameGlow() {
        if (this.alt1dMode) {
            for (let i = 0; i < this.alt1dGlowStates.length; i += 1) {
                this.alt1dGlowStates[i] = null;
            }
            for (let i = 0; i < this.lineObjects.length; i += 1) {
                const line = this.lineObjects[i];
                const baseColor = this.lineBaseColors[i];
                if (line && baseColor) {
                    this._restoreAlt1dLineAppearance(line, baseColor);
                }
            }
            for (let idx = 0; idx < this.alt1dFrames.length; idx += 1) {
                const frame = this.alt1dFrames[idx];
                const base = this.alt1dFrameBaseColors[idx];
                if (frame && frame.material && base) {
                    frame.material.color.copy(base);
                    frame.material.needsUpdate = true;
                }
                this.outlineSelection.delete(frame);
            }
        }
        if (this.lineFrames) {
            for (let i = 0; i < this.lineFrames.length; i += 1) {
                const frame = this.lineFrames[i];
                if (frame) {
                    frame.scale.set(1, 1, 1);
                }
                this._restoreFrameMaterial(i);
                this.frameGlowState[i] = null;
            }
        }
        this.outlineSelection.clear();
        this._refreshOutlineSelection();
    }
    // Update Example Images
    updateExamples(images) {
        log('Received images for update');

        // Select side container and get the individual example cells
        const sideContainer = document.getElementById('sideContainer');
        if (!sideContainer) {
            return;
        }
        const exampleCells = sideContainer.getElementsByClassName('example-cell');

        Array.from(exampleCells).forEach((exampleCell, index) => {
            const previousImage = exampleCell.querySelector('.example-image');
            if (previousImage && previousImage.parentElement === exampleCell) {
                exampleCell.removeChild(previousImage);
            }

            // Create a new canvas element for the image
            const canvas = document.createElement('canvas');
            canvas.width = images[index][0].length; // Assuming the data is [height][width]
            canvas.height = images[index].length;

            const ctx = canvas.getContext('2d');
            const imageDataObject = ctx.createImageData(canvas.width, canvas.height);

            // Assuming the data is grayscale (0-1), set pixels accordingly
            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    const pixelIndex = (y * canvas.width + x) * 4;
                    const pixelValue = images[index][y][x];

                    imageDataObject.data[pixelIndex] = pixelValue * 255; // Red
                    imageDataObject.data[pixelIndex + 1] = pixelValue * 255; // Green
                    imageDataObject.data[pixelIndex + 2] = pixelValue * 255; // Blue
                    imageDataObject.data[pixelIndex + 3] = 255; // Alpha (fully opaque)
                }
            }

            // Put the image data on the canvas
            ctx.putImageData(imageDataObject, 0, 0);
            canvas.classList.add('example-image');
            canvas.style.backgroundColor = 'transparent'; // Transparent background

            // Append the canvas to the corresponding example cell
            exampleCell.appendChild(canvas);
        });
    }
    // Update Example Predictions (Bar Charts)
    updateExamplePreds(predictions) {
        //log('Received predictions for update');

        // Select side container and get the individual example cells
        const sideContainer = document.getElementById('sideContainer');
        if (!sideContainer) {
            return;
        }
        const exampleCells = sideContainer.getElementsByClassName('example-cell');

        predictions.forEach((prediction, index) => {
            // Get or create chart container for the corresponding example cell
            if (exampleCells.length <= index) {
                return;
            }
            let chartDiv = exampleCells[index].querySelector('.example-chart');

            if (!chartDiv) {
                // If the chartDiv doesn't exist, create it
                chartDiv = document.createElement('div');
                chartDiv.id = `chartDiv${index + 1}`;
                chartDiv.classList.add('example-chart');
                exampleCells[index].appendChild(chartDiv);
            }
            let canvas = chartDiv.querySelector('canvas');
            if (!canvas) {
                canvas = document.createElement('canvas');
                canvas.className = 'example-chart-canvas';
                chartDiv.appendChild(canvas);
            }
            const ctx = canvas.getContext('2d');
            const labels = Array.from({ length: prediction.length }, (_, i) => i.toString());
            let chart = this.exampleCharts[index];
            if (!chart) {
                chart = this._createExampleChart(ctx, labels);
                this.exampleCharts[index] = chart;
            } else {
                chart.data.labels = labels;
            }
            chart.data.datasets[0].data = prediction;
            chart.update('none');
        });
    }

    _setBodyHelpOpen(isOpen) {
        if (typeof document === 'undefined' || !document.body) {
            return;
        }
        document.body.classList.toggle('help-overlay-open', Boolean(isOpen));
    }

    showImage(screenId) {
        const container = document.getElementById('imageContainer');
        if (!container) {
            return;
        }
        this.manualKeyOverlay = false;
        const isTourScreen = TOUR_SCREENS.includes(screenId);
        this._ensureHelpOverlayElements();
        if (!this.helpOverlay || !this.helpOverlayContent) {
            return;
        }
        container.classList.add('help-open');
        this._setBodyHelpOpen(true);
        this.helpOverlay.classList.add('visible');
        this.helpOverlay.dataset.mode = isTourScreen ? 'tour' : 'media';
        this.helpContentMode = isTourScreen ? 'tour' : 'media';
        if (isTourScreen) {
            this._renderTourScreen(screenId);
            this._attachHelpKeyListener();
            return;
        }
        this.helpOverlayContent.innerHTML = '';
        const img = document.createElement('img');
        img.className = 'help-overlay__image';
        img.alt = 'Help screen';
        img.src = screenId;
        this.helpOverlayContent.appendChild(img);
        this._attachHelpKeyListener();
    }

    hideImage() {
        const container = document.getElementById('imageContainer');
        if (!container) {
            return;
        }
        container.classList.remove('help-open');
        this._setBodyHelpOpen(false);
        this._detachHelpKeyListener();
        this.manualKeyOverlay = false;
        if (this.helpOverlay) {
            this.helpOverlay.classList.remove('visible');
            this.helpOverlay.removeAttribute('data-tour-screen');
            this.helpOverlay.removeAttribute('data-mode');
        }
    }

    _ensureHelpOverlayElements() {
        if (this.helpOverlay && this.helpOverlayContent) {
            return;
        }
        const container = document.getElementById('imageContainer');
        if (!container) {
            return;
        }
        const overlay = document.createElement('div');
        overlay.className = 'help-overlay';
        overlay.setAttribute('role', 'dialog');
        overlay.setAttribute('aria-label', 'Guide');

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'help-overlay__content';
        contentWrapper.addEventListener('click', (event) => event.stopPropagation());

        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'help-overlay__close';
        closeBtn.setAttribute('aria-label', 'Close help screens');
        closeBtn.innerHTML = '&times;';
        closeBtn.addEventListener('click', (event) => {
            event?.preventDefault?.();
            event?.stopPropagation?.();
            if (this.helpContentMode === 'tour' && this.state) {
                if (typeof this.state.closeHelpScreens === 'function') {
                    this.state.closeHelpScreens();
                } else {
                    this.state.helpScreenIdx = -1;
                }
            }
            this.hideImage();
        });

        const tabs = document.createElement('div');
        tabs.className = 'help-overlay__tabs';
        const controlsTab = document.createElement('button');
        controlsTab.type = 'button';
        controlsTab.dataset.helpTab = 'controls';
        controlsTab.textContent = 'Controls';
        controlsTab.addEventListener('click', () => this._showHelpTabs('controls'));
        const howTab = document.createElement('button');
        howTab.type = 'button';
        howTab.dataset.helpTab = 'how';
        howTab.textContent = 'How it works';
        howTab.addEventListener('click', () => this._showHelpTabs('how'));
        tabs.appendChild(controlsTab);
        tabs.appendChild(howTab);

        const contentBody = document.createElement('div');
        contentBody.className = 'help-overlay__body';

        contentWrapper.appendChild(closeBtn);
        contentWrapper.appendChild(tabs);
        contentWrapper.appendChild(contentBody);
        overlay.appendChild(contentWrapper);

        overlay.addEventListener('click', (event) => {
            const isMobile = this.mobileMode;
            const clickedOverlay = event.target === overlay;
            if (clickedOverlay && this.helpContentMode === 'tour') {
                this._advanceHelpScreen();
                return;
            }
            if (isMobile && event.target.closest && !event.target.closest('.tour-btn')) {
                this.hideImage();
            }
        });

        container.appendChild(overlay);
        this.helpOverlay = overlay;
        this.helpOverlayContent = contentBody;
        this.helpTabsBar = tabs;
        this.helpTabs = { controls: controlsTab, how: howTab };
    }

    _toggleTabs(isVisible) {
        if (!this.helpTabsBar) {
            return;
        }
        this.helpTabsBar.classList.toggle('is-visible', Boolean(isVisible));
    }

    _setOverlayMode(mode = 'default') {
        if (!this.helpOverlay) {
            return;
        }
        this.helpOverlay.classList.toggle('is-splash', mode === 'splash');
    }

    _renderTourScreen(screenId) {
        if (!this.helpOverlayContent) {
            return;
        }
        this.helpContentMode = 'tour';
        this.helpOverlay?.setAttribute('data-mode', 'tour');
        this.helpOverlay?.setAttribute('data-tour-screen', screenId);
        this._setOverlayMode('default');
        this._toggleTabs(false);
        this.helpOverlayContent.innerHTML = '';
        let screen = null;
        switch (screenId) {
            case 'tour_splash':
                this._setOverlayMode('splash');
                screen = this._buildSplashScreen();
                break;
            case 'tour_welcome':
                screen = this._buildWelcomeScreen();
                break;
            case 'tour_modes':
                screen = this._buildModeScreen();
                break;
            case 'tour_controls':
                screen = this._buildControlsScreen();
                break;
            case 'tour_canvas':
                screen = this._buildCanvasScreen();
                break;
            case 'tour_speed':
                screen = this._buildSpeedRunPreScreen();
                break;
            default:
                break;
        }
        if (screen) {
            this.helpOverlayContent.appendChild(screen);
        }
    }

    _buildSplashScreen() {
        const screen = document.createElement('div');
        screen.className = 'tour-screen tour-screen--splash';
        screen.classList.add('tour-screen--splash-active');
        const teardown = () => {
            window.removeEventListener('keydown', handleKey, true);
            window.removeEventListener('pointerdown', handlePointer, true);
        };
        const logo = document.createElement('img');
        logo.className = 'tour-splash__logo';
        logo.src = SPLASH_LOGO_URL;
        logo.alt = 'Human Descent logo';
        screen.append(logo);
        const dismiss = () => {
            if (!screen.classList.contains('tour-screen--splash-active')) return;
            screen.classList.remove('tour-screen--splash-active');
            screen.classList.add('tour-screen--splash-fade');
            teardown();
            clearTimeout(autoTimer);
            setTimeout(() => this._advanceHelpScreen(), 450);
        };
        const handleKey = () => dismiss();
        const handlePointer = () => dismiss();
        window.addEventListener('keydown', handleKey, true);
        window.addEventListener('pointerdown', handlePointer, true);
        const autoTimer = setTimeout(dismiss, 500);
        return screen;
    }

    _selectTourMode(mode) {
        if (this.helpTourSelection === mode) {
            return;
        }
        this.helpTourSelection = mode;
        if (this.helpOverlay?.getAttribute('data-tour-screen') === 'tour_modes') {
            this._renderTourScreen('tour_modes');
        }
    }

    _advanceHelpScreen() {
        if (this.helpContentMode !== 'tour') {
            return;
        }
        if (!this.state) {
            this.hideImage();
            return;
        }
        const prevIdx = this.state.helpScreenIdx;
        if (typeof this.state.nextHelpScreen === 'function') {
            this.state.nextHelpScreen();
        } else {
            this.state.helpScreenIdx = -1;
        }
        if (this.state.helpScreenIdx === -1 || this.state.helpScreenIdx === prevIdx) {
            this.hideImage();
            return;
        }
        const nextScreen = this.state.helpScreenFns?.[this.state.helpScreenIdx];
        if (nextScreen) {
            this.showImage(nextScreen);
        } else {
            this.hideImage();
        }
    }

    _attachHelpKeyListener() {
        if (this._helpKeyListenerAttached || typeof window === 'undefined') {
            return;
        }
        window.addEventListener('keydown', this._boundHelpKeydown, true);
        this._helpKeyListenerAttached = true;
    }

    _detachHelpKeyListener() {
        if (!this._helpKeyListenerAttached || typeof window === 'undefined') {
            return;
        }
        window.removeEventListener('keydown', this._boundHelpKeydown, true);
        this._helpKeyListenerAttached = false;
    }

    _handleHelpKeydown(event) {
        if (!this.state || this.state.helpScreenIdx === -1 || this.helpContentMode !== 'tour') {
            return;
        }
        const key = event.key?.toLowerCase?.();
        const activeScreen = this.helpOverlay?.getAttribute('data-tour-screen');
        if (activeScreen === 'tour_splash' && key !== 'escape') {
            return;
        }
        if (key === 'enter' || key === ' ') {
            event.preventDefault();
            this._advanceHelpScreen();
        } else if (key === 'escape') {
            event.preventDefault();
            if (typeof this.state.closeHelpScreens === 'function') {
                this.state.closeHelpScreens();
            } else {
                this.state.helpScreenIdx = -1;
            }
            this.hideImage();
        }
    }

    _getHudesClient() {
        if (typeof window !== 'undefined' && window.__hudesClient) {
            return window.__hudesClient;
        }
        return null;
    }

    _showHelpTabs(tabId = 'controls') {
        this._ensureHelpOverlayElements();
        const container = document.getElementById('imageContainer');
        if (!container || !this.helpOverlay || !this.helpOverlayContent) {
            return;
        }
        this.manualKeyOverlay = true;
        this.helpContentMode = 'tabs';
        container.classList.add('help-open');
        this._setBodyHelpOpen(true);
        this.helpOverlay.classList.add('visible');
        this.helpOverlay.setAttribute('data-mode', 'tabs');
        this._setOverlayMode('default');
        this._toggleTabs(true);
        this._setActiveHelpTab(tabId);
        this._renderHelpTab(tabId);
    }

    _setActiveHelpTab(tabId) {
        this.activeHelpTab = tabId;
        if (!this.helpTabs) {
            return;
        }
        Object.entries(this.helpTabs).forEach(([id, btn]) => {
            if (btn) {
                btn.classList.toggle('is-active', id === tabId);
            }
        });
    }

    _renderHelpTab(tabId) {
        if (!this.helpOverlayContent) {
            return;
        }
        this.helpOverlayContent.innerHTML = '';
        if (tabId === 'how') {
            const panel = document.createElement('div');
            panel.className = 'help-tab help-tab--how';
            const list = document.createElement('ul');
            HOW_IT_WORKS_POINTS.forEach((point) => {
                const item = document.createElement('li');
                item.innerHTML = point;
                list.appendChild(item);
            });
            panel.append(list);
            this.helpOverlayContent.appendChild(panel);
            return;
        }
        const panel = document.createElement('div');
        panel.className = 'help-tab help-tab--controls tour-screen tour-screen--keys';
        const list = document.createElement('ul');
        list.className = 'tour-legend';
        const rows = this.mobileMode ? MOBILE_CONTROLS_GRID : DESKTOP_CONTROLS_GRID;
        rows.forEach((row) => {
            const item = document.createElement('li');
            const key = document.createElement('span');
            key.className = 'tour-key';
            key.textContent = row.value;
            const label = document.createElement('span');
            label.className = 'tour-label';
            label.textContent = row.label;
            item.append(key, label);
            list.appendChild(item);
        });
        panel.append(list);
        this.helpOverlayContent.appendChild(panel);
    }

    openHelpOverlay(tabId = 'controls') {
        this._showHelpTabs(tabId);
    }

    openTutorialOverlay() {
        this._ensureHelpOverlayElements();
        const container = document.getElementById('imageContainer');
        if (!container || !this.helpOverlay || !this.helpOverlayContent) {
            return;
        }
        this.manualKeyOverlay = true;
        this.helpContentMode = 'tour';
        container.classList.add('help-open');
        this._setBodyHelpOpen(true);
        this.helpOverlay.classList.add('visible');
        this.helpOverlay.setAttribute('data-mode', 'tour');
        this.helpOverlay.setAttribute('data-tour-screen', 'tour_canvas');
        this._toggleTabs(false);
        this._renderTourScreen('tour_canvas');
    }

    isManualKeyOverlayVisible() {
        return Boolean(this.manualKeyOverlay);
    }

    _buildWelcomeScreen() {
        const screen = document.createElement('div');
        screen.className = 'tour-screen tour-screen--welcome';
        const hero = document.createElement('div');
        hero.className = 'tour-hero';

        const visual = document.createElement('div');
        visual.className = 'tour-hero__visual';
        const gif = document.createElement('img');
        gif.className = 'tour-hero__gif';
        gif.src = SPLASH_MEDIA_URL;
        gif.alt = 'CNN loss landscape animation';
        visual.appendChild(gif);

        const body = document.createElement('div');
        body.className = 'tour-hero__body';
        const eyebrow = document.createElement('p');
        eyebrow.className = 'tour-eyebrow';
        eyebrow.textContent = 'YOU ARE THE OPTIMIZER.';
        const list = document.createElement('ul');
        list.className = 'tour-welcome__list';
        [
            'The loss landscape, decoded: height = loss; lighter color = lower loss.',
            'X and Y are step sizes in weight space.',
            'Pick the right descent through saddles, steer around local minima, and balance batch size, validation frequency, and speed to steal the 2-minute MNIST crown.',
            'Whichever path you choose, we’ll see you at the bottom.',
        ].forEach((copy) => {
            const item = document.createElement('li');
            item.textContent = copy;
            list.appendChild(item);
        });
        const ctas = document.createElement('div');
        ctas.className = 'tour-cta-row';
        const playBtn = document.createElement('button');
        playBtn.type = 'button';
        playBtn.className = 'tour-btn primary';
        playBtn.textContent = 'Play';
        playBtn.addEventListener('click', (event) => {
            event?.preventDefault?.();
            this._advanceHelpScreen();
        });
        const infoBtn = document.createElement('button');
        infoBtn.type = 'button';
        infoBtn.className = 'tour-btn ghost';
        infoBtn.textContent = 'What is this?';
        infoBtn.addEventListener('click', (event) => {
            event?.preventDefault?.();
            this._showHelpTabs('how');
        });
        ctas.append(playBtn, infoBtn);

        body.append(eyebrow, list, ctas);
        hero.append(visual, body);
        screen.append(hero);
        return screen;
    }
    _buildModeScreen() {
        const screen = document.createElement('div');
        screen.className = 'tour-screen tour-screen--modes';
        const eyebrow = document.createElement('p');
        eyebrow.className = 'tour-eyebrow';
        eyebrow.textContent = 'Pick your mode';
        screen.appendChild(eyebrow);
        const modesWrap = document.createElement('div');
        modesWrap.className = 'tour-modes';
        MODE_CARDS.forEach((cardDef) => {
            const card = document.createElement('button');
            card.type = 'button';
            card.className = 'tour-mode-card';
            card.dataset.mode = cardDef.id;
            if (this.helpTourSelection === cardDef.id) {
                card.classList.add('is-selected');
            }
            card.innerHTML = `
                <div class="tour-mode-heading">
                    <h3>${cardDef.title}</h3>
                    <p>${cardDef.eyebrow}</p>
                </div>
                <p class="tour-mode-body">${cardDef.description}</p>
            `;
            card.addEventListener('click', (event) => {
                event?.preventDefault?.();
                this._selectTourMode(cardDef.id);
                modesWrap
                    .querySelectorAll('.tour-mode-card')
                    .forEach((btn) => btn.classList.toggle('is-selected', btn === card));
            });
            modesWrap.appendChild(card);
        });
        screen.appendChild(modesWrap);
        const continueBtn = document.createElement('button');
        continueBtn.type = 'button';
        continueBtn.className = 'tour-btn primary';
        continueBtn.textContent = 'Continue';
        continueBtn.addEventListener('click', (event) => {
            event?.preventDefault?.();
            this._commitModeSelection();
        });
        screen.appendChild(continueBtn);
        return screen;
    }
    _buildControlsScreen() {
        const screen = document.createElement('div');
        screen.className = 'tour-screen tour-screen--keys';
        const eyebrow = document.createElement('p');
        eyebrow.className = 'tour-eyebrow';
        eyebrow.textContent = 'Controls';
        const title = document.createElement('h2');
        title.textContent = this.mobileMode ? 'Touch controls' : 'Keyboard overlay';
        const list = document.createElement('ul');
        list.className = 'tour-legend';
        const rows = this.mobileMode ? MOBILE_CONTROLS_GRID : DESKTOP_CONTROLS_GRID;
        rows.forEach((row) => {
            const item = document.createElement('li');
            const key = document.createElement('span');
            key.className = 'tour-key';
            key.textContent = row.value;
            const label = document.createElement('span');
            label.className = 'tour-label';
            label.textContent = row.label;
            item.append(key, label);
            list.appendChild(item);
        });
        const footer = document.createElement('p');
        footer.className = 'tour-footer';
        footer.textContent = 'Press Enter to start. Press ? anytime.';
        const startBtn = document.createElement('button');
        startBtn.type = 'button';
        startBtn.className = 'tour-btn primary';
        startBtn.textContent = 'Start exploring';
        startBtn.addEventListener('click', (event) => {
            event?.preventDefault?.();
            this._advanceHelpScreen();
        });
        screen.append(eyebrow, title, list, footer, startBtn);
        return screen;
    }
    _buildCanvasScreen() {
        const screen = document.createElement('div');
        screen.className = 'tour-screen tour-screen--canvas';
        const eyebrow = document.createElement('p');
        eyebrow.className = 'tour-eyebrow';
        eyebrow.textContent = 'Micro-tutorial';
        const list = document.createElement('ul');
        list.className = 'tour-checklist';
        const completedCount = this._getCompletedTutorialCount();
        TUTORIAL_TASKS.forEach((task) => {
            const item = document.createElement('li');
            const isDone = this.tutorialProgress.get(task.id);
            item.className = isDone ? 'is-complete' : '';
            const head = document.createElement('div');
            head.className = 'tour-checklist__head';
            head.textContent = task.title;
            const detail = document.createElement('p');
            const copy = this.mobileMode ? task.detailMobile || task.detailDesktop : task.detailDesktop;
            detail.textContent = copy || '';
            item.append(head, detail);
            list.appendChild(item);
        });
        const progress = document.createElement('p');
        progress.className = 'tour-footer';
        progress.textContent = `${completedCount} of ${TUTORIAL_TASKS.length} complete`;
        screen.append(eyebrow, list, progress);
        return screen;
    }
    _buildSpeedRunPreScreen() {
        const screen = document.createElement('div');
        screen.className = 'tour-screen tour-screen--speed';
        const eyebrow = document.createElement('p');
        eyebrow.className = 'tour-eyebrow';
        eyebrow.textContent = 'Speed-Run Mode';
        const title = document.createElement('h2');
        title.textContent = '2-minute ranked challenge';
        const goal = document.createElement('p');
        goal.className = 'tour-desc';
        goal.textContent = 'Set your strategy, dive into the landscape, and chase your best val-loss before the clock runs out.';
        const rules = document.createElement('ul');
        rules.className = 'tour-rules';
        const ruleSet = this.mobileMode ? SPEEDRUN_RULES_MOBILE : SPEEDRUN_RULES;
        ruleSet.forEach((rule) => {
            const item = document.createElement('li');
            item.textContent = rule;
            rules.appendChild(item);
        });
        const compact = document.createElement('p');
        compact.className = 'tour-compact-keys';
        compact.textContent = this.mobileMode ? SPEEDRUN_COMPACT_KEYS_MOBILE : SPEEDRUN_COMPACT_KEYS;
        const footer = document.createElement('p');
        footer.className = 'tour-footer';
        footer.textContent = 'Timer starts on Start. Press ? anytime.';
        const startBtn = document.createElement('button');
        startBtn.type = 'button';
        startBtn.className = 'tour-btn primary';
        startBtn.textContent = 'Start Speed-Run';
        startBtn.addEventListener('click', (event) => {
            event?.preventDefault?.();
            this._startSpeedrunFromTour();
        });
        screen.append(eyebrow, title, goal, rules, compact, footer, startBtn);
        return screen;
    }
    _selectTourMode(mode) {
        this.helpTourSelection = mode;
    }

    _commitModeSelection() {
        const mode = this.helpTourSelection || 'explore';
        if (mode === 'speed') {
            this._startSpeedrunFlow();
        } else {
            this._startExploreFlow();
        }
    }

    _setTourFlow(flowKey) {
        const flow = TOUR_FLOWS[flowKey] || [];
        this.currentTourFlow = flowKey;
        this.state?.setHelpScreenFns(flow);
        if (!flow.length) {
            if (typeof this.state.closeHelpScreens === 'function') {
                this.state.closeHelpScreens();
            } else {
                this.state.helpScreenIdx = -1;
            }
            this.hideImage();
            return;
        }
        this.state.helpScreenIdx = 0;
        this.showImage(flow[0]);
    }

    _startExploreFlow() {
        this._resetTutorialProgress();
        this._setTourFlow('explore');
    }

    _startSpeedrunFlow() {
        this._setTourFlow('speed');
    }

    _completeTutorialIntro() {
        if (typeof this.state.closeHelpScreens === 'function') {
            this.state.closeHelpScreens();
        } else {
            this.state.helpScreenIdx = -1;
        }
        this.hideImage();
    }

    _startSpeedrunFromTour() {
        const client = this._getHudesClient();
        if (typeof this.state.closeHelpScreens === 'function') {
            this.state.closeHelpScreens();
        } else {
            this.state.helpScreenIdx = -1;
        }
        this.hideImage();
        client?.startSpeedRun?.();
    }
    _resetTutorialProgress() {
        this.tutorialProgress = new Map();
        TUTORIAL_TASKS.forEach((task) => this.tutorialProgress.set(task.id, false));
        this.tutorialComplete = false;
        if (this.tutorialToastEl) {
            this.tutorialToastEl.classList.remove('visible');
        }
    }

    _getCompletedTutorialCount() {
        let count = 0;
        this.tutorialProgress.forEach((value) => {
            if (value) count += 1;
        });
        return count;
    }

    notifyTutorialEvent(stepId) {
        if (!stepId || !this.tutorialProgress || !this.tutorialProgress.has(stepId)) {
            return;
        }
        if (this.tutorialProgress.get(stepId)) {
            return;
        }
        this.tutorialProgress.set(stepId, true);
        if (
            this.helpOverlay?.getAttribute('data-tour-screen') === 'tour_canvas' &&
            this.helpContentMode === 'tour'
        ) {
            this._renderTourScreen('tour_canvas');
        }
        if (!this.tutorialComplete && this._getCompletedTutorialCount() >= TUTORIAL_TASKS.length) {
            this.tutorialComplete = true;
            this._showTutorialCompletionToast();
        }
    }

    _showTutorialCompletionToast() {
        const toast = this._ensureTutorialToast();
        if (!toast) return;
        toast.textContent = 'Tutorial complete — press Z to start Speed-Run.';
        toast.classList.add('visible');
    }

    _ensureTutorialToast() {
        if (this.tutorialToastEl) {
            return this.tutorialToastEl;
        }
        if (typeof document === 'undefined') {
            return null;
        }
        const toast = document.createElement('div');
        toast.id = 'tutorialToast';
        document.body.appendChild(toast);
        this.tutorialToastEl = toast;
        return toast;
    }

    showLevelUp(level, isSpeedRun) {
        console.log('[LandscapeView] showLevelUp:', level, 'isSpeedRun:', isSpeedRun);
        if (typeof document === 'undefined') return;

        if (isSpeedRun) {
            const toast = document.getElementById('levelToast');
            const scoreEl = document.getElementById('levelToastScore');
            const titleEl = document.getElementById('levelToastTitle');
            if (toast && scoreEl && titleEl) {
                scoreEl.textContent = level.loss.toFixed(2);
                titleEl.textContent = level.title;
                toast.classList.add('visible');

                if (this.levelToastTimeout) {
                    clearTimeout(this.levelToastTimeout);
                }
                this.levelToastTimeout = setTimeout(() => {
                    toast.classList.remove('visible');
                    this.levelToastTimeout = null;
                }, 1500);
            }
        } else {
            const modal = document.getElementById('levelModal');
            const titleEl = document.getElementById('levelTitle');
            const insightEl = document.getElementById('levelInsight');
            const closeBtn = document.getElementById('levelCloseBtn');
            const eyebrowEl = modal?.querySelector('.level-eyebrow');

            if (modal && titleEl && insightEl) {
                if (eyebrowEl) {
                    eyebrowEl.textContent = `LEVEL UP! YOU REACHED LEVEL ${level.levelNumber}`;
                }
                titleEl.textContent = `${level.loss.toFixed(2)} — ${level.title}`;
                insightEl.textContent = level.insight;
                modal.classList.add('visible');

                // Clear any existing auto-close timer
                if (this.levelModalTimeout) {
                    clearTimeout(this.levelModalTimeout);
                }

                // Auto-close after 5 seconds
                this.levelModalTimeout = setTimeout(() => {
                    modal.classList.remove('visible');
                    this.levelModalTimeout = null;
                }, 5000);

                // Ensure we don't stack listeners
                if (!this._levelDismissHandler) {
                    this._levelDismissHandler = () => {
                        modal.classList.remove('visible');
                        if (this.levelModalTimeout) {
                            clearTimeout(this.levelModalTimeout);
                            this.levelModalTimeout = null;
                        }
                    };
                    closeBtn?.addEventListener('click', this._levelDismissHandler);
                }
            }
        }
    }

}
