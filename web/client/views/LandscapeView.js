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
import { CONTROL_GROUPS, formatHudMarkup } from '../hud.js';
import { log } from '../../utils/logger.js'; // Import your logging utility

const DEBUG_SELECTION = false;

const debugSelection = (message) => {
    if (DEBUG_SELECTION) {
        log(message);
    }
};

//import Plotly from 'plotly.js-dist-min'; // Import Plotly
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

        this.state=clientState;
        this.lossChart = null;
        this.lastStepsChart = null;
        this.stepSizeChart = null;
        this.dimsAndStepsChart = null;
        this.exampleImagesContainer = document.getElementById('exampleImages');
        this.confusionMatrixContainer = document.getElementById('confusionMatrixChart');
        this.helpOverlay = null;
        this.helpImageEl = null;


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

            this.raycaster = new Raycaster();
            this.pointer = new Vector2();

            // Set the y-offset for the camera to position the grids 2/3 of the way down the view
            // The y offset is calculated as (1/3 of the totalHeight) since we want the grids to be 2/3 of the way down
            this.camera_yOffset = (totalWidth / 8);

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
    annotateBottomScreen(text, size = 20) {
        const bottomTextContainer = document.getElementById('bottomTextContainer');
        if (bottomTextContainer) {
            bottomTextContainer.innerHTML = formatHudMarkup(
                text ?? '',
                CONTROL_GROUPS,
            );
        }
    }
    highlightLossLine(index, durationMs = this.glowDuration) {
        if (this.mode !== '1d') {
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

        // Calculate camera position to fit all grids
        const fovRadians = (this.camera.fov * Math.PI) / 180;
        const totalWidth = (this.numGrids + this.selectedGridScale - 1.0) * this.gridSize + (this.numGrids - 1) * this.spacing;
        let distance = (totalWidth / 2) / Math.tan(fovRadians / 2);
        if (this.customCameraDistance != null) {
            distance = this.customCameraDistance;
        }
        this.camera_distance = distance;
        this.camera_yOffset = (totalWidth / 8);
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




    // Initialize all charts (including Plotly-based heatmap)
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
        if (this.confusionMatrixContainer) {
            const initialData = [
                {
                    z: [], // Initialize with empty data
                    type: 'heatmap',
                    colorscale: 'Blues',
                    zmin: 0,
                    zmax: 255,
                },
            ];

            const layout = {
                title: 'Confusion Matrix Heatmap',
                xaxis: {
                    title: 'Predicted Label',
                    tickmode: 'linear',
                },
                yaxis: {
                    title: 'Actual Label',
                    tickmode: 'linear',
                    autorange: 'reversed', // Reverse y-axis to match visualization expectations
                },
                responsive: true,
            };

            Plotly.newPlot(this.confusionMatrixContainer, initialData, layout);
        }
    }

    initializeLossChart() {
        if (document.getElementById('lossChart')) {
            this.lossChart = new Chart(document.getElementById('lossChart'), {
                type: 'line',
                data: {
                    labels: [], // Step numbers
                    datasets: [
                        { label: 'Train Loss', data: [], borderColor: 'blue', fill: false },
                        { label: 'Validation Loss', data: [], borderColor: 'red', fill: false },
                    ],
                },
                options: {
                    responsive: true,
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
                        point: {
                            radius: 3,
                        },
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Step',
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
        if (document.getElementById('lastStepsChart')) {
            this.lastStepsChart = new Chart(document.getElementById('lastStepsChart'), {
                type: 'line',
                data: {
                    labels: [], // Recent steps
                    datasets: [{ label: 'Loss (Last Steps)', data: [], borderColor: 'green', fill: false }],
                },
                options: {
                    responsive: true,
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
                        point: {
                            radius: 3,
                        },
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Step',
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
        if (this.lastStepsChart) {
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
        if (!document.getElementById('confusionMatrixChart')) {
            console.error("Element with id 'confusionMatrixChart' not found.");
            return;
        }

        try {
            // Validate input matrix
            if (!Array.isArray(confusionMatrix) || confusionMatrix.length === 0) {
                throw new Error("Confusion matrix data must be a non-empty array.");
            }

            const numRows = confusionMatrix.length;
            const numCols = Array.isArray(confusionMatrix[0]) ? confusionMatrix[0].length : 0;

            if (numCols === 0 || !confusionMatrix.every(row => Array.isArray(row) && row.length === numCols)) {
                throw new Error("Confusion matrix data is empty or improperly formatted.");
            }

            // Prepare data and layout for Plotly
            const zMaxValue = Math.max(...confusionMatrix.flat());
            const updatedData = [
                {
                    z: confusionMatrix,
                    type: 'heatmap',
                    colorscale: 'Viridis', // Colorscale suitable for dark background
                    zmin: 0,
                    zmax: zMaxValue,
                },
            ];

            const updatedLayout = {
                title: {
                    text: 'Confusion Matrix Heatmap',
                    font: {
                        color: 'white'
                    }
                },
                xaxis: {
                    title: {
                        text: 'Predicted Label',
                        font: { color: 'white' }
                    },
                    tickmode: 'linear',
                    dtick: 1,
                    tickfont: { color: 'white' },
                    gridcolor: 'rgba(255, 255, 255, 0.2)', // Light grid lines for visibility
                },
                yaxis: {
                    title: {
                        text: 'Actual Label',
                        font: { color: 'white' }
                    },
                    tickmode: 'linear',
                    dtick: 1,
                    autorange: 'reversed',
                    tickfont: { color: 'white' },
                    gridcolor: 'rgba(255, 255, 255, 0.2)',
                },
                margin: { t: 30, l: 1, r: 1, b: 1 }, // Small margins for compactness
                plot_bgcolor: 'rgba(0, 0, 0, 0)', // Transparent plot background
                paper_bgcolor: 'rgba(0, 0, 0, 0)', // Transparent paper background
                responsive: true,
            };

            // Plot or update the heatmap
            Plotly.newPlot('confusionMatrixChart', updatedData, updatedLayout);
            //console.log("Confusion matrix heatmap updated successfully.");
        } catch (error) {
            console.error("Failed to update confusion matrix heatmap:", error);
        }
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
        const exampleCells = sideContainer.getElementsByClassName('example-cell');

        // Clear the current images in each cell
        Array.from(exampleCells).forEach((exampleCell, index) => {
            // Clear previous canvas if exists
            const previousImage = exampleCell.querySelector('canvas');
            if (previousImage) {
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
        const exampleCells = sideContainer.getElementsByClassName('example-cell');

        predictions.forEach((prediction, index) => {
            // Get or create chart container for the corresponding example cell
            if (exampleCells.length<=index) {
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

            // Create Plotly bar chart for predicted classifications
            const trace = {
                x: Array.from({ length: 10 }, (_, i) => i), // Labels 0 to 9
                y: prediction, // Corresponding probabilities
                type: 'bar',
                marker: {
                    color: 'rgba(255, 165, 0, 0.8)' // Bright orange bars for visibility on a black background
                }
            };

            const layout = {
                plot_bgcolor: 'rgba(0, 0, 0, 0)', // Transparent plot background
                paper_bgcolor: 'rgba(0, 0, 0, 0)', // Transparent paper background
                margin: { t: 20, l: 30, r: 20, b: 40 },
                xaxis: {
                    title: 'Class',
                    titlefont: { size: 10, color: '#ffffff' }, // White text for better contrast
                    tickfont: { size: 8, color: '#ffffff' } // White tick labels
                },
                yaxis: {
                    title: 'Probability',
                    titlefont: { size: 10, color: '#ffffff' }, // White text for better contrast
                    tickfont: { size: 8, color: '#ffffff' }, // White tick labels
                    range: [0, 2] // Set y-axis limit to [0, 2]
                },
                showlegend: false // Hide legend to save space
            };

            Plotly.newPlot(chartDiv, [trace], layout, { displayModeBar: false });
        });
    }

    showImage(filename) {
        const container = document.getElementById('imageContainer');
        if (!container) {
            return;
        }

        if (!this.helpOverlay) {
            const overlay = document.createElement('div');
            overlay.className = 'help-overlay';

            const closeBtn = document.createElement('button');
            closeBtn.type = 'button';
            closeBtn.className = 'help-overlay__close';
            closeBtn.setAttribute('aria-label', 'Close help screens');
            closeBtn.innerHTML = '&times;';
            closeBtn.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();
                if (this.state) {
                    if (typeof this.state.closeHelpScreens === 'function') {
                        this.state.closeHelpScreens();
                    } else {
                        this.state.helpScreenIdx = -1;
                    }
                }
                this.hideImage();
            });

            const img = document.createElement('img');
            img.className = 'help-overlay__image';
            img.alt = 'Help screen';

            overlay.appendChild(closeBtn);
            overlay.appendChild(img);
            container.appendChild(overlay);

            this.helpOverlay = overlay;
            this.helpImageEl = img;
        }

        const overlay = this.helpOverlay;
        const img = this.helpImageEl;
        if (!overlay || !img) {
            return;
        }

        container.classList.add('help-open');
        overlay.classList.add('visible');

        if (img.dataset.currentSrc === filename) {
            return;
        }

        img.dataset.currentSrc = filename;
        img.src = filename;
    }

    hideImage() {
        const container = document.getElementById('imageContainer');
        if (!container) {
            return;
        }
        container.classList.remove('help-open');

        if (this.helpOverlay) {
            this.helpOverlay.classList.remove('visible');
        } else {
            const overlay = container.querySelector('.help-overlay');
            if (overlay) {
                overlay.classList.remove('visible');
            }
        }
    }


}
