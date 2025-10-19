import {
    Scene,
    PerspectiveCamera,
    MathUtils,
    WebGLRenderer,
    Color,
    PlaneGeometry,
    BufferAttribute,
    ShaderMaterial,
    Mesh,
    SphereGeometry,
    MeshBasicMaterial,
    Raycaster,
    Vector2,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; // Import OrbitControls
import annotationPlugin from 'chartjs-plugin-annotation';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js'; // Import CSS2DRenderer for annotations
import { CONTROL_GROUPS, formatHudMarkup } from './hud.js';
import { log } from '../utils/logger.js'; // Import your logging utility

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


export default class View {
    constructor(gridSize, numGrids, clientState) {
        this.gridSize = gridSize;
        this.numGrids = numGrids;
        this.effectiveGrids = numGrids;

        this.state=clientState;
        this.lossChart = null;
        this.lastStepsChart = null;
        this.stepSizeChart = null;
        this.dimsAndStepsChart = null;
        this.exampleImagesContainer = document.getElementById('exampleImages');
        this.confusionMatrixContainer = document.getElementById('confusionMatrixChart');


        // Grid properties
        this.gridObjects = [];
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
    // Function to handle window resizing
    onWindowResize() {

        // Calculate camera position to fit all grids
        const fovRadians = (this.camera.fov * Math.PI) / 180;
        const totalWidth = (this.numGrids + this.selectedGridScale - 1.0) * this.gridSize + (this.numGrids - 1) * this.spacing;
        const distance = (totalWidth / 2) / Math.tan(fovRadians / 2);
        this.camera.position.set(0, 0, distance);

        // Update camera aspect ratio and projection matrix
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();

        // Update renderer size
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.labelRenderer.setSize(window.innerWidth, window.innerHeight);
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
        if (this.scene == null) {
            return;
        }
        // Use previous meshGrids if no parameter is provided
        if (meshGrids === null && this.previousMeshGrids) {
            meshGrids = this.previousMeshGrids;
        } else {
            this.previousMeshGrids = meshGrids; // Save current meshGrids for reuse
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
    render() {
        requestAnimationFrame(() => this.render());

        // Rotate each grid according to the current angles
        this.gridObjects.forEach((grid, index) => {
            grid.rotation.z = MathUtils.degToRad(this.angleH); // Rotate around Z axis
            grid.rotation.x = MathUtils.degToRad(this.angleV - 90); // Rotate around X axis
        });

        // Render the scene
        this.renderer.render(this.scene, this.camera);
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
        // Select the image container div
        const container = document.getElementById('imageContainer');

        // Clear any previous image from the container
        container.innerHTML = '';

        // Create a new img element
        const img = document.createElement('img');
        img.src = filename; // Set the source to the provided filename
        img.alt = 'Floating Image';

        // Style the image
        img.style.position = 'absolute';
        img.style.top = '50%';
        img.style.left = '50%';
        img.style.transform = 'translate(-50%, -50%)'; // Center the image in the container
        img.style.zIndex = '9999'; // Ensure it appears above everything else
        img.style.pointerEvents = 'none'; // Allow clicks through the image
        img.style.height = '70vh'; // Set the image height to 70% of the window height
        img.style.width = 'auto'; // Maintain aspect ratio

        // Append the image to the container
        container.appendChild(img);
    }

    hideImage() {
        const container = document.getElementById('imageContainer');
        if (container) {
            container.innerHTML = ''; // Clear the container
        }
    }


}
