import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'; // Import OrbitControls

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
    Legend
);

import Plotly from 'plotly.js-dist-min'; // Import Plotly

import { log } from '../utils/logger.js'; // Import your logging utility

export default class View {
    constructor(gridSize, numGrids) {
        this.gridSize = gridSize;
        this.numGrids = numGrids;
        this.effectiveGrids = numGrids;

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
        this.selectedGridScale = 2*1.5; //1.5; // Scale multiplier for the selected grid
        // Three.js setup
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.domElement.style.position = 'absolute';
        this.renderer.domElement.style.top = '0';
        this.renderer.domElement.style.left = '0';
        this.renderer.domElement.style.width = '100%';
        this.renderer.domElement.style.height = '100%';
        document.body.appendChild(this.renderer.domElement);

        // Calculate camera position to fit all grids
        const fovRadians = (this.camera.fov * Math.PI) / 180;
        const totalWidth = (this.numGrids + this.selectedGridScale - 1.0) * this.gridSize + (this.numGrids - 1) * this.spacing;
        const distance = (totalWidth / 2) / Math.tan(fovRadians / 2);
        this.camera.position.set(0, 0, distance);


        // Orbit controls setup
        //this.controls = new OrbitControls(this.camera, this.renderer.domElement);

        // Colors for the grids
        this.gridColors = [
            new THREE.Color(0.0, 1.0, 1.0), // Cyan
            new THREE.Color(1.0, 0.0, 1.0), // Magenta
            new THREE.Color(1.0, 1.0, 0.0), // Yellow
            new THREE.Color(0.0, 1.0, 0.0), // Green
            new THREE.Color(1.0, 0.5, 0.0), // Orange
            new THREE.Color(1.0, 0.0, 0.0), // Red
            new THREE.Color(0.0, 0.0, 1.0), // Blue
            new THREE.Color(0.5, 0.0, 1.0), // Purple
            new THREE.Color(0.0, 0.5, 1.0), // Sky Blue
            new THREE.Color(1.0, 0.0, 0.5), // Pink
            new THREE.Color(0.5, 1.0, 0.0), // Lime
            new THREE.Color(1.0, 0.75, 0.8), // Light Pink
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
    // Function to handle window resizing
    onWindowResize() {
        // Update camera aspect ratio and projection matrix
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();

        // Update renderer size
        this.renderer.setSize(window.innerWidth, window.innerHeight);
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

    // Initialize Loss Chart
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
                    scales: { x: { title: { display: true, text: 'Step' } } },
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
                    scales: { x: { title: { display: true, text: 'Step' } } },
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
        if (document.getElementById('confusionMatrixChart')) {
            try {

                // Check the shape of the received confusionMatrix
                if (!Array.isArray(confusionMatrix)) {
                    throw new Error("Confusion matrix data must be an array");
                }

                const numRows = confusionMatrix.length;
                const numCols = numRows > 0 && Array.isArray(confusionMatrix[0]) ? confusionMatrix[0].length : 0;

                if (numRows === 0 || numCols === 0) {
                    throw new Error("Confusion matrix data is empty or improperly formatted");
                }

                // Validate each row to ensure consistent column length
                for (let i = 0; i < numRows; i++) {
                    if (!Array.isArray(confusionMatrix[i])) {
                        throw new Error(`Row ${i} of confusion matrix is not an array`);
                    }
                    if (confusionMatrix[i].length !== numCols) {
                        throw new Error(`Row ${i} of confusion matrix does not match expected column length of ${numCols}`);
                    }
                }

                // Prepare zData and determine zmax for the heatmap
                const zData = confusionMatrix;
                const zMaxValue = Math.max(...zData.flat());

                // Prepare the updated data for Plotly
                const updatedData = [
                    {
                        z: zData,
                        type: 'heatmap',
                        // colorscale: 'Blues',
                        // zmin: 0,
                        // zmax: zMaxValue, // Dynamically adjust max value for the heatmap
                    },
                ];

                // Simplify the layout to help determine the issue with scaling
                const updatedLayout = {
                    title: 'Confusion Matrix Heatmap',
                    xaxis: {
                        title: 'Predicted Label',
                        tickmode: 'linear',
                        dtick: 1, // Ensures each value has a corresponding tick mark
                        autorange: true, // Let Plotly handle the range
                    },
                    yaxis: {
                        title: 'Actual Label',
                        tickmode: 'linear',
                        dtick: 1,
                        autorange: 'reversed', // Reverse y-axis to match visualization expectations
                    },
                    responsive: true,
                };

                // Update the existing plot with new data and layout
                //Plotly.react(this.confusionMatrixContainer, updatedData, updatedLayout);
                Plotly.newPlot('confusionMatrixChart', updatedData)

                console.log("Confusion matrix heatmap updated successfully.");
            } catch (error) {
                console.error("Failed to update confusion matrix heatmap:", error);
            }
        }
    }
    updateMeshGrids(meshGrids = null) {
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
        const scale = 1.5*this.gridSize / (maxAbsValue + eps);
    
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
            const geometry = new THREE.PlaneGeometry(this.gridSize, this.gridSize, this.gridSize - 1, this.gridSize - 1);
            const color = this.gridColors[i % this.gridColors.length];
            let baseOpacity = this.alpha;
    
            // Apply lower opacity for non-selected grids
            if (i !== this.selectedGridIndex) {
                baseOpacity *= 0.5; // Make non-selected grids more transparent
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
                    alphas[j * this.gridSize + k] = zValue >= 0 ? baseOpacity * 0.3 : baseOpacity;
                }
            }
    
            geometry.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1)); // Add alpha as a vertex attribute
            geometry.attributes.position.needsUpdate = true;
    
            // Use ShaderMaterial to apply alpha for each vertex
            const material = new THREE.ShaderMaterial({
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
    
            const mesh = new THREE.Mesh(geometry, material);
            mesh.rotation.x = -Math.PI / 2; // Rotate to lay flat
    
            // Calculate position to center grids
            let xOffset = -totalWidth / 2 + i * (this.gridSize + this.spacing) + this.gridSize / 2;
    
            // Apply scaling if it's the selected grid
            if (i > this.selectedGridIndex) {
                xOffset += this.gridSize * (this.selectedGridScale - 1) ; // Adjust position for scaled grid
            }
            if (i === this.selectedGridIndex) {
                xOffset +=this.gridSize * (this.selectedGridScale - 1)/2
                mesh.scale.set(this.selectedGridScale, this.selectedGridScale, 1); // Scale selected grid on X, Z axes
            } else {
                mesh.scale.set(1, 1, 1); // Reset scale for non-selected grids
            }
    
            mesh.position.set(xOffset, 0, 0);
    
            // Add new grid to scene and store it for later reference
            this.scene.add(mesh);
            this.gridObjects.push(mesh);
    
            // Create a red sphere to represent the center point of each grid
            const sphereGeometry = new THREE.SphereGeometry(1, 16, 16); // Radius = 1, segments for smoother look
            const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 }); // Red color
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    
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
    render() {
        requestAnimationFrame(() => this.render());

        // Rotate each grid according to the current angles
        this.gridObjects.forEach((grid, index) => {
            grid.rotation.z = THREE.MathUtils.degToRad(this.angleH); // Rotate around Z axis
            grid.rotation.x = THREE.MathUtils.degToRad(this.angleV - 90); // Rotate around X axis
        });

        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }

}


