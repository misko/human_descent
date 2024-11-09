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
    constructor() {
        this.lossChart = null;
        this.lastStepsChart = null;
        this.stepSizeChart = null;
        this.dimsAndStepsChart = null;
        this.exampleImagesContainer = document.getElementById('exampleImages');
        this.confusionMatrixContainer = document.getElementById('confusionMatrixChart');
        
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

    // Initialize Loss Chart
    initializeLossChart() {
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

    // Initialize Last Steps Chart
    initializeLastStepsChart() {
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

    // Initialize Step Size Chart
    initializeStepSizeChart() {
        this.stepSizeChart = new Chart(document.getElementById('stepSizeChart'), {
            type: 'bar',
            data: {
                labels: ['Step Size'],
                datasets: [{ label: 'Log Step Size', data: [], backgroundColor: 'orange' }],
            },
            options: { responsive: true, indexAxis: 'y' },
        });
    }

    // Initialize Dims and Steps Chart
    initializeDimsAndStepsChart() {
        this.dimsAndStepsChart = new Chart(document.getElementById('dimsAndStepsChart'), {
            type: 'bar',
            data: {
                labels: [], // Dimensions
                datasets: [{ label: 'Cumulative Step', data: [], backgroundColor: 'teal' }],
            },
            options: { responsive: true, scales: { x: { title: { display: true, text: 'Dimension' } } } },
        });
    }


    // Update Loss Chart
    updateLossChart(trainLoss, valLoss, steps) {
        log("Update loss.");
        log(trainLoss);
        this.lossChart.data.labels = steps;
        this.lossChart.data.datasets[0].data = trainLoss;
        this.lossChart.data.datasets[1].data = valLoss;
        this.lossChart.update();
    }

    // Update Last Steps Chart
    updateLastStepsChart(lastSteps, losses) {
        this.lastStepsChart.data.labels = lastSteps;
        this.lastStepsChart.data.datasets[0].data = losses;
        this.lastStepsChart.update();
    }

    // Update Step Size Chart
    updateStepSizeChart(logStepSize) {
        this.stepSizeChart.data.datasets[0].data = [logStepSize];
        this.stepSizeChart.update();
    }

    // Update Dims and Steps Chart
    updateDimsAndStepsChart(dimLabels, steps) {
        this.dimsAndStepsChart.data.labels = dimLabels;
        this.dimsAndStepsChart.data.datasets[0].data = steps;
        this.dimsAndStepsChart.update();
    }
    // Update Confusion Matrix Heatmap with new data
    updateConfusionMatrix(confusionMatrix) {
        try {
            console.log("Received confusion matrix data:", confusionMatrix);

            // Check the shape of the received confusionMatrix
            if (!Array.isArray(confusionMatrix)) {
                throw new Error("Confusion matrix data must be an array");
            }

            const numRows = confusionMatrix.length;
            const numCols = numRows > 0 && Array.isArray(confusionMatrix[0]) ? confusionMatrix[0].length : 0;

            console.log(`Confusion Matrix Shape: Rows = ${numRows}, Columns = ${numCols}`);

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

            console.log("Confusion Matrix Data:", zData);
            console.log("Maximum value in confusion matrix:", zMaxValue);

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

            console.log("Updating confusion matrix heatmap...");

            // Update the existing plot with new data and layout
            //Plotly.react(this.confusionMatrixContainer, updatedData, updatedLayout);
            Plotly.newPlot('confusionMatrixChart',updatedData)

            console.log("Confusion matrix heatmap updated successfully.");
        } catch (error) {
            console.error("Failed to update confusion matrix heatmap:", error);
        }
    }


    // Update Example Images
    updateExamples(images, labels) {
        log('Received images for update');
        this.exampleImagesContainer.innerHTML = ''; // Clear previous images

        images.forEach((imageData, index) => {
            const canvas = document.createElement('canvas');
            canvas.width = imageData[0].length; // Assuming the data is [height][width]
            canvas.height = imageData.length;

            const ctx = canvas.getContext('2d');
            const imageDataObject = ctx.createImageData(canvas.width, canvas.height);

            // Assuming the data is grayscale (0-1), set pixels accordingly
            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    const pixelIndex = (y * canvas.width + x) * 4;
                    const pixelValue = imageData[y][x];

                    imageDataObject.data[pixelIndex] = pixelValue * 255;     // Red
                    imageDataObject.data[pixelIndex + 1] = pixelValue * 255; // Green
                    imageDataObject.data[pixelIndex + 2] = pixelValue * 255; // Blue
                    imageDataObject.data[pixelIndex + 3] = 255;        // Alpha (fully opaque)
                }
            }

            // Put the image data on the canvas
            ctx.putImageData(imageDataObject, 0, 0);

            // Convert canvas to data URL and add it as an image
            const img = document.createElement('img');
            img.src = canvas.toDataURL();
            img.alt = `Example ${index}`;
            this.exampleImagesContainer.appendChild(img);
        });
    }
}
