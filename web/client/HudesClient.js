import ClientState from './ClientState.js';
import { loadProto } from './ProtoLoader.js';
import { sleep } from '../utils/sleep.js';
import { log } from '../utils/logger.js';
import View from './View.js';

export default class HudesClient {
    constructor(addr, port) {
        this.socket = new WebSocket(`ws://${addr}:${port}`);
        this.socket.binaryType = "arraybuffer";
        this.requestIdx = 0;
        this.running = true;

        this.state = new ClientState(-0.05, 6);
        this.grids=3;
        this.grid_size=31;
        this.view = new View(this.grid_size,this.grids, this.state); // Add a View instance
        this.view.initializeCharts(); // Initialize charts
        this.Control = null;
        this.ControlType = null;
        this.keyHolds = {};
        this.cooldownTimeMs = 200;
        this.keySecondsPressed = 200; // ms
        this.stepSizeMultiplier = 1;


        // Initialize loss and step tracking arrays
        this.trainLosses = [];
        this.valLosses = [];
        this.trainSteps = [];
        this.valSteps = [];

        this.initKeys();
        this.setupSocket();
        this.loadProto();
    }

    async loadProto() {
        const { Control, ControlType } = await loadProto('hudes.proto');
        this.Control = Control;
        this.ControlType = ControlType;
        log("Proto file and enums loaded successfully.");
    }
    _handleMessage(event) {
        try {
            const buffer = new Uint8Array(event.data);
            const message = this.Control.decode(buffer); // Decode the protobuf message
            //log("Decoded Message:", message);

            switch (message.type) {
                case this.ControlType.values.CONTROL_TRAIN_LOSS_AND_PREDS:
                    if (!message.trainLossAndPreds) {
                        throw new Error("trainLossAndPreds field missing");
                    }

                    //log("hudes_client: receive message: loss and preds");

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

                    this.view.updateLastStepsChart(this.trainSteps.slice(-9),this.trainLosses.slice(-9));

                    // Update the view with predictions and confusion matrix
                    this.view.updateConfusionMatrix(confusionMatrix);
                    this.view.updateExamplePreds(trainPreds);

                    //log("hudes_client: receive message: loss and preds: done");

                    break;

                case this.ControlType.values.CONTROL_VAL_LOSS:
                    if (!message.valLoss) {
                        throw new Error("valLoss field missing");
                    }

                    //log(`Validation loss: ${message.valLoss.valLoss}`);

                    // Update validation loss
                    log('valloss',message.valLoss.valLoss);
                    log(message.valLoss.valLoss);
                    while (this.valLosses.length<this.trainSteps.length) {
                        this.valLosses.push(message.valLoss.valLoss);
                        this.valSteps.push(message.requestIdx);
                    }
                    log(this.valLosses.length);
                    log(this.trainLosses.length);
                    // if (this.valLosses.length>1) {
                    //     var diffLoss = this.valLosses[this.valLosses.length-1]-this.valLosses[this.valLosses.length-2];
                    //     var diffSteps = this.valSteps[this.valSteps.length-1]-this.valSteps[this.valSteps.length-2];
                    //     for (let i = 0; i < diffSteps; i++) {
                    //         console.log(i); // Will print numbers from 0 to 9
                    //     }
                    //     while (this.valLosses.length)<message.requestIdx:

                    // }


                    this.state.updateBestScoreOrNot(this.valLosses[this.valLosses.length - 1]);

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
                    } else {
                        //log("Batch example missing in message.");
                    }
                    break;


                case this.ControlType.values.CONTROL_MESHGRID_RESULTS:
                    if (!message.meshGridResults || !message.meshGridShape) {
                        throw new Error("meshGridResults or meshGridShape field missing");
                    }
    
                    log("hudes_client: received message: mesh grid results");
    
                    // Parse the mesh grid results and reshape to the original dimensions
                    const meshGridResultsFlattened = message.meshGridResults;
                    const meshGridShape = message.meshGridShape;
    
                    // Reshape the flattened results into a 3D array using meshGridShape
                    const meshGridResults = this._reshapeArray(
                        meshGridResultsFlattened,
                        meshGridShape
                    );
    
                    log(`Mesh grid results received:`, meshGridResults);
    
                    // Update the view with the reshaped mesh grid results
                    this.view.updateMeshGrids(meshGridResults);
                    break;


                case this.ControlType.values.CONTROL_SGD_STEP:
                    log(`SGD step acknowledged. Steps: ${message.sgdSteps}`);
                    break;
                default:
                    log(`Unknown message type: ${message.type}`);
            }
        } catch (error) {
            log(`Failed to handle message: ${error}`);
        }
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


    setupSocket() {
        this.socket.onopen = () => log("WebSocket connected.");
        this.socket.onclose = () => {
            log("WebSocket disconnected.");
            this.running = false;
        };
        this.socket.onmessage = (event) => this._handleMessage(event);
        this.socket.onerror = (error) => log(`WebSocket error: ${error}`);
    }

    initKeys() {
        this._handleKeyDown = this._handleKeyDown.bind(this);
        this._handleKeyUp = this._handleKeyUp.bind(this);

        window.addEventListener("keydown", this._handleKeyDown);
        window.addEventListener("keyup", this._handleKeyUp);
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
        const currentTime = performance.now();
        if (!this.keyHolds[event.code]) {
            this.keyHolds[event.code] = { firstPress: currentTime, lastExec: 0 };
        } else if (this.keyHolds[event.code].firstPress === -1) {
            this.keyHolds[event.code].firstPress = currentTime;
        }
        this.processKeyPress(event); // Call the processKeyPress method
    }

    _handleKeyUp(event) {
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
        const currentTime = Date.now();

        // Check cooldown
        const keyInfo = this.keyHolds[event.code];
        if (keyInfo && currentTime - keyInfo.lastExec < this.cooldownTimeMs) {
            return;
        }

        if (event.type === "keydown") {
            switch (event.code) {
                case "KeyX":
                    log("Next help screen.");
                    this.state.nextHelpScreen();
                    break;
                // default:
                //       log(`Unhandled key: ${event.code}`);
            }
        }

        if (this.state.helpScreenIdx!=-1) {
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
                    log("Performing SGD step.");
                    this.getSGDStep();
                    break;

                case "Space":
                    if (currentTime - keyInfo.lastExec > this.keySecondsPressed) {
                        log("Getting next dimensions.");
                        this.getNextDims();
                    }
                    break;

                case "BracketLeft":
                    log("Increasing step size.");
                    this.state.increaseStepSize(this.stepSizeMultiplier);
                    this.sendConfig();
                    break;

                case "BracketRight":
                    log("Decreasing step size.");
                    this.state.decreaseStepSize(this.stepSizeMultiplier);
                    this.sendConfig();
                    break;

                case "Quote":
                    log("Toggling dtype.");
                    this.state.toggleDtype();
                    this.sendConfig();
                    break;

                case "Semicolon":
                    log("Toggling batch size.");
                    this.state.toggleBatchSize();
                    this.sendConfig();
                    break;

                case "Enter":
                    log("Getting next batch.");
                    this.getNextBatch();
                    break;

            }

            // Update last execution time
            if (this.keyHolds[event.code]) {
                this.keyHolds[event.code].lastExec = currentTime;
            }
        }
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
        this.socket.send(buffer);

        // Increment the request index
        this.requestIdx++;

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

        const payload = {
            type: this.ControlType.values.CONTROL_SGD_STEP,
            sgdSteps: 1,
        };

        this.sendQ(payload);
        console.log("SGD step message sent.");
    }


    getNextBatch() {
        if (!this.Control || !this.ControlType) {
            console.error("Proto definitions not loaded. Cannot request next batch.");
            return;
        }

        const payload = {
            type: this.ControlType.values.CONTROL_NEXT_BATCH,
        };

        this.sendQ(payload);
        console.log("Next batch message sent.");
    }


    async getNextDims() {
        if (!this.Control || !this.ControlType) {
            console.error("Proto definitions not loaded. Cannot request next dims.");
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

        //console.log("Next dimensions requested and steps reset.");
    }


    zeroDimsAndStepsOnCurrentDims() {
        this.dimsAndStepsOnCurrentDims = new Array(this.state.n).fill(0);
        //console.log("Dims and steps reset to zero.");
    }
    dimsAndStepsUpdated() {
        if (this.view && typeof this.view.updateDimsSinceLastUpdate === 'function') {
            this.view.updateDimsSinceLastUpdate(this.dimsAndStepsOnCurrentDims || []);
        } else {
            console.warn("View or updateDimsSinceLastUpdate is not defined.");
        }
    }
    async sendDimsAndSteps(dimsAndSteps) {
        try {
            // Debug: Log control type status
            if (!this.Control || !this.ControlType) {
                console.error("Proto definitions not loaded. Cannot send dims and steps.");
                return;
            }
    
            // Construct the payload
            const payload = {
                type: this.ControlType.values.CONTROL_DIMS,
                dimsAndSteps: Object.entries(dimsAndSteps).map(([dim, step]) => ({
                    dim: parseInt(dim, 10),
                    step,
                })),
                requestIdx: this.requestIdx,
            };
    
            // Debug: Log the payload before sending
            //console.log("sendDimsAndSteps - Payload:", payload);
    
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
    
            // Increment the request index
            this.requestIdx++;
    
        } catch (error) {
            console.error("Error in sendDimsAndSteps:", error);
        }
    }
    

    async sendConfig() {
        if (!this.Control || !this.ControlType) {
            console.error("Proto definitions not loaded. Cannot send config.");
            return;
        }

        // Create config payload
        const payload = {
            type: this.ControlType.values.CONTROL_CONFIG,
            config: {
                seed: Math.floor(Math.random() * 1000),
                dimsAtATime: this.state.n,
                meshGridSize: this.grid_size,
                meshStepSize: this.state.stepSize,
                meshGrids: this.grids,
                batchSize: this.state.batchSize,
                dtype: this.state.dtype,
            },
        };

        // Use sendQ to send the message
        this.sendQ(payload);

        //console.log("Config sent.");
    }



    async runLoop() {
        while (!this.Control || !this.ControlType) {
            await sleep(100);
        }
        this.sendConfig();
        while (this.running) {
            await sleep(10);
        }
        log("Exiting run loop...");
    }
}
