export default class ClientState {
    constructor(stepSizeResolution, initialStepSizeIdx) {
        this.bestScore = Infinity;
        this.sgdSteps = 0;
        this.evalSteps = 0;
        this.speedRunActive = false;
        this.speedRunSecondsRemaining = 0;
        this.dtypes = ["float16", "float32"];
        this.batchSizes = [2, 8, 32, 64, 128, 256, 512];
        this.n = 6;
        this.minLogStepSize = -4;
        this.maxLogStepSize = 2;
        this.stepSizeResolution = stepSizeResolution;
        this.stepSizeIdx = initialStepSizeIdx;
        this.helpScreenIdx = 0;
        this.dimsUsed = 0;
        this.updateStepSize();
    }

    updateBestScoreOrNot(newScore) {
        if (newScore < this.bestScore) {
            this.bestScore = newScore;
        }
    }

    toString() {
        const secs = Number(this.speedRunSecondsRemaining ?? 0);
        const timer = this.speedRunActive
            ? `, SPEED RUN: ${secs.toFixed(1)}s`
            : '';
        return `Best-val-loss: ${this.bestScore.toFixed(3)}, Batch-size: ${this.batchSize}, StepSize: ${this.stepSize.toExponential(3)}, SGD-steps: ${this.sgdSteps}, Eval-steps: ${this.evalSteps}, Dtype: ${this.dtype}${timer}`;
    }

    updateStepSize() {
        const logStepSize = Math.min(
            Math.max(this.stepSizeIdx * this.stepSizeResolution, this.minLogStepSize),
            this.maxLogStepSize
        );
        this.stepSize = Math.pow(10, logStepSize);
    }
    setEvalSteps(count) {
        this.evalSteps = count;
    }

    increaseStepSize(mag = 1) {
        this.stepSizeIdx += mag;
        this.updateStepSize();
    }

    decreaseStepSize(mag = 1) {
        this.stepSizeIdx -= mag;
        this.updateStepSize();
    }

    toggleBatchSize() {
        const currentIndex = this.batchSizes.indexOf(this.batchSize);
        const nextIndex = (currentIndex + 1) % this.batchSizes.length;
        this.batchSize = this.batchSizes[nextIndex];
    }
    setBatchSize(batchSize) {
        const batchSizeIdx = this.batchSizes.indexOf(batchSize);
        if (batchSizeIdx === -1) {
            console.error(`Invalid batch size: ${batchSize}. Available sizes are: ${this.batchSizes.join(', ')}`);
            return;
        }
        this.batchSizeIdx = batchSizeIdx;
        this.batchSize = batchSize;
        console.log(`Batch size set to ${batchSize}`);
    }
    toggleDtype() {
        const currentIndex = this.dtypes.indexOf(this.dtype);
        const nextIndex = (currentIndex + 1) % this.dtypes.length;
        this.dtype = this.dtypes[nextIndex];
    }
    setDtype(dtype) {
        const dtypeIdx = this.dtypes.indexOf(dtype);
        if (dtypeIdx === -1) {
            console.error(`Invalid dtype: ${dtype}. Available dtypes are: ${this.dtypes.join(', ')}`);
            return;
        }
        this.dtypeIdx = dtypeIdx;
        this.dtype = dtype;
        console.log(`Dtype set to ${dtype}`);
    }
    setHelpScreenFns(helpScreenFns) {
        if (Array.isArray(helpScreenFns)) {
            this.helpScreenFns = [...helpScreenFns];
            if (this.helpScreenFns.length === 0) {
                this.helpScreenIdx = -1;
            } else if (
                this.helpScreenIdx === -1 ||
                this.helpScreenIdx >= this.helpScreenFns.length
            ) {
                this.helpScreenIdx = 0;
            }
        } else {
            this.helpScreenFns = [];
            this.helpScreenIdx = -1;
        }
    }

    closeHelpScreens() {
        this.helpScreenIdx = -1;
    }

    nextHelpScreen() {
        if (this.helpScreenIdx === -1) return;
        this.helpScreenIdx += 1;
        if (this.helpScreenIdx >= this.helpScreenFns.length) {
            this.helpScreenIdx = -1;
        }
    }
}
