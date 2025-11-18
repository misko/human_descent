import LandscapeView from './views/LandscapeView.js';

export default class ViewRouter {
    constructor(gridSize, numGrids, clientState, options = {}) {
        const mode = (options.mode || '3d').toLowerCase();
        this.mode = mode === '1d' ? '1d' : '3d';
        const viewOptions = { ...options, mode: this.mode };
        this.impl = new LandscapeView(gridSize, numGrids, clientState, viewOptions);
    }

    initializeCharts(...args) {
        return this.impl?.initializeCharts?.(...args);
    }

    updateLossChart(...args) {
        return this.impl?.updateLossChart?.(...args);
    }

    updateLastStepsChart(...args) {
        return this.impl?.updateLastStepsChart?.(...args);
    }

    updateConfusionMatrix(...args) {
        return this.impl?.updateConfusionMatrix?.(...args);
    }

    updateExamplePreds(...args) {
        return this.impl?.updateExamplePreds?.(...args);
    }

    updateExamples(...args) {
        return this.impl?.updateExamples?.(...args);
    }

    updateMeshGrids(...args) {
        return this.impl?.updateMeshGrids?.(...args);
    }

    updateLossLines(...args) {
        return this.impl?.updateLossLines?.(...args);
    }

    annotateBottomScreen(...args) {
        return this.impl?.annotateBottomScreen?.(...args);
    }

    showImage(...args) {
        return this.impl?.showImage?.(...args);
    }

    hideImage(...args) {
        return this.impl?.hideImage?.(...args);
    }

    resetAngle(...args) {
        return this.impl?.resetAngle?.(...args);
    }

    adjustAngles(...args) {
        return this.impl?.adjustAngles?.(...args);
    }

    getAngles(...args) {
        return this.impl?.getAngles?.(...args);
    }

    getCanvasElement(...args) {
        return this.impl?.getCanvasElement?.(...args);
    }

    incrementSelectedGrid(...args) {
        return this.impl?.incrementSelectedGrid?.(...args);
    }

    decrementSelectedGrid(...args) {
        return this.impl?.decrementSelectedGrid?.(...args);
    }

    getSelectedGrid(...args) {
        return this.impl?.getSelectedGrid?.(...args);
    }

    selectGridAt(...args) {
        return this.impl?.selectGridAt?.(...args);
    }

    get modeName() {
        return this.mode;
    }

    highlightLossLine(...args) {
        return this.impl?.highlightLossLine?.(...args);
    }

    openHelpOverlay(...args) {
        return this.impl?.openHelpOverlay?.(...args);
    }

    isManualKeyOverlayVisible(...args) {
        return this.impl?.isManualKeyOverlayVisible?.(...args);
    }

    notifyTutorialEvent(...args) {
        return this.impl?.notifyTutorialEvent?.(...args);
    }

    openTutorialOverlay(...args) {
        return this.impl?.openTutorialOverlay?.(...args);
    }

}
