import { selectableTeamPanels } from './teamPanels.js';

const TERMINAL = new Set(['finished', 'cancelled', 'failed']);
const PHASE_ORDER = { preparing: 0, prepared: 1, countdown: 2, running: 3, finalizing: 4, finished: 5, cancelled: 5, failed: 5 };

// Gameplay owns this snapshot; neither social state nor browser defaults enter it.
export default class TeamGameState {
    constructor(launch) {
        for (const key of ['matchId', 'sessionId', 'playerId', 'token']) {
            if (typeof launch?.[key] !== 'string' || !launch[key]) throw new Error(`Missing launch ${key}`);
        }
        this.launch = { ...launch };
        this.generation = null;
        this.modelRevision = 0;
        this.phase = 'preparing';
        this.minimumJobId = 0;
        this.latestJobId = 0;
        this.roster = [];
        this.voteSentGeneration = null;
        this.updating = true;
    }

    apply(snapshot, jobId = 0) {
        if (!snapshot || snapshot.sessionId !== this.launch.sessionId ||
            snapshot.matchId !== this.launch.matchId || snapshot.playerId !== this.launch.playerId) return null;
        const generation = Number(snapshot.generation ?? 0);
        const revision = Number(snapshot.modelRevision ?? 0);
        if (this.generation !== null && (generation < this.generation ||
            (generation === this.generation && revision < this.modelRevision))) return null;
        if (this.terminal || (PHASE_ORDER[snapshot.phase] ?? 0) < (PHASE_ORDER[this.phase] ?? 0)) return null;
        const changed = generation !== this.generation;
        const surfaceChanged = changed || Number(snapshot.batchIdx || 0) !== this.batchIdx || snapshot.config?.meshStepSize !== this.config?.meshStepSize;
        this.minimumJobId = Math.max(this.minimumJobId, Number(jobId || 0));
        Object.assign(this, {
            generation, modelRevision: revision, phase: snapshot.phase, teamId: snapshot.teamId,
            panelIndex: Number(snapshot.panelIndex), dimsOffset: Number(snapshot.dimsOffset ?? 0),
            roster: (snapshot.roster || []).map(member => ({ ...member })),
            config: { ...snapshot.config }, results: (snapshot.results || []).map(result => ({ ...result })),
            startAt: Number(snapshot.startAt || 0), deadline: Number(snapshot.deadline || 0),
            batchIdx: Number(snapshot.batchIdx || 0),
            switchesRemaining: Object.hasOwn(snapshot, 'dimensionSwitchesRemaining') ? Number(snapshot.dimensionSwitchesRemaining) : null,
            reason: snapshot.reason || '',
        });
        if (surfaceChanged) this.updating = true;
        if (changed) {
            this.voteSentGeneration = null;
            this.updating = true;
        }
        if (this.roster.some(member => member.playerId === this.launch.playerId && member.voted)) {
            this.voteSentGeneration = generation;
        }
        return { generationChanged: changed, surfaceChanged };
    }

    acceptsResult(message) {
        if (this.terminal || this.generation === null || message.teamSessionId !== this.launch.sessionId ||
            message.teamMatchId !== this.launch.matchId || Number(message.generation ?? 0) !== this.generation ||
            Number(message.modelRevision ?? 0) < this.modelRevision ||
            Number(message.jobId || 0) < Math.max(this.minimumJobId, this.latestJobId)) return false;
        this.modelRevision = Number(message.modelRevision ?? 0);
        this.latestJobId = Number(message.jobId || 0);
        return true;
    }

    get terminal() { return TERMINAL.has(this.phase); }
    get inputEnabled() { return this.phase === 'running' && !this.updating && !this.terminal; }
    get canVote() { return this.inputEnabled && this.switchesRemaining !== 0 && this.voteSentGeneration !== this.generation; }
    get selectablePanels() { return selectableTeamPanels(this.panelIndex, this.roster); }
    ownsDimension(dim) { return Number.isInteger(dim) && dim >= 0 && dim < 6 && this.selectablePanels.includes(Math.floor(dim / 2)); }
}
