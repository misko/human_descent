export default class TeamGameHud {
    constructor(onReturn) {
        this.onReturn = onReturn;
        this.root = document.createElement('section');
        this.root.id = 'teamGameStatus';
        this.root.setAttribute('aria-label', 'Team status');
        this.title = document.createElement('span');
        this.title.className = 'team-hud-label';
        this.title.textContent = 'Team';
        this.switches = document.createElement('div');
        this.switches.dataset.testid = 'team-switches-remaining';
        this.switches.className = 'team-hud-switches';
        this.switches.setAttribute('aria-live', 'polite');
        this.members = document.createElement('div');
        this.members.className = 'team-hud-members';
        this.readiness = document.createElement('span');
        this.readiness.className = 'team-hud-readiness';
        this.readiness.setAttribute('aria-live', 'polite');
        this.root.append(this.title, this.members, this.readiness, this.switches);
        document.body.append(this.root);
    }

    update(team) {
        const votes = team.roster.filter(member => member.voted).length;
        this.readiness.textContent = team.switchesRemaining === 0 ? 'Directions locked' : `${votes}/${team.roster.length} ready`;
        this.root.dataset.phase = team.phase;
        this.root.dataset.generation = String(team.generation ?? '');
        this.switches.textContent = `Switches left: ${team.switchesRemaining ?? 'Unlimited'}`;
        this.root.title = team.switchesRemaining === 0
            ? 'No new direction sets left. You can still select and optimize your own or unassigned landscapes.'
            : 'Shift or click selects your own or unassigned landscapes. The third landscape is shared in two-player teams. Space requests new dimensions; everyone must request.';
        const signature = JSON.stringify(team.roster);
        if (signature !== this._rosterSignature) {
            this._rosterSignature = signature;
            this.members.replaceChildren(...team.roster.map(member => {
                const item = document.createElement('span');
                item.dataset.teamPlayer = member.playerId;
                item.dataset.panelIndex = String(member.panelIndex);
                item.dataset.voted = String(Boolean(member.voted));
                item.dataset.connected = String(Boolean(member.connected));
                const name = member.name || `Player ${member.panelIndex + 1}`;
                const own = member.playerId === team.launch.playerId;
                const status = !member.connected ? 'offline' : member.voted ? 'requested new directions' : 'playing';
                item.title = `${name}${own ? ' (you)' : ''} · ${status}`;
                item.setAttribute('aria-label', item.title);
                const dot = document.createElement('span');
                dot.className = 'team-hud-dot';
                dot.setAttribute('aria-hidden', 'true');
                const label = document.createElement('span');
                label.className = 'team-hud-name';
                label.textContent = name;
                item.append(dot, label);
                if (own) {
                    const you = document.createElement('span');
                    you.className = 'team-hud-you';
                    you.textContent = 'you';
                    item.append(you);
                }
                return item;
            }));
        }
        if (team.terminal && !this.results) this.showResults(team);
    }

    showResults(team) {
        this.results = document.createElement('div');
        this.results.id = 'teamGameResults';
        this.results.setAttribute('role', 'dialog');
        const title = document.createElement('h2');
        const ownTeamId = team.launch.teamId || team.teamId;
        const own = team.results.find(result => result.teamId === ownTeamId);
        const opponents = team.results.filter(result => result.teamId !== ownTeamId);
        let outcome = team.phase;
        if (team.phase === 'finished') {
            outcome = 'complete';
            if (own && Number.isFinite(own.loss) && opponents.length && opponents.every(result => Number.isFinite(result.loss))) {
                const bestOpponent = Math.min(...opponents.map(result => result.loss));
                outcome = own.loss < bestOpponent ? 'win' : own.loss > bestOpponent ? 'loss' : 'tie';
            }
        }
        this.results.dataset.result = outcome;
        title.textContent = ({ win: 'Your team wins!', loss: 'Opponent wins', tie: 'Match tied', complete: 'Run complete' })[outcome] || `Match ${team.phase}`;
        const scores = document.createElement('p');
        scores.textContent = team.results.map(result => `${result.teamId === ownTeamId ? 'Your team' : 'Opponent'}: ${Number.isFinite(result.loss) ? result.loss.toFixed(4) : '—'}`).join(' · ') || team.reason;
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.action = 'return-lobby';
        button.textContent = 'Return to Lobby';
        button.onclick = () => this.onReturn?.();
        this.results.append(title, scores, button);
        this.root.append(this.results);
    }

    destroy() { this.root?.remove(); this.onReturn = null; }
}
