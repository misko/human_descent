// Ownership follows the fixed match roster, including disconnected teammates.
export function selectableTeamPanels(owner, roster, count = 3) {
    if (!Number.isInteger(owner) || owner < 0 || owner >= count) return [];
    if (!Array.isArray(roster) || !roster.length) return [owner];
    const assigned = new Set(roster.map(member => Number(member.panelIndex)));
    return Array.from({ length: count }, (_, panel) => panel)
        .filter(panel => panel === owner || !assigned.has(panel));
}
