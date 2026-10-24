import assert from 'node:assert/strict';
import { Scene, Color } from 'three';
import LandscapeView from '../client/views/LandscapeView.js';

globalThis.document = { getElementById: () => null, body: { classList: { remove() {} } } };
globalThis.window = { removeEventListener() {} };
const grids = () => Array.from({ length: 3 }, () => [[2, 1, 2], [1, 0, 1], [2, 1, 2]]);
for (const size of [1, 2, 3]) {
    for (let owner = 0; owner < size; owner++) {
        const view = new LandscapeView(3, 3, { helpScreenIdx: -1 }, { teamMode: true });
        const roster = Array.from({ length: size }, (_, panelIndex) => ({ panelIndex, voted: panelIndex === 0 }));
        const snapshot = { panelIndex: owner, roster, config: { meshGridSize: 3, meshGrids: 3 } };
        view.configureTeam(snapshot);
        view.scene = new Scene();
        view.gridColors = [new Color('cyan'), new Color('magenta'), new Color('yellow')];
        view.setTeamReadiness(roster);
        view.updateMeshGrids(grids());
        assert.equal(view.numGrids, 3, `${size}-player teams still show three landscapes`);
        assert.equal(view.gridObjects.length, 3);
        assert.deepEqual(view.gridObjects.map(mesh => mesh.scale.x), [0, 1, 2].map(i => i === owner ? view.selectedGridScale : 1));
        let disposed = 0;
        view.readyFrames[0].geometry.addEventListener('dispose', () => disposed++);
        const allowed = [0, 1, 2].filter(panel => panel === owner || panel >= size);
        assert.deepEqual(view.teamSelectablePanels, allowed);
        view.incrementSelectedGrid();
        assert.equal(view.getSelectedGrid(), allowed[(allowed.indexOf(owner) + 1) % allowed.length]);
        view.decrementSelectedGrid();
        assert.equal(view.getSelectedGrid(), owner);
        for (const panel of [0, 1, 2, -1, 3, 0.5, NaN]) {
            view.setSelectedGrid(owner);
            view.selectGrid(panel);
            assert.equal(view.getSelectedGrid(), allowed.includes(panel) ? panel : owner);
        }
        const selected = allowed.at(-1);
        view.setSelectedGrid(selected);
        view.configureTeam({ ...snapshot, roster: roster.map(member => ({ ...member, connected: false })) });
        view.selectGridAt(0, 0);
        view.updateMeshGrids(grids());
        assert.equal(view.getSelectedGrid(), selected, 'status and reconnect snapshots preserve selection');
        assert.deepEqual(view.gridObjects.map(mesh => mesh.scale.x), [0, 1, 2].map(i => i === selected ? view.selectedGridScale : 1));
        assert.equal(disposed, 1, 'mesh rebuild disposes the previous persistent frame');
        assert.equal(view.readyFrames[0].visible, true, 'readiness survives movement and mesh rebuilds');
        assert.equal(view.readyFrames[2].visible, false, 'unassigned or unready landscapes have no vote border');
        view.setTeamReadiness(roster.map(member => ({ ...member, voted: false })));
        assert.equal(view.readyFrames[0].visible, false);
        view.destroy();
        view.destroy();
        view.updateMeshGrids(grids());
        assert.deepEqual(view.gridObjects, [], 'destroyed view cannot resurrect graphics');
    }
}
const solo = new LandscapeView(3, 3, { helpScreenIdx: -1 });
solo.incrementSelectedGrid();
assert.equal(solo.getSelectedGrid(), 1, 'solo players can still change the controlled landscape');
solo.decrementSelectedGrid();
assert.equal(solo.getSelectedGrid(), 0);
solo.selectGrid(2);
assert.equal(solo.getSelectedGrid(), 2);
solo.destroy();

// Identical sampled losses must produce the solo appearance at full resolution.
for (const owner of [0, 1, 2]) {
    const soloView = new LandscapeView(31, 3, { helpScreenIdx: -1 });
    const teamView = new LandscapeView(31, 3, { helpScreenIdx: -1 }, { teamMode: true });
    soloView.setSelectedGrid(owner);
    teamView.configureTeam({ panelIndex: owner, config: { meshGridSize: 31, meshGrids: 3 } });
    for (const view of [soloView, teamView]) {
        view.scene = new Scene();
        view.gridColors = [new Color('cyan'), new Color('magenta'), new Color('yellow')];
        view.updateMeshGrids(Array.from({ length: 3 }, (_, p) => Array.from({ length: 31 }, (_, y) =>
            Array.from({ length: 31 }, (_, x) => 2 + (x - 15) * .02 + (y - 15) ** 2 * .001 * (p + 1)))));
    }
    for (let panel = 0; panel < 3; panel++) {
        const a = soloView.gridObjects[panel], b = teamView.gridObjects[panel];
        assert.deepEqual(b.geometry.attributes.position.array, a.geometry.attributes.position.array);
        assert.deepEqual(b.geometry.attributes.alpha.array, a.geometry.attributes.alpha.array);
        assert.deepEqual(b.position, a.position);
        assert.deepEqual(b.scale, a.scale);
        assert.deepEqual(b.rotation.toArray(), a.rotation.toArray());
        assert.equal(b.material.wireframe, a.material.wireframe);
        assert.equal(b.material.uniforms.color.value.getHex(), a.material.uniforms.color.value.getHex());
    }
    soloView.destroy();
    teamView.destroy();
}
