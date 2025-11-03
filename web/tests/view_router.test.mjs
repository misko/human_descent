global.document = {
  getElementById() { return null; },
};

import View from '../client/View.js';

const dummyState = {
  toString() {
    return 'state';
  },
};

(() => {
  const view = new View(31, 3, dummyState, { mode: '3d' });
  if (view.modeName !== '3d') {
    throw new Error(`Expected 3d view, got ${view.modeName}`);
  }
  if (typeof view.updateMeshGrids !== 'function') {
    throw new Error('3d view should expose updateMeshGrids');
  }
})();

(() => {
  const view = new View(31, 0, dummyState, { mode: '1d', lossLines: 4 });
  if (view.modeName !== '1d') {
    throw new Error(`Expected 1d view, got ${view.modeName}`);
  }
  if (typeof view.updateLossLines !== 'function') {
    throw new Error('1d view should expose updateLossLines');
  }
})();
