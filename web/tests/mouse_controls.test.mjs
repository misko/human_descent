import { installMouseControls, computeStepVector } from '../client/mouseControls.js';

const createStubElement = () => {
  const listeners = new Map();
  return {
    listeners,
    addEventListener(type, handler) {
      listeners.set(type, handler);
    },
    removeEventListener(type) {
      listeners.delete(type);
    },
    setPointerCapture() {},
    releasePointerCapture() {},
    onpointerdown: null,
  };
};

// Dragging should report horizontal and vertical deltas
(() => {
  const element = createStubElement();
  const deltas = [];
  installMouseControls(element, {
    onDrag: ({ deltaX, deltaY }) => deltas.push({ deltaX, deltaY }),
  });

  const down = element.listeners.get('pointerdown');
  const move = element.listeners.get('pointermove');

  if (!down || !move) {
    throw new Error('Mouse control listeners were not attached');
  }

  down({ button: 0, clientX: 100, clientY: 100, pointerId: 1 });
  move({ clientX: 80, clientY: 120 });
  move({ clientX: 60, clientY: 90 });

  if (deltas.length !== 2) {
    throw new Error('Expected two drag delta callbacks');
  }

  const [first, second] = deltas;
  if (first.deltaX !== -20 || first.deltaY !== 20) {
    throw new Error(`Unexpected first delta: ${JSON.stringify(first)}`);
  }
  if (second.deltaX !== -20 || second.deltaY !== -30) {
    throw new Error(`Unexpected second delta: ${JSON.stringify(second)}`);
  }
})();

// Wheel scrolling should forward events and call preventDefault
(() => {
  const element = createStubElement();
  let receivedDirection = 0;
  installMouseControls(element, {
    onScroll: ({ deltaY }) => {
      receivedDirection = deltaY > 0 ? 1 : -1;
    },
  });

  const wheel = element.listeners.get('wheel');
  if (!wheel) {
    throw new Error('Wheel listener was not attached');
  }

  let prevented = false;
  wheel({ deltaY: -5, preventDefault: () => { prevented = true; } });

  if (!prevented) {
    throw new Error('Wheel preventDefault was not called');
  }

  if (receivedDirection !== -1) {
    throw new Error('Scroll callback did not receive deltaY');
  }
})();

// Step vector helper should rotate based on horizontal angle
(() => {
  const vectorForward = computeStepVector(0, 'forward');
  if (Math.abs(vectorForward.x + 1) > 1e-6 || Math.abs(vectorForward.y) > 1e-6) {
    throw new Error('Forward vector should point along -X when angle is 0');
  }

  const vectorBackward = computeStepVector(90, 'backward');
  if (Math.abs(vectorBackward.x) > 1e-6 || Math.abs(vectorBackward.y + 1) > 1e-6) {
    throw new Error('Backward vector should align with -Y at 90 degrees');
  }
})();

// Click handlers should fire for left and right buttons
(() => {
  const element = createStubElement();
  let clickCount = 0;
  let contextCount = 0;

  installMouseControls(element, {
    onClick: () => {
      clickCount += 1;
    },
    onContext: () => {
      contextCount += 1;
    },
  });

  const down = element.listeners.get('pointerdown');
  if (!down) {
    throw new Error('Pointerdown listener missing');
  }

  down({ button: 0, clientX: 10, clientY: 15, pointerId: 7 });
  if (clickCount !== 1) {
    throw new Error('Left click did not trigger onClick');
  }

  down({
    button: 2,
    clientX: 12,
    clientY: 18,
    pointerId: 8,
    preventDefault: () => {},
  });
  if (contextCount !== 1) {
    throw new Error('Right click did not trigger onContext');
  }
})();
