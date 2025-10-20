import { log } from '../utils/logger.js';

const DEBUG_MOUSE = false;

const debugLog = (message) => {
  if (DEBUG_MOUSE) {
    log(message);
  }
};

const defaultWindow = typeof window !== 'undefined' ? window : undefined;

export function installMouseControls(
  element,
  {
    onDrag,
    onScroll,
    onClick,
    onContext,
    moveTarget = defaultWindow,
  } = {},
) {
  if (!element || typeof element.addEventListener !== 'function') {
    throw new Error('installMouseControls: element must support addEventListener');
  }

  const target = moveTarget ?? defaultWindow ?? element;
  const supportsPointer = 'onpointerdown' in (element ?? {});
  debugLog(
    `[mouseControls] installing handlers pointerSupport=${supportsPointer ? 1 : 0}`,
  );

  let isDragging = false;
  let lastX = 0;
  let lastY = 0;

  const triggerContext = (event, source) => {
    debugLog(
      `[mouseControls] ${source} context button=${event.button ?? 'n/a'} x=${event.clientX} y=${event.clientY}`,
    );
    event.preventDefault?.();
    if (typeof onContext === 'function') {
      onContext({ event });
    }
  };

  const beginDrag = (event, source) => {
    if (typeof event.button === 'number') {
      if (event.button === 2) {
        triggerContext(event, source);
        return;
      }
      if (event.button !== 0) {
        debugLog(
          `[mouseControls] ${source} ignored button=${event.button} x=${event.clientX} y=${event.clientY}`,
        );
        return;
      }
    }

    if (typeof onClick === 'function') {
      debugLog(
        `[mouseControls] ${source} click x=${event.clientX} y=${event.clientY}`,
      );
      onClick({ event, clientX: event.clientX, clientY: event.clientY });
    }

    if (event.button === 0 || event.button === undefined) {
      isDragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      debugLog(
        `[mouseControls] ${source} start id=${event.pointerId ?? 'n/a'} x=${event.clientX} y=${event.clientY}`,
      );
      // In some browsers/environments, calling setPointerCapture with an
      // inactive/undefined pointerId throws NotFoundError. Guard it.
      if (typeof event.pointerId === 'number' && event.pointerId >= 0 && element.setPointerCapture) {
        try { element.setPointerCapture(event.pointerId); } catch (_) { /* ignore */ }
      }
      return;
    }
  };

  const updateDrag = (event, source) => {
    if (!isDragging) {
      return;
    }
    const deltaX = event.clientX - lastX;
    const deltaY = event.clientY - lastY;

    lastX = event.clientX;
    lastY = event.clientY;

    if (typeof onDrag === 'function' && (deltaX || deltaY)) {
      debugLog(
        `[mouseControls] ${source} move id=${event.pointerId ?? 'n/a'} deltaX=${deltaX} deltaY=${deltaY}`,
      );
      onDrag({ deltaX, deltaY, event });
    }
  };

  const finishDrag = (event, source) => {
    if (!isDragging) {
      return;
    }
    isDragging = false;
    debugLog(
      `[mouseControls] ${source} end id=${event.pointerId ?? 'n/a'} x=${event.clientX} y=${event.clientY}`,
    );
    if (typeof event.pointerId === 'number' && event.pointerId >= 0 && element.releasePointerCapture) {
      try { element.releasePointerCapture(event.pointerId); } catch (_) { /* ignore */ }
    }
  };

  const handlePointerDown = (event) => beginDrag(event, 'pointerdown');
  const handlePointerMove = (event) => updateDrag(event, 'pointermove');
  const handlePointerUp = (event) => finishDrag(event, 'pointerup');

  const handleMouseDown = (event) => beginDrag(event, 'mousedown');
  const handleMouseMove = (event) => updateDrag(event, 'mousemove');
  const handleMouseUp = (event) => finishDrag(event, 'mouseup');

  const handleContextMenu = (event) => {
    debugLog('[mouseControls] contextmenu');
    event.preventDefault?.();
  };

  const handleWheel = (event) => {
    debugLog(
      `[mouseControls] wheel deltaY=${event.deltaY} ctrlKey=${event.ctrlKey ? 1 : 0}`,
    );
    if (typeof onScroll === 'function') {
      onScroll({ deltaY: event.deltaY, event });
    }
    event.preventDefault?.();
  };

  if (supportsPointer) {
    element.addEventListener('pointerdown', handlePointerDown);
    target.addEventListener('pointermove', handlePointerMove);
    target.addEventListener('pointerup', handlePointerUp);
    target.addEventListener('pointercancel', handlePointerUp);
  }

  element.addEventListener('mousedown', handleMouseDown);
  target.addEventListener('mousemove', handleMouseMove);
  target.addEventListener('mouseup', handleMouseUp);
  target.addEventListener('mouseleave', handleMouseUp);

  element.addEventListener('wheel', handleWheel, { passive: false });
  element.addEventListener('contextmenu', handleContextMenu);

  return {
    dispose() {
      debugLog('[mouseControls] disposing handlers');
      if (supportsPointer) {
        element.removeEventListener('pointerdown', handlePointerDown);
        target.removeEventListener('pointermove', handlePointerMove);
        target.removeEventListener('pointerup', handlePointerUp);
        target.removeEventListener('pointercancel', handlePointerUp);
      }
      element.removeEventListener('mousedown', handleMouseDown);
      target.removeEventListener('mousemove', handleMouseMove);
      target.removeEventListener('mouseup', handleMouseUp);
      target.removeEventListener('mouseleave', handleMouseUp);
      element.removeEventListener('wheel', handleWheel);
      element.removeEventListener('contextmenu', handleContextMenu);
    },
    handlers: {
      handlePointerDown,
      handlePointerMove,
      handlePointerUp,
      handlePointerCancel: handlePointerUp,
      handleMouseDown,
      handleMouseMove,
      handleMouseUp,
      handleContextMenu,
      handleWheel,
    },
  };
}

export function computeStepVector(angleH, direction) {
  if (direction !== 'forward' && direction !== 'backward') {
    throw new Error(`Unknown direction: ${direction}`);
  }

  const normalizedDirection = direction === 'forward' ? -1 : 1;
  const angleRad = (-angleH * Math.PI) / 180;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);

  const baseX = normalizedDirection;
  const baseY = 0;

  return {
    x: baseX * cos - baseY * sin,
    y: baseX * sin + baseY * cos,
  };
}
