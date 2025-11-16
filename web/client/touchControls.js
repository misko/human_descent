const DEFAULT_INTERVAL = 90;
const DEFAULT_DEADZONE = 12;

export function installTouchControls(target, { onVector, intervalMs = DEFAULT_INTERVAL, deadzonePx = DEFAULT_DEADZONE } = {}) {
  if (!target || typeof window === 'undefined') {
    return () => {};
  }

  let activeId = null;
  let origin = null;
  let lastVector = { x: 0, y: 0 };
  let loop = null;

  const emit = () => {
    if (typeof onVector === 'function') {
      onVector({ ...lastVector });
    }
  };

  const stopLoop = () => {
    if (loop) {
      clearInterval(loop);
      loop = null;
    }
  };

  const startLoop = () => {
    if (!loop) {
      loop = setInterval(emit, Math.max(30, intervalMs));
    }
  };

  const maxRadius = () => {
    const w = window.innerWidth || 0;
    const h = window.innerHeight || 0;
    return Math.max(deadzonePx * 4, Math.min(w, h) * 0.4);
  };

  const computeVector = (touch) => {
    if (!origin) return { x: 0, y: 0 };
    const dx = touch.clientX - origin.x;
    const dy = touch.clientY - origin.y;
    const distance = Math.hypot(dx, dy);
    if (distance <= deadzonePx) {
      return { x: 0, y: 0 };
    }
    const radius = maxRadius();
    const magnitude = Math.min(1, distance / radius);
    const unitX = dx / (distance || 1);
    const unitY = dy / (distance || 1);
    return { x: unitX * magnitude, y: unitY * magnitude };
  };

  const findActiveTouch = (touchList) => {
    if (activeId == null) return null;
    for (let i = 0; i < touchList.length; i += 1) {
      if (touchList[i].identifier === activeId) {
        return touchList[i];
      }
    }
    return null;
  };

  const clearState = () => {
    activeId = null;
    origin = null;
    lastVector = { x: 0, y: 0 };
    stopLoop();
    emit();
  };

  const handleStart = (event) => {
    const touchCount = event.touches?.length || event.changedTouches?.length || 0;
    if (touchCount > 1) {
      clearState();
      return;
    }
    if (activeId != null) return;
    const touch = event.changedTouches?.[0];
    if (!touch) return;
    activeId = touch.identifier;
    origin = { x: touch.clientX, y: touch.clientY };
    lastVector = { x: 0, y: 0 };
    startLoop();
    event.preventDefault?.();
  };

  const handleMove = (event) => {
    const touchCount = event.touches?.length || 0;
    if (touchCount > 1) {
      clearState();
      return;
    }
    const touch = findActiveTouch(event.touches || []);
    if (!touch) return;
    lastVector = computeVector(touch);
    event.preventDefault?.();
  };

  const handleEnd = (event) => {
    const touch = findActiveTouch(event.changedTouches || []);
    if (!touch) return;
    event.preventDefault?.();
    clearState();
  };

  const handleCancel = (event) => {
    const touch = findActiveTouch(event.changedTouches || []);
    if (!touch) return;
    event.preventDefault?.();
    clearState();
  };

  target.addEventListener('touchstart', handleStart, { passive: false });
  target.addEventListener('touchmove', handleMove, { passive: false });
  target.addEventListener('touchend', handleEnd, { passive: false });
  target.addEventListener('touchcancel', handleCancel, { passive: false });

  return () => {
    stopLoop();
    target.removeEventListener('touchstart', handleStart);
    target.removeEventListener('touchmove', handleMove);
    target.removeEventListener('touchend', handleEnd);
    target.removeEventListener('touchcancel', handleCancel);
  };
}
