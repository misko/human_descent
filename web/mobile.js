const FALLBACK_MOBILE_WIDTH = 768;

const isPointerCoarse = () => {
  if (typeof window === 'undefined') return false;
  try {
    if (window.matchMedia('(pointer: coarse)').matches) return true;
  } catch {}
  return false;
};

const hasTouchPoints = () => {
  if (typeof navigator === 'undefined') return false;
  return Number(navigator.maxTouchPoints || 0) > 0;
};

const smallViewport = () => {
  if (typeof window === 'undefined') return false;
  return window.innerWidth <= FALLBACK_MOBILE_WIDTH;
};

export const detectMobileMode = (params = new URLSearchParams()) => {
  const qp = params.get('mobile');
  if (qp != null) {
    return /^(1|true|on|yes)$/i.test(qp);
  }
  return isPointerCoarse() || hasTouchPoints() || smallViewport();
};

// Multiplayer requires desktop input. Layout overrides and viewport width do
// not change device eligibility (a narrow desktop window remains a desktop).
export const supportsMultiplayer = () => {
  if (typeof navigator === 'undefined') return true;
  if (navigator.userAgentData?.mobile === true) return false;
  if (/Android|iPhone|iPad|iPod|Mobile|IEMobile|Windows Phone/i.test(navigator.userAgent || '')) return false;
  // iPadOS can identify itself as a Mac while retaining multiple touch points.
  if (navigator.platform === 'MacIntel' && Number(navigator.maxTouchPoints || 0) > 1) return false;
  return true;
};

export const setMobileFlag = (value) => {
  const enabled = Boolean(value);
  if (typeof window !== 'undefined') {
    window.__hudesIsMobile = enabled;
  }
  if (typeof document !== 'undefined') {
    document.documentElement?.classList?.toggle('hudes-mobile', enabled);
    document.body?.classList?.toggle('hudes-mobile', enabled);
  }
};
