import assert from 'node:assert/strict';
import { supportsMultiplayer } from '../mobile.js';

const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
const previousWindow = Object.getOwnPropertyDescriptor(globalThis, 'window');
try {
  const cases = [
    [{ userAgent: 'Mozilla/5.0 (X11; Linux x86_64)', platform: 'Linux x86_64' }, true],
    [{ userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', maxTouchPoints: 10 }, true],
    [{ userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Mobile', platform: 'iPhone' }, false],
    [{ userAgent: 'Mozilla/5.0 (Linux; Android 15)', platform: 'Linux' }, false],
    [{ userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X)', platform: 'MacIntel', maxTouchPoints: 5 }, false],
    [{ userAgent: 'Desktop UA', userAgentData: { mobile: true } }, false],
  ];
  // A narrow viewport and the solo layout override do not redefine a device.
  Object.defineProperty(globalThis, 'window', { configurable: true, value: { innerWidth: 390, location: { search: '?mobile=0' } } });
  for (const [navigator, expected] of cases) {
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: navigator });
    assert.equal(supportsMultiplayer(), expected, navigator.userAgent);
  }
} finally {
  if (previousNavigator) Object.defineProperty(globalThis, 'navigator', previousNavigator);
  else delete globalThis.navigator;
  if (previousWindow) Object.defineProperty(globalThis, 'window', previousWindow);
  else delete globalThis.window;
}
