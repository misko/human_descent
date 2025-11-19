import HudesClient from './HudesClient.js';
import { log } from '../utils/logger.js';
import { TOUR_FLOWS } from './helpTour.js';

export default class KeyboardClient extends HudesClient {
  constructor(addr, port, options = {}) {
    super(addr, port, options);
    this.altKeys = Boolean(options.altKeys);
    this.pairedKeys = [];
    this.keyToParamAndSign = {};
    this.activeRepeats = new Map();
    this.keyRepeatIntervalMs = options.repeatIntervalMs ?? 100;
    this.keyRepeatDelayMs = options.repeatDelayMs ?? 100;
    this._handleKeyUpOverride = this._handleKeyUpOverride.bind(this);
    this.initInput();


    // Replace base key up handler with extended behaviour.
    window.removeEventListener('keyup', this._handleKeyUp);
    window.addEventListener('keyup', this._handleKeyUpOverride);
  }

  initInput() {
    const useAltLayout = Boolean(this.altKeys);
    if (this.debug) {
      log(`[KeyboardClient] initInput altKeys=${useAltLayout}`);
    }
    this.pairedKeys = useAltLayout
      ? [
          ['u', 'w'],
          ['i', 'e'],
          ['o', 'r'],
          ['j', 's'],
          ['k', 'd'],
          ['l', 'f'],
        ]
      : [
          ['w', 's'],
          ['e', 'd'],
          ['r', 'f'],
          ['u', 'j'],
          ['i', 'k'],
          ['o', 'l'],
        ];

    this.setN(this.pairedKeys.length);

    // Map paired keys to actions
    const firstKeySign = 1;

    this.pairedKeys.forEach(([upKey, downKey], index) => {
      this.addKeyToWatch(upKey);
      this.addKeyToWatch(downKey);

      this.keyToParamAndSign[upKey] = { dim: index, sign: firstKeySign };
      this.keyToParamAndSign[downKey] = { dim: index, sign: -firstKeySign };
    });

    const defaultBatchSize =
      this.renderMode === '1d' ? 512 : this.isMobile ? 32 : 256;
    this.state.setBatchSize(defaultBatchSize);
    this.state.setDtype('float16');

  }

  _handleKeyUpOverride(event) {
    this._stopRepeat(event.key?.toLowerCase?.());
    super._handleKeyUp(event);
  }

  usageStr() {
    return `
Keyboard controller usage:

Hold (q) to quit
Use [ , ] to decrease/increase step size respectively
Tap space to get a new random projection
Enter/Return to get a new training batch

This keyboard controller configuration controls a random ${
      this.state.n
    }-dimensional subspace of the target model.

To control each dimension, use:
${this.pairedKeys
  .map(
    ([upKey, downKey], idx) =>
      `dim${idx + 1}: ${upKey} +, ${downKey} -`
  )
  .join('\n')}
GOOD LUCK!
`;
  }

  processKeyPress(event) {
    const keyLower = event.key?.toLowerCase?.();
    const mapping = keyLower ? this.keyToParamAndSign[keyLower] : undefined;
    const isHelpActive = this.state.helpScreenIdx !== -1;

    if (event.metaKey || event.ctrlKey) {
      return this.processCommonKeys(event);
    }

    if (mapping && !isHelpActive) {
      if (event.type === 'keydown') {
        if (!event.repeat) {
          this._applyDimStep(mapping.dim, mapping.sign);
          this._startDimRepeat(keyLower, mapping.dim, mapping.sign);
        }
        event.preventDefault();
        return true;
      }
      if (event.type === 'keyup') {
        this._stopDimRepeat(keyLower);
        event.preventDefault();
        return true;
      }
    }

    const redraw = this.processCommonKeys(event);

    if (isHelpActive) {
      this.view.showImage(this.state.helpScreenFns[this.state.helpScreenIdx]);
    } else {
      this.view.hideImage();
    }

    return redraw;
  }

  _applyDimStep(dim, sign) {
    if (this.state.helpScreenIdx !== -1) {
      return;
    }
    const delta = this.state.stepSize * sign;
    if (this.debug) {
      const before = Array.isArray(this.dimsAndStepsOnCurrentDims)
        ? [...this.dimsAndStepsOnCurrentDims]
        : null;
      log(
        `[KeyboardClient] applyDimStep dim=${dim} sign=${sign} stepSize=${this.state.stepSize}`,
      );
      if (before) {
        log(`[KeyboardClient] dims before step: ${before.join(', ')}`);
      }
    }
    this.sendDimsAndSteps({ [dim]: delta });
    this._notifyTutorialStep?.('move');
    if (this.debug) {
      const after = Array.isArray(this.dimsAndStepsOnCurrentDims)
        ? [...this.dimsAndStepsOnCurrentDims]
        : null;
      if (after) {
        log(`[KeyboardClient] dims after step: ${after.join(', ')}`);
      }
    }
    this.view.highlightLossLine?.(dim, this.keyRepeatIntervalMs * 2);
  }

  _startDimRepeat(key, dim, sign) {
    this._startRepeat(key, () => this._applyDimStep(dim, sign));
  }

  _startRepeat(key, action) {
    if (!key || typeof action !== 'function') {
      return;
    }
    this._stopRepeat(key);
    const entry = { timeout: null, interval: null };
    const fire = () => {
      if (this.state.helpScreenIdx !== -1) {
        return;
      }
      if (this.debug) {
        log(`[KeyboardClient] repeat fire key=${key}`);
      }
      action();
    };

    if (this.debug) {
      log(
        `[KeyboardClient] start repeat key=${key} delay=${this.keyRepeatDelayMs} interval=${this.keyRepeatIntervalMs}`,
      );
    }
    entry.timeout = setTimeout(() => {
      fire();
      entry.interval = setInterval(fire, this.keyRepeatIntervalMs);
      entry.timeout = null;
      this.activeRepeats.set(key, entry);
    }, this.keyRepeatDelayMs);

    this.activeRepeats.set(key, entry);
  }

  _stopRepeat(key) {
    if (!key) {
      return;
    }
    const entry = this.activeRepeats.get(key);
    if (!entry) {
      return;
    }
    if (this.debug) {
      log(`[KeyboardClient] stop repeat key=${key}`);
    }
    if (entry.timeout) {
      clearTimeout(entry.timeout);
    }
    if (entry.interval) {
      clearInterval(entry.interval);
    }
    this.activeRepeats.delete(key);
  }
}
