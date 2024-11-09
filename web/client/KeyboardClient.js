import HudesClient from './HudesClient.js';
import { log } from '../utils/logger.js';

export default class KeyboardClient extends HudesClient {
  constructor(addr, port) {
    super(addr, port);
    this.pairedKeys = [];
    this.keyToParamAndSign = {};
    this.initInput();
  }

  initInput() {
    this.pairedKeys = [
      ['w', 's'],
      ['e', 'd'],
      ['r', 'f'],
      ['u', 'j'],
      ['i', 'k'],
      ['o', 'l'],
    ];

    this.setN(this.pairedKeys.length);

    // Map paired keys to actions
    this.pairedKeys.forEach(([upKey, downKey], index) => {
      this.addKeyToWatch(upKey);
      this.addKeyToWatch(downKey);

      this.keyToParamAndSign[upKey] = { dim: index, sign: 1 };
      this.keyToParamAndSign[downKey] = { dim: index, sign: -1 };
    });

    this.state.setBatchSize(512);
    this.state.setDtype('float32');

    this.state.setHelpScreenFns([
      'help_screens/hudes_help_start.png',
      'help_screens/hudes_1.png',
      'help_screens/hudes_2.png',
      'help_screens/hudes_3.png',
      'help_screens/hudes_1d_keyboard_controls.png',
    ]);
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
    const currentTime = performance.now(); // High-precision timing
    let redraw = this.processCommonKeys(event);

    // if (this.state.helpScreenIdx !== -1) {
    //   return redraw; // Skip further processing if help screen is active
    // }

    if (event.type === 'keydown') {
      const key = event.key.toLowerCase(); // Normalize to lowercase

        log(`ClientState: process press down!`);
      if (this.keyToParamAndSign[key]) {
        const { dim, sign } = this.keyToParamAndSign[key];
        if (
          this.checkCooldownTime(event, key, currentTime, this.cooldownTimeMs / 1000)
        ) {
          this.sendDimsAndSteps({ [dim]: this.state.stepSize * sign });
        }
      }
    }

    return redraw;
  }
}
