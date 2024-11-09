// import HudesClient from './client/HudesClient.js';

// const client = new HudesClient("localhost", 8765);
// client.runLoop();

import KeyboardClient from './client/KeyboardClient.js';

// Initialize the client
const client = new KeyboardClient('localhost', 8765);
client.runLoop();