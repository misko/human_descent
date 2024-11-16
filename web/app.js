// import HudesClient from './client/HudesClient.js';

// const client = new HudesClient("localhost", 8765);
// client.runLoop();

//import KeyboardClient from './client/KeyboardClient.js';
import KeyboardClientGL from './client/KeyboardClientGL.js';

// Initialize the client
//const client = new KeyboardClient('localhost', 8765);
//const client = new KeyboardClientGL('localhost', 8765);
const client = new KeyboardClientGL('humandescent.net', 10000);


client.runLoop();