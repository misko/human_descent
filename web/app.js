// import HudesClient from './client/HudesClient.js';

// const client = new HudesClient("localhost", 8765);
// client.runLoop();

//import KeyboardClient from './client/KeyboardClient.js';
import KeyboardClientGL from './client/KeyboardClientGL.js';

// Initialize the client
//const client = new KeyboardClient('localhost', 8765);
// Default: connect to backend on port 8765 at current host
// Overridable via query params ?host=&port=
const defaultHost = (typeof window !== 'undefined' && window.location?.hostname) || 'localhost';
const client = new KeyboardClientGL(defaultHost, 8765);
// Expose for tests/console
if (typeof window !== 'undefined') {
	window.__hudesClient = client;
}

client.runLoop();
