export function log(message) {
    console.log(message);
    const logElem = document.getElementById("log");
    if (logElem) {
      logElem.textContent += `${message}\n`;
    }
  }
