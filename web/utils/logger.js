const hasDocument = typeof document !== 'undefined';

export function log(message) {
  console.log(message);
  if (!hasDocument) {
    return;
  }
  const logElem = document.getElementById('log');
  if (logElem) {
    logElem.textContent += `${message}\n`;
  }
}
