const LEVELS = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3,
};

const envLevel =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_LOG_LEVEL) ||
  (typeof process !== 'undefined' && process.env?.VITE_LOG_LEVEL) ||
  'info';
const ACTIVE_LEVEL = LEVELS[envLevel.toLowerCase()] ?? LEVELS.info;

function formatMessage(scope, level, message) {
  const parts = [];
  if (scope) parts.push(`[${scope}]`);
  parts.push(level.toUpperCase());
  parts.push(message);
  return parts.join(' ');
}

export function createLogger(scope = 'app') {
  const write = (level, message, context) => {
    const lvlIdx = LEVELS[level] ?? LEVELS.info;
    if (lvlIdx > ACTIVE_LEVEL) return;
    const formatted = formatMessage(scope, level, message);
    const consoleFn =
      level === 'error'
        ? console.error
        : level === 'warn'
          ? console.warn
          : console.log;

    if (context && typeof context === 'object') {
      consoleFn(formatted, context);
    } else if (context != null) {
      consoleFn(`${formatted} ${String(context)}`);
    } else {
      consoleFn(formatted);
    }
  };

  return {
    debug(message, context) {
      write('debug', message, context);
    },
    info(message, context) {
      write('info', message, context);
    },
    warn(message, context) {
      write('warn', message, context);
    },
    error(message, context) {
      write('error', message, context);
    },
  };
}

const defaultLogger = createLogger('hudes');

export function log(message, context = null, level = 'info') {
  const upper = typeof level === 'string' ? level.toLowerCase() : 'info';
  const loggerFn = defaultLogger[upper] || defaultLogger.info;
  loggerFn(message, context);
}
