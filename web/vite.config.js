import { defineConfig } from 'vite';
import { visualizer } from 'rollup-plugin-visualizer';
import path from 'node:path';

const devWsPort = Number(
  process.env.HUDES_PORT ||
    process.env.VITE_DEV_WS_PORT ||
    process.env.VITE_WS_PORT ||
    10001,
);
const devHttpPort = Number(
  process.env.HUDES_HEALTH_PORT ||
    process.env.VITE_DEV_HTTP_PORT ||
    process.env.VITE_HTTP_PORT ||
    devWsPort + 1,
);

// Small helper to rewrite clean URLs like /highscores -> /highscores.html in dev
const cleanUrlRewrite = {
  name: 'rewrite-clean-urls',
  configureServer(server) {
    server.middlewares.use((req, _res, next) => {
      if (!req.url || req.method !== 'GET') return next();
      if (req.url === '/highscores' || req.url === '/highscores/') {
        req.url = '/highscores.html';
      }
      next();
    });
  },
  configurePreviewServer(server) {
    server.middlewares.use((req, _res, next) => {
      if (!req.url || req.method !== 'GET') return next();
      if (req.url === '/highscores' || req.url === '/highscores/') {
        req.url = '/highscores.html';
      }
      next();
    });
  },
};

export default defineConfig({
  // Treat as a multi-page app so both index.html and highscores.html are first-class pages
  appType: 'mpa',
  plugins: [
    cleanUrlRewrite,
    visualizer({ open: true, filename: 'bundle-visualization.html' }),
  ],
  server: {
    proxy: {
      '/ws': {
        target: `http://localhost:${devWsPort}`,
        ws: true,
        changeOrigin: true,
      },
      '/api': {
        target: `http://localhost:${devHttpPort}`,
        changeOrigin: true,
      },
      '/health': {
        target: `http://localhost:${devHttpPort}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        highscores: path.resolve(__dirname, 'highscores.html'),
      },
    },
  },
});
