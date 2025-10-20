import { defineConfig } from 'vite';
import { visualizer } from 'rollup-plugin-visualizer';
import path from 'node:path';

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
  build: {
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        highscores: path.resolve(__dirname, 'highscores.html'),
      },
    },
  },
});
