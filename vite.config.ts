import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig(({ mode }) => ({
  plugins: [react(), ...(mode === 'artifact' ? [viteSingleFile()] : [])],
  server: { proxy: { '/api/investigate': { target: 'http://127.0.0.1:8788', changeOrigin: false } } },
  build: { outDir: mode === 'artifact' ? 'dist-artifact' : 'dist' },
}));
