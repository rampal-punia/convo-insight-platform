import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.js'],
    include: ['**/__tests__/**/*.{test,spec}.{js,jsx}'],
    css: false,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname),
    },
  },
});
