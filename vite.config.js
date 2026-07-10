import { defineConfig } from 'vite';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdirSync, writeFileSync } from 'node:fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: resolve(__dirname, 'src/web'),
  base: './',
  build: {
    outDir: resolve(__dirname, 'dist-web'),
    emptyOutDir: true,
  },
  plugins: [
    {
      name: 'nojekyll',
      closeBundle() {
        const out = resolve(__dirname, 'dist-web');
        mkdirSync(out, { recursive: true });
        writeFileSync(resolve(out, '.nojekyll'), '');
      },
    },
  ],
});
