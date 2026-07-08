import { defineConfig } from 'vite';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const target = process.env.BUILD_TARGET || 'web';

// 双形态构建管线：
//   BUILD_TARGET=web  -> 输出 GitHub Pages 静态站点 (dist-web/)，入口在根
//   BUILD_TARGET=ext  -> 输出浏览器扩展包 (dist-ext/)：manifest + popup + background，均在根
// 通过为两种形态分别设置 root 与绝对入口路径，使 HTML/JS 直接落在产物根目录，
// 从而与 manifest.json 的 default_popup / service_worker 相对路径一致。
const root = target === 'web' ? resolve(__dirname, 'src/web') : resolve(__dirname, 'src/ext');
const entry = target === 'web'
  ? resolve(root, 'index.html')
  : { popup: resolve(root, 'popup.html'), background: resolve(root, 'background.js') };

export default defineConfig({
  root,
  base: target === 'web' ? './' : '',
  build: {
    outDir: target === 'web' ? resolve(__dirname, 'dist-web') : resolve(__dirname, 'dist-ext'),
    emptyOutDir: true,
    rollupOptions: {
      input: entry,
      output:
        target === 'ext'
          ? { entryFileNames: '[name].js', chunkFileNames: '[name].js', assetFileNames: '[name].[ext]' }
          : undefined,
    },
  },
  define: {
    // 注入构建目标，供运行时做形态判断（如扩展态路由网络请求到 background）
    'import.meta.env.BUILD_TARGET': JSON.stringify(target),
  },
  plugins: [
    {
      name: 'copy-static-assets',
      closeBundle() {
        if (target === 'ext') {
          const out = resolve(__dirname, 'dist-ext');
          mkdirSync(out, { recursive: true });
          copyFileSync(resolve(__dirname, 'src/ext/manifest.json'), resolve(out, 'manifest.json'));
        } else {
          // GitHub Pages 默认会用 Jekyll 处理静态文件，可能破坏带下划线前缀的资源；
          // 写入 .nojekyll 关闭 Jekyll。base 已设为 './' 以兼容项目子路径 (/<repo>/)。
          const out = resolve(__dirname, 'dist-web');
          mkdirSync(out, { recursive: true });
          writeFileSync(resolve(out, '.nojekyll'), '');
        }
      },
    },
  ],
});
