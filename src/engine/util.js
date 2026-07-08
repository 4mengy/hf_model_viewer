/* engine/util.js — 并发受限工具 */

/**
 * 以固定并发上限遍历 items 执行 fn（避免 160+ 分片同时发起请求打爆浏览器）。
 */
export async function mapLimit(items, limit, fn) {
  let i = 0;
  const n = Math.max(1, Math.min(limit, items.length));
  const workers = Array.from({ length: n }, async () => {
    while (i < items.length) {
      const idx = i++;
      await fn(items[idx], idx);
    }
  });
  await Promise.all(workers);
}
