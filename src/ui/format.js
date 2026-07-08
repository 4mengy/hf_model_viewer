/* ui/format.js — 展示格式化工具 */

export function fmtNum(n) {
  if (!Number.isFinite(n)) return '—';
  return Math.round(n).toLocaleString('en-US');
}

export function fmtGB(gb) {
  if (!Number.isFinite(gb)) return '—';
  return `${gb.toFixed(2)} GB`;
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}
