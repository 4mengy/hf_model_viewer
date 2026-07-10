/* ui/format.js — Display formatting helpers */

export function fmtNum(n) {
  if (!Number.isFinite(n)) return '—';
  return Math.round(n).toLocaleString('en-US');
}

export function fmtGB(gb) {
  if (!Number.isFinite(gb)) return '—';
  return `${gb.toFixed(2)} GB`;
}

export function fmtBytesGB(bytes) {
  return fmtGB(bytes / (1024 ** 3));
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
