/* ui/format.js — Display formatting helpers */

const BYTES_PER_GIB = 1024 ** 3;

export function fmtNum(n) {
  if (!Number.isFinite(n)) return '—';
  return Math.round(n).toLocaleString('en-US');
}

export function fmtGB(gb) {
  if (!Number.isFinite(gb)) return '—';
  return `${gb.toFixed(2)} GB`;
}

export function fmtBytesGB(bytes) {
  return fmtGB(bytes / BYTES_PER_GIB);
}

export function fmtBytesAuto(bytes) {
  if (!Number.isFinite(bytes)) return '—';

  const units = ['B', 'KB', 'MB', 'GB'];
  const absolute = Math.abs(bytes);
  let unitIndex = 0;
  let scale = 1;

  while (absolute >= scale * 1024 && unitIndex < units.length - 1) {
    scale *= 1024;
    unitIndex += 1;
  }

  const value = bytes / scale;
  return unitIndex === 0
    ? `${Math.round(value)} ${units[unitIndex]}`
    : `${value.toFixed(2)} ${units[unitIndex]}`;
}

export function fmtGiBAuto(gib) {
  return fmtBytesAuto(gib * BYTES_PER_GIB);
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
