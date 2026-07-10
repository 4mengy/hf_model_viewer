/* ui/chart.js — VRAM composition horizontal bar chart (Chart.js)
 * Renders est.composition: fine-grained breakdown by tensor category
 * (dense weight modules / MoE experts / KV / overhead). */

import Chart from 'chart.js/auto';
import { t } from '../i18n.js';
import { fmtGiBAuto } from './format.js';

// Per-category colors; reused by the overview data list in app.js
// so the chart and the text stay visually consistent.
export const COLORS = {
  embedding: '#2563eb',
  attn: '#4f46e5',
  mlp: '#0891b2',
  norm: '#0d9488',
  other: '#64748b',
  lmhead: '#d97706',
  expert: '#7c3aed',
  sharedExpert: '#c026d3',
  kv: '#db2777',
  overhead: '#94a3b8',
  weight: '#2563eb',
};

let instance = null;

export function renderChart(canvas, est) {
  const comp = est.composition || [];
  const labels = comp.map((c) => t(c.labelKey));
  const values = comp.map((c) => c.gb);
  const colors = comp.map((c) => COLORS[c.key] || '#94a3b8');

  if (instance) instance.destroy();

  instance = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: t('chart.vramLabel'),
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${fmtGiBAuto(ctx.parsed.x)}`,
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          title: { display: true, text: t('chart.gb') },
        },
      },
    },
  });
}
