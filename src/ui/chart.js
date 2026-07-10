/* ui/chart.js — Tensor-name VRAM horizontal bar chart (Chart.js)
 * Renders Tensor Name Pattern totals plus KV / overhead, largest first. */

import Chart from 'chart.js/auto';
import { t } from '../i18n.js';
import { fmtGiBAuto } from './format.js';

const COLORS = {
  kv: '#db2777',
  overhead: '#94a3b8',
  weight: '#2563eb',
};

let instance = null;

const valueLabelsPlugin = {
  id: 'compositionValueLabels',
  afterDatasetsDraw(chart) {
    const dataset = chart.data.datasets[0];
    const bars = chart.getDatasetMeta(0).data;
    const { ctx } = chart;
    ctx.save();
    ctx.fillStyle = '#64748b';
    ctx.font = '11px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    bars.forEach((bar, index) => {
      ctx.fillText(dataset.valueLabels[index], bar.x + 8, bar.y);
    });
    ctx.restore();
  },
};

export function renderChart(canvas, est) {
  const comp = est.composition || [];
  const labels = comp.map((c) => c.label || t(c.labelKey));
  const values = comp.map((c) => c.gb);
  const colors = comp.map((c) => COLORS[c.colorKey || c.key] || COLORS.weight);
  const valueLabels = comp.map((c) => {
    const percentage = est.complete && est.vTotal > 0 ? ` · ${((c.gb / est.vTotal) * 100).toFixed(1)}%` : '';
    return `${fmtGiBAuto(c.gb)}${percentage}`;
  });

  if (instance) instance.destroy();
  canvas.parentElement.style.height = `${Math.max(320, comp.length * 34 + 56)}px`;

  instance = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: t('chart.vramLabel'),
          data: values,
          valueLabels,
          backgroundColor: colors,
          borderRadius: 6,
        },
      ],
    },
    plugins: [valueLabelsPlugin],
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { right: 116 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${fmtGiBAuto(ctx.parsed.x)}`,
          },
        },
      },
      scales: {
        y: {
          ticks: { autoSkip: false },
        },
        x: {
          beginAtZero: true,
          title: { display: true, text: t('chart.gb') },
        },
      },
    },
  });
}
