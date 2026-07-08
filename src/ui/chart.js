/* ui/chart.js — 显存占比水平条形图（Chart.js）
 * 渲染 est.composition：按张量类别拆分的细粒度组成（权重各模块 / MoE 专家 / KV / 开销）。 */

import Chart from 'chart.js/auto';

// 各分类配色（app.js 的「组成明细」列表复用，保证图表与文字一致）
export const COLORS = {
  embedding: '#2563eb',
  attn: '#4f46e5',
  mlp: '#0891b2',
  norm: '#0d9488',
  other: '#64748b',
  lmhead: '#d97706',
  expert: '#7c3aed',
  kv: '#db2777',
  overhead: '#94a3b8',
  weight: '#2563eb',
};

let instance = null;

export function renderChart(canvas, est) {
  const comp = est.composition || [];
  const labels = comp.map((c) => c.label);
  const values = comp.map((c) => c.gb);
  const colors = comp.map((c) => COLORS[c.key] || '#94a3b8');

  if (instance) instance.destroy();

  instance = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: '显存占用 (GB)',
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
            label: (ctx) => ` ${ctx.parsed.x.toFixed(3)} GB`,
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          title: { display: true, text: 'GB' },
        },
      },
    },
  });
}
