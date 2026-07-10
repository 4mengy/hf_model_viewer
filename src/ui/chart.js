/* ui/chart.js — Tensor-name VRAM horizontal bar chart (Chart.js)
 * Renders Tensor Name Pattern totals plus KV, largest first. */

import Chart from 'chart.js/auto';
import { t } from '../i18n.js';
import { fmtGiBAuto } from './format.js';

const COLORS = {
  kv: '#db2777',
  weight: '#2563eb',
};

const KEY_ROW_HEIGHT = 34;
const AXIS_SPACE_HEIGHT = 56;
const MAX_VISIBLE_KEYS = 10;
const MIN_KEY_COLUMN_WIDTH = 180;
const MAX_KEY_COLUMN_WIDTH = 640;
const APPROX_KEY_CHARACTER_WIDTH = 7;

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

function renderSelectableKeys(canvas, labels) {
  const content = canvas.closest('.overview-chart-content');
  const container = content?.querySelector('.overview-chart-keys');
  if (!container) return { content: canvas.parentElement, container: null, rows: [] };

  const keyWidth = Math.min(
    MAX_KEY_COLUMN_WIDTH,
    Math.max(MIN_KEY_COLUMN_WIDTH, ...labels.map((label) => label.length * APPROX_KEY_CHARACTER_WIDTH)),
  );
  content.style.setProperty('--overview-chart-key-width', `${keyWidth}px`);

  const rows = labels.map((label) => {
    const row = canvas.ownerDocument.createElement('span');
    row.className = 'overview-chart-key';
    row.setAttribute('role', 'listitem');
    row.title = label;
    const input = canvas.ownerDocument.createElement('input');
    input.className = 'overview-chart-key-text';
    input.type = 'text';
    input.readOnly = true;
    input.value = label;
    input.setAttribute('aria-label', label);
    row.append(input);
    return row;
  });
  container.replaceChildren(...rows);
  return { content, container, rows };
}

function selectableKeysPlugin(rows) {
  return {
    id: 'selectableCompositionKeys',
    afterDatasetsDraw(chart) {
      const bars = chart.getDatasetMeta(0).data;
      rows.forEach((row, index) => {
        if (bars[index]) row.style.top = `${bars[index].y}px`;
      });
    },
  };
}

function compositionKeyLabel(item) {
  const key = item.label || t(item.labelKey);
  return item.dtypes?.length ? `${key} · ${item.dtypes.join(' / ')}` : key;
}

export function renderChart(canvas, est) {
  const comp = est.composition || [];
  const labels = comp.map(compositionKeyLabel);
  const values = comp.map((c) => c.gb);
  const colors = comp.map((c) => COLORS[c.colorKey || c.key] || COLORS.weight);
  const valueLabels = comp.map((c) => {
    const percentage = est.complete && est.vTotal > 0 ? ` · ${((c.gb / est.vTotal) * 100).toFixed(1)}%` : '';
    return `${fmtGiBAuto(c.gb)}${percentage}`;
  });

  if (instance) instance.destroy();
  const { content, rows } = renderSelectableKeys(canvas, labels);
  const viewport = content.parentElement;
  content.style.height = '';
  const contentHeight = Math.max(content.clientHeight, comp.length * KEY_ROW_HEIGHT + AXIS_SPACE_HEIGHT);
  content.style.height = `${contentHeight}px`;
  viewport.style.height = `${Math.min(contentHeight, MAX_VISIBLE_KEYS * KEY_ROW_HEIGHT + AXIS_SPACE_HEIGHT)}px`;

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
    plugins: [valueLabelsPlugin, selectableKeysPlugin(rows)],
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
          ticks: { display: false, autoSkip: false },
          grid: { display: false },
          border: { display: false },
        },
        x: {
          beginAtZero: true,
          title: { display: true, text: t('chart.gb') },
        },
      },
    },
  });
}
