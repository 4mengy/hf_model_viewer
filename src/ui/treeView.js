/* ui/treeView.js — 全量结构树渲染（<details> 折叠树 + <table> 算子表）
 * ------------------------------------------------------------
 * 显存列按「逐张量有效字节」渲染（与计算器一致）：
 *   - 每个张量单元格带 data-key="t:<name>"
 *   - 聚合单元格（层/块/专家/MoE）带各自 data-key
 * 精度/量化策略切换时仅调用 updateTreeBytes 刷新文本，不重建 DOM，
 * 从而保留用户已展开的折叠状态。data-key 与 buildByteMap 一一对应。
 * ------------------------------------------------------------ */

import { fmtNum, fmtGB, esc } from './format.js';
import { tensorParams } from '../tree/buildTree.js';

function pCount(t) {
  return t.params != null ? t.params : tensorParams(t.shape);
}
function effBpp(name, map) {
  return map.get(name) ?? 2;
}
function tensorByte(t, map) {
  return pCount(t) * effBpp(t.name, map);
}

/** 由树 + 有效字节映射，重算所有聚合显存（与渲染时同一套规则） */
function buildByteMap(tree, map) {
  const m = new Map();
  for (const grp of tree.nonLayer) {
    m.set('n:' + grp.group, grp.tensors.reduce((s, t) => s + tensorByte(t, map), 0));
  }
  tree.layers.forEach((layer, i) => {
    if (!layer) return;
    let layerBytes = 0;
    const acc = (list) => {
      layerBytes += (list || []).reduce((s, t) => s + tensorByte(t, map), 0);
    };
    acc(layer.attn);
    acc(layer.mlp);
    acc(layer.norm);
    acc(layer.other);
    if (layer.experts) {
      const rep = layer.experts.representative || [];
      const eb = rep.reduce((s, t) => s + tensorByte(t, map), 0) * layer.experts.count;
      layerBytes += eb;
      m.set('e:' + i, eb);
    }
    m.set('l:' + i, layerBytes);
  });
  return m;
}

function tensorTable(tensors, map) {
  const rows = tensors
    .map(
      (t) => `
      <tr>
        <td><code>${esc(t.name)}</code></td>
        <td>${esc(t.shape.join('×'))}</td>
        <td>${esc(t.dtype)}</td>
        <td class="num">${fmtNum(pCount(t))}</td>
        <td class="num byte-cell" data-key="t:${esc(t.name)}">${fmtGB(tensorByte(t, map))}</td>
      </tr>`,
    )
    .join('');
  return `<table class="ops"><thead><tr><th>算子</th><th>Shape</th><th>DType</th><th class="num">参数量</th><th class="num">显存</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function blockHTML(title, tensors, map, key) {
  if (!tensors || !tensors.length) return '';
  const total = tensors.reduce((a, t) => a + pCount(t), 0);
  const bytes = tensors.reduce((a, t) => a + tensorByte(t, map), 0);
  return `<details class="block"><summary>${esc(title)} <span class="meta">${fmtNum(total)} 参数 · <span class="byte-cell" data-key="${key}">${fmtGB(bytes)}</span></span></summary>${tensorTable(tensors, map)}</details>`;
}

function expertsHTML(layer, map, i) {
  const e = layer.experts;
  const rep = e.representative || [];
  const repTotal = rep.reduce((a, t) => a + pCount(t), 0);
  const totalBytes = rep.reduce((a, t) => a + tensorByte(t, map), 0) * e.count;
  return `
    <details class="block">
      <summary>MoE Experts <span class="tag moe">×${e.count}</span>
        <span class="meta">${fmtNum(repTotal * e.count)} 参数 · <span class="byte-cell" data-key="e:${i}">${fmtGB(totalBytes)}</span></span>
      </summary>
      <p class="note">每专家同构，以下展示单一代表专家（参数量 ×${e.count} = 合计）。</p>
      ${tensorTable(rep, map)}
    </details>`;
}

function layerHTML(i, layer, map) {
  let layerBytes = 0;
  const acc = (list) => {
    layerBytes += (list || []).reduce((s, t) => s + tensorByte(t, map), 0);
  };
  acc(layer.attn);
  acc(layer.mlp);
  acc(layer.norm);
  acc(layer.other);
  if (layer.experts) {
    const rep = layer.experts.representative || [];
    layerBytes += rep.reduce((s, t) => s + tensorByte(t, map), 0) * layer.experts.count;
  }
  const meta = `<span class="meta"><b>${fmtNum(layer.layerParams)}</b> 参数 · <span class="byte-cell" data-key="l:${i}">${fmtGB(layerBytes)}</span></span>`;
  const inner =
    blockHTML('Self-Attention', layer.attn, map, `l:${i}:attn`) +
    blockHTML('MLP (稠密)', layer.mlp, map, `l:${i}:mlp`) +
    (layer.experts ? expertsHTML(layer, map, i) : '') +
    blockHTML('Norm', layer.norm, map, `l:${i}:norm`) +
    blockHTML('Other', layer.other, map, `l:${i}:other`);
  return `<details class="layer"><summary>Layer ${i} ${meta}</summary>${inner}</details>`;
}

export function renderTree(container, tree, map) {
  const parts = [];
  for (const grp of tree.nonLayer) {
    const title =
      grp.group === 'Embedding' ? 'Embedding' : grp.group === 'LM Head' ? 'LM Head' : grp.group;
    parts.push(blockHTML(title, grp.tensors, map, 'n:' + grp.group));
  }
  tree.layers.forEach((layer, i) => {
    if (!layer) return;
    parts.push(layerHTML(i, layer, map));
  });
  container.innerHTML = parts.join('');
}

/** 精度/量化策略切换时，仅刷新所有显存文本（按最新有效字节映射） */
export function updateTreeBytes(container, tree, map) {
  const bm = buildByteMap(tree, map);
  container.querySelectorAll('.byte-cell').forEach((el) => {
    const k = el.getAttribute('data-key');
    if (bm.has(k)) el.textContent = fmtGB(bm.get(k));
  });
}
