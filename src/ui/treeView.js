/* ui/treeView.js — Full structure tree (collapsible <details> + <table> ops)
 * ------------------------------------------------------------
 * The VRAM column is rendered from the "per-tensor effective bytes" map,
 * matching the calculator:
 *   - each tensor cell carries data-key="t:<name>"
 *   - aggregate cells (layer / block / expert / MoE) carry their own data-key
 * On precision / quantization change we only call updateTreeBytes to refresh
 * the text (not rebuild the DOM), preserving the user's expanded state.
 * data-keys line up 1:1 with buildByteMap.
 * ------------------------------------------------------------ */

import { fmtNum, fmtBytesGB, esc } from './format.js';
import { tensorParams } from '../tree/buildTree.js';
import { t } from '../i18n.js';

function pCount(tensor) {
  return tensor.params != null ? tensor.params : tensorParams(tensor.shape);
}
function effBpp(name, map) {
  return map.get(name) ?? 2;
}
function tensorByte(tensor, map) {
  return pCount(tensor) * effBpp(tensor.name, map);
}

/** Recompute all aggregate VRAM from the tree + effective-byte map
 *  (same rules as at render time). */
function buildByteMap(tree, map) {
  const m = new Map();
  const sumTensors = (list = []) => list.reduce((sum, tensor) => {
    const bytes = tensorByte(tensor, map);
    m.set('t:' + tensor.name, bytes);
    return sum + bytes;
  }, 0);
  for (const grp of tree.nonLayer) {
    m.set('n:' + grp.group, sumTensors(grp.tensors));
  }
  tree.layers.forEach((layer, i) => {
    if (!layer) return;
    let layerBytes = 0;
    const acc = (list, key) => {
      const bytes = sumTensors(list);
      layerBytes += bytes;
      m.set(key, bytes);
    };
    acc(layer.attn, `l:${i}:attn`);
    acc(layer.mlp, `l:${i}:mlp`);
    acc(layer.norm, `l:${i}:norm`);
    acc(layer.other, `l:${i}:other`);
    if (layer.experts) {
      const rep = layer.experts.representative || [];
      const eb = sumTensors(rep) * layer.experts.count;
      layerBytes += eb;
      m.set('e:' + i, eb);
    }
    if (layer.sharedExperts) {
      const sb = sumTensors(layer.sharedExperts.tensors);
      layerBytes += sb;
      m.set('se:' + i, sb);
    }
    m.set('l:' + i, layerBytes);
  });
  return m;
}

function tensorTable(tensors, map) {
  const rows = tensors
    .map(
      (tn) => `
      <tr>
        <td><code>${esc(tn.name)}</code></td>
        <td>${esc(tn.shape.join('×'))}</td>
        <td>${esc(tn.dtype)}</td>
        <td class="num">${fmtNum(pCount(tn))}</td>
        <td class="num byte-cell" data-key="t:${esc(tn.name)}">${fmtBytesGB(tensorByte(tn, map))}</td>
      </tr>`,
    )
    .join('');
  return `<table class="ops"><thead><tr><th>${esc(t('tree.col.op'))}</th><th>${esc(t('tree.col.shape'))}</th><th>${esc(t('tree.col.dtype'))}</th><th class="num">${esc(t('tree.col.params'))}</th><th class="num">${esc(t('tree.col.vram'))}</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function blockHTML(title, tensors, map, key) {
  if (!tensors || !tensors.length) return '';
  const total = tensors.reduce((a, tn) => a + pCount(tn), 0);
  const bytes = tensors.reduce((a, tn) => a + tensorByte(tn, map), 0);
  return `<details class="block"><summary>${esc(title)} <span class="meta">${fmtNum(total)} ${t('tree.paramsUnit')} · <span class="byte-cell" data-key="${key}">${fmtBytesGB(bytes)}</span></span></summary>${tensorTable(tensors, map)}</details>`;
}

function expertsHTML(layer, map, i) {
  const e = layer.experts;
  const rep = e.representative || [];
  const repTotal = rep.reduce((a, tn) => a + pCount(tn), 0);
  const totalBytes = rep.reduce((a, tn) => a + tensorByte(tn, map), 0) * e.count;
  return `
    <details class="block">
      <summary>${esc(t('tree.moeExperts'))} <span class="tag moe">×${e.count}</span>
        <span class="meta">${fmtNum(repTotal * e.count)} ${t('tree.paramsUnit')} · <span class="byte-cell" data-key="e:${i}">${fmtBytesGB(totalBytes)}</span></span>
      </summary>
      <p class="note">${esc(t('tree.expertNote', { n: e.count }))}</p>
      ${tensorTable(rep, map)}
    </details>`;
}

function sharedExpertsHTML(layer, map, i) {
  const se = layer.sharedExperts;
  if (!se) return '';
  const total = se.params;
  const bytes = se.tensors.reduce((a, tn) => a + tensorByte(tn, map), 0);
  return `
    <details class="block">
      <summary>${esc(t('tree.sharedExpert'))} <span class="tag shared">1</span>
        <span class="meta">${fmtNum(total)} ${t('tree.paramsUnit')} · <span class="byte-cell" data-key="se:${i}">${fmtBytesGB(bytes)}</span></span>
      </summary>
      <p class="note">${esc(t('tree.sharedExpertNote'))}</p>
      ${tensorTable(se.tensors, map)}
    </details>`;
}

function layerHTML(i, layer, map) {
  let layerBytes = 0;
  const acc = (list) => {
    layerBytes += (list || []).reduce((s, tn) => s + tensorByte(tn, map), 0);
  };
  acc(layer.attn);
  acc(layer.mlp);
  acc(layer.norm);
  acc(layer.other);
  if (layer.experts) {
    const rep = layer.experts.representative || [];
    layerBytes += rep.reduce((s, tn) => s + tensorByte(tn, map), 0) * layer.experts.count;
  }
  if (layer.sharedExperts) {
    layerBytes += layer.sharedExperts.tensors.reduce((s, tn) => s + tensorByte(tn, map), 0);
  }
  const meta = `<span class="meta"><b>${fmtNum(layer.layerParams)}</b> ${t('tree.paramsUnit')} · <span class="byte-cell" data-key="l:${i}">${fmtBytesGB(layerBytes)}</span></span>`;
  const inner =
    blockHTML('Self-Attention', layer.attn, map, `l:${i}:attn`) +
    (layer.experts ? expertsHTML(layer, map, i) : '') +
    (layer.sharedExperts ? sharedExpertsHTML(layer, map, i) : '') +
    blockHTML(t('tree.mlpDense'), layer.mlp, map, `l:${i}:mlp`) +
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

/** On precision / quantization change, only refresh the VRAM text
 *  (using the latest effective-byte map). */
export function updateTreeBytes(container, tree, map) {
  const bm = buildByteMap(tree, map);
  container.querySelectorAll('.byte-cell').forEach((el) => {
    const k = el.getAttribute('data-key');
    if (bm.has(k)) el.textContent = fmtBytesGB(bm.get(k));
  });
}
