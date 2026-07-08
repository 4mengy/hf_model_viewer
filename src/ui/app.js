/* ui/app.js — 应用编排（双形态共享）
 * ------------------------------------------------------------
 * 构建「左配置计算 / 右结果渲染」双栏布局，串接：
 *   analyze -> buildTree -> estimateVRAM -> 树渲染 + 图表 + 显存明细
 * Web 与扩展共用同一份此逻辑，仅挂载入口不同。
 * ------------------------------------------------------------ */

import '../styles.css';
import { analyze } from '../engine/index.js';
import { buildTree } from '../tree/index.js';
import { estimateVRAM, buildEffBppMap } from '../vram/index.js';
import { renderTree, updateTreeBytes } from './treeView.js';
import { renderChart, COLORS } from './chart.js';
import { fmtNum, fmtGB, esc } from './format.js';

// 读取模型最大上下文长度：优先 max_position_embeddings，回退 max_sequence_length
function getMaxContextLength(config) {
  const v = config.max_position_embeddings ?? config.max_sequence_length ?? config.max_position_embedding;
  return typeof v === 'number' && v > 0 ? v : null;
}

const LAYOUT = `
<div class="app">
  <aside class="sidebar">
    <div class="brand">LLM 架构与显存计算器<small>零下载解析 · 动态 VRAM 计算 · 双形态分发</small></div>

    <div class="search-row">
      <input id="repo" placeholder="org/repo，如 Qwen/Qwen2.5-7B-Instruct" />
      <button id="analyze">Analyze</button>
    </div>

    <details class="field">
      <summary>高级（受限模型可选 Token）</summary>
      <input id="token" class="seq-input" style="margin-top:8px" placeholder="hf_xxx（私有/受限仓库时填写）" />
    </details>

    <div class="field">
      <label>Quantization 精度</label>
      <div class="radios">
        <label><input type="radio" name="q" value="fp16" checked /> FP16/BF16</label>
        <label><input type="radio" name="q" value="int8" /> INT8</label>
        <label><input type="radio" name="q" value="int4" /> INT4</label>
      </div>
    </div>

    <div class="field">
      <label>量化策略</label>
      <select id="qstrat">
        <option value="uniform">均匀量化（全部张量按精度滑杆）</option>
        <option value="keep-fp16">仅 Linear 量化（Embedding/Norm/LM Head 保留 FP16）</option>
        <option value="native">按磁盘实际精度（忽略滑杆，含预量化）</option>
      </select>
      <p class="hint">权重显存以磁盘 dtype 为真值：已预量化模型（如 FP4/INT4）直接按实际占用计。</p>
    </div>

    <div class="field">
      <label>Batch Size：<span class="bubble" id="batchVal">1</span></label>
      <input type="range" id="batch" min="1" max="128" value="1" />
    </div>

    <div class="field">
      <label>Context Length</label>
      <div class="chips">
        <button data-seq="8192">8K</button>
        <button data-seq="32768">32K</button>
        <button data-seq="131072">128K</button>
      </div>
      <input type="number" id="seq" class="seq-input" value="8192" min="1024" max="131072" />
    </div>

    <div id="summary" class="summary" style="display:none"></div>
    <div id="status" class="status"></div>
  </aside>

  <main class="main">
    <section class="overview">
      <h2>总览</h2>
      <div class="summary-grid" id="stats"><div class="empty">尚未分析</div></div>
      <canvas id="chart"></canvas>
      <h3 class="comp-title">组成明细</h3>
      <div id="comp" class="comp-wrap"><div class="empty">尚未分析</div></div>
    </section>
    <section class="tree">
      <h2>层级结构树</h2>
      <div id="tree"><div class="empty">输入仓库 ID 并点击 Analyze 开始解析</div></div>
    </section>
  </main>
</div>`;

export function mountApp(rootEl) {
  rootEl.innerHTML = LAYOUT;

  const $ = (id) => rootEl.querySelector('#' + id);
  const repoInput = $('repo');
  const tokenInput = $('token');
  const analyzeBtn = $('analyze');
  const batchInput = $('batch');
  const batchVal = $('batchVal');
  const seqInput = $('seq');
  const chips = rootEl.querySelectorAll('.chips button');
  const statusEl = $('status');
  const summaryEl = $('summary');
  const statsEl = $('stats');
  const treeEl = $('tree');
  const canvas = $('chart');

  let state = null; // { config, tree, tensors }

  function getPrecision() {
    const el = rootEl.querySelector('input[name="q"]:checked');
    return el ? el.value : 'fp16';
  }

  function getStrategy() {
    const el = $('qstrat');
    return el ? el.value : 'uniform';
  }

  // 由当前精度 + 量化策略，计算每个张量的有效每参数字节（与计算器一致）
  function buildEffMap() {
    return buildEffBppMap(state.tensors, { targetPrecision: getPrecision(), strategy: getStrategy() });
  }

  function setStatus(msg, kind = '') {
    statusEl.className = 'status' + (kind ? ' ' + kind : '');
    statusEl.textContent = msg;
  }

  function recompute() {
    if (!state) return;
    const precision = getPrecision();
    const batch = parseInt(batchInput.value, 10) || 1;
    const seq = parseInt(seqInput.value, 10) || 8192;

    const est = estimateVRAM(state.config, state.tree, { precision, batch, seq, tensors: state.tensors, strategy: getStrategy() });

    renderChart(canvas, est);
    renderComposition(est);
    updateTreeBytes(treeEl, state.tree, buildEffMap());

    // 显存明细卡片（不含显卡推荐）
    summaryEl.style.display = '';
    summaryEl.innerHTML = `
      <div class="hw-note" style="font-size:13px">总显存需求：<b>${fmtGB(est.vTotal)}</b> ｜ 权重 ${fmtGB(est.vWeights)} ｜ KV ${est.kvUnknown ? '—' : fmtGB(est.vKV)} ｜ 开销 ${fmtGB(est.vOverhead)}</div>
      ${est.weightNote ? `<div class="hw-note">权重策略：${esc(est.weightNote)}</div>` : ''}
      ${est.kvFormulaLabel ? `<div class="hw-note">注意力架构：<span class="tag ${esc(est.attnArch)}">${esc(est.attnArch.toUpperCase())}</span> ｜ KV 公式：${esc(est.kvFormulaLabel)}</div>` : ''}
      ${est.kvNote ? `<div class="hw-note dsa-note">${esc(est.kvNote)}</div>` : ''}
    `;
  }

  // 总览「组成明细」：按 group 分组列出每个分类的细粒度构成与占比
  function renderComposition(est) {
    const compEl = $('comp');
    if (!est.composition || !est.composition.length) {
      compEl.innerHTML = '<div class="empty">尚未分析</div>';
      return;
    }
    const total = est.vTotal || 1;
    const groups = [
      { key: 'weight', title: '权重显存（稠密基础）' },
      { key: 'moe', title: 'MoE 专家层' },
      { key: 'kv', title: 'KV Cache' },
      { key: 'overhead', title: '固有开销' },
    ];
    let html = '<div class="comp">';
    for (const g of groups) {
      const items = est.composition.filter((c) => c.group === g.key);
      if (!items.length) continue;
      const sub = items.reduce((s, c) => s + c.gb, 0);
      html += `<div class="comp-group"><div class="comp-ghead"><span>${esc(g.title)}</span><b>${fmtGB(sub)}</b></div>`;
      for (const it of items) {
        const pct = (it.gb / total) * 100;
        const color = COLORS[it.key] || '#94a3b8';
        html += `<div class="comp-row"><span class="dot" style="background:${color}"></span><span class="comp-name">${esc(it.label)}</span><span class="comp-val">${fmtGB(it.gb)} · ${pct.toFixed(1)}%</span></div>`;
      }
      html += '</div>';
    }
    html += '</div>';
    compEl.innerHTML = html;
  }

  function renderStats() {
    const { config, tree } = state;
    const arch = Array.isArray(config.architectures)
      ? config.architectures.join(', ')
      : config.model_type || '—';
    statsEl.innerHTML = `
      <div class="stat">总参数<b>${fmtNum(tree.totalParams)}</b></div>
      <div class="stat">层数<b>${tree.numLayers}</b></div>
      <div class="stat">MoE<b>${tree.isMoe ? '是 ×' + tree.numExperts + ' 专家' : '否'}</b></div>
      <div class="stat">架构<b style="font-size:13px">${esc(arch)}</b></div>
      <div class="stat">分片数<b>${state.shardCount ?? '—'}</b></div>
    `;
  }

  async function run() {
    const repo = repoInput.value.trim();
    if (!repo) {
      setStatus('请输入仓库 ID（org/repo）', 'error');
      return;
    }
    analyzeBtn.disabled = true;
    setStatus('解析中：拉取 config.json …');
    try {
      const result = await analyze(repo, {
        token: tokenInput.value.trim() || undefined,
        onShard: (done, total, file) => {
          setStatus(`解析分片头部 ${done}/${total}：${file}`);
        },
      });
      state = { config: result.config, tree: buildTree(result.tensors), tensors: result.tensors, shardCount: result.shardCount };
      renderStats();

      // Context Length 默认填充为该模型的最大上下文长度
      const maxCtx = getMaxContextLength(result.config);
      if (maxCtx) {
        seqInput.max = String(Math.max(maxCtx, 131072));
        seqInput.value = String(maxCtx);
      } else {
        seqInput.value = '8192';
      }
      chips.forEach((x) => x.classList.remove('active'));

      renderTree(treeEl, state.tree, buildEffMap());
      recompute();
      setStatus(`解析完成：${result.shardCount} 个分片，${result.tensors.length} 个张量`, 'ok');
    } catch (e) {
      setStatus(e.message || String(e), 'error');
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  // ---- 事件绑定 ----
  analyzeBtn.addEventListener('click', run);
  repoInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') run();
  });

  batchInput.addEventListener('input', () => {
    batchVal.textContent = batchInput.value;
    recompute();
  });

  seqInput.addEventListener('input', () => recompute());
  chips.forEach((c) =>
    c.addEventListener('click', () => {
      seqInput.value = c.dataset.seq;
      chips.forEach((x) => x.classList.remove('active'));
      c.classList.add('active');
      recompute();
    }),
  );

  rootEl.querySelectorAll('input[name="q"]').forEach((r) => r.addEventListener('change', recompute));
  $('qstrat').addEventListener('change', recompute);
}
