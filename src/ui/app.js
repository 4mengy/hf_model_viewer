/* ui/app.js — App orchestration (shared by web & extension)
 * ------------------------------------------------------------
 * Two-column layout: left = config & calc controls, right = results.
 * Pipeline: analyze -> buildTree -> estimateVRAM -> tree + chart + breakdown.
 * Web and extension share this logic; only the mount entry differs.
 * Language: strings come from i18n (t); switching re-paints the shell.
 * ------------------------------------------------------------ */

import '../styles.css';
import { analyze } from '../engine/index.js';
import { buildTree } from '../tree/index.js';
import { estimateVRAM, buildEffBppMap } from '../vram/index.js';
import { renderTree, updateTreeBytes } from './treeView.js';
import { renderChart, COLORS } from './chart.js';
import { fmtNum, fmtGB, esc } from './format.js';
import { t, getLang, setLang, onLangChange } from '../i18n.js';

// Read model max context length: prefer max_position_embeddings, fall back
// to max_sequence_length / max_position_embedding.
function getMaxContextLength(config) {
  const v = config.max_position_embeddings ?? config.max_sequence_length ?? config.max_position_embedding;
  return typeof v === 'number' && v > 0 ? v : null;
}

function buildLayout() {
  const lang = getLang();
  const active = (l) => (lang === l ? ' active' : '');
  return `
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      ${esc(t('brand.title'))}<small>${esc(t('brand.sub'))}</small>
      <div class="lang-toggle">
        <button class="lang-btn${active('zh')}" data-lang="zh">中文</button>
        <button class="lang-btn${active('en')}" data-lang="en">EN</button>
      </div>
    </div>

    <div class="search-row">
      <input id="repo" placeholder="${esc(t('ctl.repoPlaceholder'))}" />
      <button id="analyze">${esc(t('ctl.analyze'))}</button>
    </div>

    <details class="field">
      <summary>${esc(t('ctl.advanced'))}</summary>
      <input id="token" class="seq-input" style="margin-top:8px" placeholder="${esc(t('ctl.tokenPlaceholder'))}" />
    </details>

    <div class="field">
      <label>${esc(t('ctl.quantPrecision'))}</label>
      <div class="radios">
        <label><input type="radio" name="q" value="fp16" checked /> FP16/BF16</label>
        <label><input type="radio" name="q" value="int8" /> INT8</label>
        <label><input type="radio" name="q" value="int4" /> INT4</label>
      </div>
    </div>

    <div class="field">
      <label>${esc(t('ctl.quantStrategy'))}</label>
      <select id="qstrat">
        <option value="uniform">${esc(t('ctl.stratUniform'))}</option>
        <option value="keep-fp16">${esc(t('ctl.stratKeepFp16'))}</option>
        <option value="native">${esc(t('ctl.stratNative'))}</option>
      </select>
      <p class="hint">${esc(t('ctl.quantHint'))}</p>
    </div>

    <div class="field">
      <label>${esc(t('ctl.batchSize'))}<span class="bubble" id="batchVal">1</span></label>
      <input type="range" id="batch" min="1" max="128" value="1" />
    </div>

    <div class="field">
      <label>${esc(t('ctl.contextLength'))}</label>
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
      <h2>${esc(t('ov.title'))}</h2>
      <div class="summary-grid" id="stats"><div class="empty">${esc(t('ctl.empty'))}</div></div>
      <canvas id="chart"></canvas>
      <h3 class="comp-title">${esc(t('ov.compTitle'))}</h3>
      <div id="comp" class="comp-wrap"><div class="empty">${esc(t('ctl.empty'))}</div></div>
      <h3 class="comp-title">${esc(t('ov.kvTitle'))}</h3>
      <div id="kvdetails" class="kv-details"><div class="empty">${esc(t('ctl.empty'))}</div></div>
    </section>
    <section class="tree">
      <h2>${esc(t('ov.treeTitle'))}</h2>
      <div id="tree"><div class="empty">${esc(t('ov.treeEmpty'))}</div></div>
    </section>
  </main>
</div>`;
}

export function mountApp(rootEl) {
  let state = null; // { config, tree, tensors, shardCount }
  let lastRepo = '';

  const $ = (id) => rootEl.querySelector('#' + id);

  function getPrecision() {
    const el = rootEl.querySelector('input[name="q"]:checked');
    return el ? el.value : 'fp16';
  }

  function getStrategy() {
    const el = $('qstrat');
    return el ? el.value : 'uniform';
  }

  // Effective per-parameter bytes for every tensor (matches the calculator).
  function buildEffMap() {
    return buildEffBppMap(state.tensors, { targetPrecision: getPrecision(), strategy: getStrategy() });
  }

  function setStatus(msg, kind = '') {
    const statusEl = $('status');
    statusEl.className = 'status' + (kind ? ' ' + kind : '');
    statusEl.textContent = msg;
  }

  function recompute() {
    if (!state) return;
    const precision = getPrecision();
    const batch = parseInt($('batch').value, 10) || 1;
    const seq = parseInt($('seq').value, 10) || 8192;

    const est = estimateVRAM(state.config, state.tree, {
      precision,
      batch,
      seq,
      tensors: state.tensors,
      strategy: getStrategy(),
    });

    renderChart($('chart'), est);
    renderComposition(est);
    renderKVDetails(est);
    updateTreeBytes($('tree'), state.tree, buildEffMap());

    // VRAM breakdown card (no GPU recommendation).
    const summaryEl = $('summary');
    summaryEl.style.display = '';
    const profile = est.kvProfile;
    summaryEl.innerHTML = `
      <div class="hw-note" style="font-size:13px">${esc(t('sum.total'))}<b>${fmtGB(est.vTotal)}</b> ｜ ${esc(t('sum.weights'))} ${fmtGB(est.vWeights)} ｜ KV ${est.kvUnknown ? '—' : fmtGB(est.vKV)} ｜ ${esc(t('sum.overhead'))} ${fmtGB(est.vOverhead)}</div>
      ${est.weightNote ? `<div class="hw-note">${esc(t('sum.weightStrategy'))}${esc(est.weightNote)}</div>` : ''}
      ${profile ? `<div class="hw-note">${esc(t('sum.kvProfile'))}<span class="tag profile">${esc(profile.label)}</span> ｜ ${esc(t('sum.kvLayout'))}${esc(profile.layout.id)}@${esc(profile.layout.version)}</div>` : `<div class="hw-note incomplete">${esc(t('kv.totalUnknown'))}</div>`}
      ${est.kvNote ? `<div class="hw-note dsa-note">${esc(est.kvNote)}</div>` : ''}
    `;
  }

  function diagnosticLabel(diagnostic) {
    if (!diagnostic) return t('kv.diag.unknown');
    const key = `kv.diag.${diagnostic.code}`;
    const translated = t(key);
    return translated === key ? diagnostic.code : translated;
  }

  function layerGroupLabel(group = {}) {
    const count = Number.isFinite(group.count) ? ` × ${group.count}` : '';
    if (Array.isArray(group.range)) return `${group.label || t('kv.layers')} ${group.range[0]}–${group.range[1]}${count}`;
    if (Array.isArray(group.indices)) return `${group.label || t('kv.layers')} [${group.indices.join(', ')}]${count}`;
    return `${group.label || '—'}${count}`;
  }

  function renderKVDetails(est) {
    const el = $('kvdetails');
    if (est.kvStatus !== 'verified' || !est.kvProfile) {
      const diagnostic = est.kvDiagnostic;
      const ids = diagnostic?.modelClassIdentifiers?.join(', ') || '—';
      const mismatches = diagnostic?.details?.mismatches || [];
      const shown = mismatches.slice(0, 12);
      const remainder = mismatches.length - shown.length;
      el.innerHTML = `
        <div class="kv-unknown">
          <b>${esc(t('kv.unsupported'))}</b>
          <div>${esc(diagnosticLabel(diagnostic))}</div>
          <div><code>${esc(ids)}</code></div>
          ${shown.length ? `<div class="kv-mismatch">${esc(t('kv.mismatches'))}${shown.map(esc).join(', ')}${remainder > 0 ? ` (+${remainder})` : ''}</div>` : ''}
        </div>`;
      return;
    }

    const profile = est.kvProfile;
    const rows = est.kvBuffers.map((buffer) => `
      <tr>
        <td><b>${esc(buffer.label)}</b><code>${esc(buffer.id)}</code></td>
        <td>${esc(layerGroupLabel(buffer.layerGroup))}</td>
        <td class="num">${esc(fmtNum(buffer.elements))}</td>
        <td><code>${esc(buffer.dtype)}</code> × ${buffer.bytesPerElement}B</td>
        <td class="num">${esc(fmtNum(buffer.bytes))} B<br><span>${esc(fmtGB(buffer.gb))}</span></td>
        <td><code>${esc(buffer.formula)}</code><small>${esc((buffer.evidenceIds || []).join(', '))}</small></td>
      </tr>`).join('');
    const evidence = profile.evidence.map((item) => `
      <li><a href="${esc(item.url)}" target="_blank" rel="noreferrer">${esc(item.label)}</a><code>${esc(item.revision)}</code></li>`).join('');
    el.innerHTML = `
      <div class="kv-profile-head">
        <div><b>${esc(profile.label)}</b><code>${esc(profile.id)}@${esc(profile.version)}</code></div>
        <span class="tag verified">${esc(t('kv.verified'))}</span>
      </div>
      <div class="kv-layout-id">${esc(t('sum.kvLayout'))}<code>${esc(profile.layout.id)}@${esc(profile.layout.version)}</code></div>
      <div class="kv-table-wrap"><table class="kv-table">
        <thead><tr><th>${esc(t('kv.buffer'))}</th><th>${esc(t('kv.layers'))}</th><th>${esc(t('kv.elements'))}</th><th>${esc(t('kv.dtype'))}</th><th>${esc(t('kv.bytes'))}</th><th>${esc(t('kv.formula'))}</th></tr></thead>
        <tbody>${rows}</tbody>
        <tfoot><tr><th colspan="4">${esc(t('group.kv'))}</th><th class="num">${esc(fmtNum(est.kvBuffers.reduce((sum, buffer) => sum + buffer.bytes, 0)))} B<br><span>${esc(fmtGB(est.vKV))}</span></th><th></th></tr></tfoot>
      </table></div>
      <details class="kv-evidence"><summary>${esc(t('kv.evidence'))}</summary><ul>${evidence}</ul></details>`;
  }

  // Overview "composition breakdown": group rows by category with size & share.
  function renderComposition(est) {
    const compEl = $('comp');
    if (!est.composition || !est.composition.length) {
      compEl.innerHTML = `<div class="empty">${esc(t('ctl.empty'))}</div>`;
      return;
    }
    const total = est.complete ? est.vTotal : null;
    const groups = [
      { key: 'weight', title: t('group.weight') },
      { key: 'moe', title: t('group.moe') },
      { key: 'kv', title: t('group.kv') },
      { key: 'overhead', title: t('group.overhead') },
    ];
    let html = '<div class="comp">';
    for (const g of groups) {
      const items = est.composition.filter((c) => c.group === g.key);
      if (!items.length) continue;
      const sub = items.reduce((s, c) => s + c.gb, 0);
      html += `<div class="comp-group"><div class="comp-ghead"><span>${esc(g.title)}</span><b>${fmtGB(sub)}</b></div>`;
      for (const it of items) {
        const pct = total > 0 ? (it.gb / total) * 100 : null;
        const color = COLORS[it.key] || '#94a3b8';
        html += `<div class="comp-row"><span class="dot" style="background:${color}"></span><span class="comp-name">${esc(t(it.labelKey))}</span><span class="comp-val">${fmtGB(it.gb)}${pct == null ? '' : ` · ${pct.toFixed(1)}%`}</span></div>`;
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
    const moe = tree.isMoe ? t('stat.moeYes', { n: tree.numExperts }) : t('stat.no');
    const shared = tree.hasSharedExperts ? t('stat.sharedYes') : '';
    $('stats').innerHTML = `
      <div class="stat">${esc(t('stat.totalParams'))}<b>${fmtNum(tree.totalParams)}</b></div>
      <div class="stat">${esc(t('stat.layers'))}<b>${tree.numLayers}</b></div>
      <div class="stat">${esc(t('stat.moe'))}<b>${moe}</b></div>
      ${shared ? `<div class="stat">${esc(t('stat.moe'))}<b style="color:#be185d">${esc(shared)}</b></div>` : ''}
      <div class="stat">${esc(t('stat.arch'))}<b style="font-size:13px">${esc(arch)}</b></div>
      <div class="stat">${esc(t('stat.shards'))}<b>${state.shardCount ?? '—'}</b></div>
    `;
  }

  async function run() {
    const repoInput = $('repo');
    const repo = repoInput.value.trim();
    if (!repo) {
      setStatus(t('status.enterRepo'), 'error');
      return;
    }
    const analyzeBtn = $('analyze');
    analyzeBtn.disabled = true;
    setStatus(t('status.fetching'));
    try {
      const result = await analyze(repo, {
        token: $('token').value.trim() || undefined,
        onShard: (done, total, file) => {
          setStatus(t('status.shard', { done, total, file }));
        },
      });
      state = {
        config: result.config,
        tree: buildTree(result.tensors),
        tensors: result.tensors,
        shardCount: result.shardCount,
      };
      lastRepo = repo;
      renderStats();

      // Context Length defaults to this model's max context length.
      const maxCtx = getMaxContextLength(result.config);
      const seqInput = $('seq');
      if (maxCtx) {
        seqInput.max = String(Math.max(maxCtx, 131072));
        seqInput.value = String(maxCtx);
      } else {
        seqInput.value = '8192';
      }
      rootEl.querySelectorAll('.chips button').forEach((x) => x.classList.remove('active'));

      renderTree($('tree'), state.tree, buildEffMap());
      recompute();
      setStatus(t('status.done', { shards: result.shardCount, tensors: result.tensors.length }), 'ok');
    } catch (e) {
      setStatus(e.message || String(e), 'error');
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  // ---- Shell render + event binding ----
  function bindShell() {
    const analyzeBtn = $('analyze');
    const repoInput = $('repo');
    const batchInput = $('batch');
    const seqInput = $('seq');

    analyzeBtn.addEventListener('click', run);
    repoInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') run();
    });

    batchInput.addEventListener('input', () => {
      $('batchVal').textContent = batchInput.value;
      recompute();
    });

    seqInput.addEventListener('input', () => recompute());
    rootEl.querySelectorAll('.chips button').forEach((c) =>
      c.addEventListener('click', () => {
        seqInput.value = c.dataset.seq;
        rootEl.querySelectorAll('.chips button').forEach((x) => x.classList.remove('active'));
        c.classList.add('active');
        recompute();
      }),
    );

    rootEl.querySelectorAll('input[name="q"]').forEach((r) => r.addEventListener('change', recompute));
    $('qstrat').addEventListener('change', recompute);

    rootEl.querySelectorAll('.lang-btn').forEach((b) =>
      b.addEventListener('click', () => setLang(b.dataset.lang)),
    );
  }

  function paint() {
    rootEl.innerHTML = buildLayout();
    document.title = t('brand.title');
    document.documentElement.lang = getLang() === 'zh' ? 'zh-CN' : 'en';
    bindShell();
    if (state) {
      $('repo').value = lastRepo;
      renderStats();
      renderTree($('tree'), state.tree, buildEffMap());
      recompute();
    }
  }

  // Re-paint the shell (and results, if any) when language changes.
  onLangChange(paint);

  paint();
}
