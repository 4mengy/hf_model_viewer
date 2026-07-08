/* vram/estimate.js — 动态显存（VRAM）计算器
 * ------------------------------------------------------------
 * 数学模型（Spec §4）：
 *   Vtotal = Vweights + Vkv_cache + Voverhead
 *   Vweights = Σparams × B_precision / 1024^3
 *   Vkv_cache = 4·B·S·L·Hkv·Dhead / 1024^3   (MHA/GQA/MQA)
 *   Vkv_cache = 2·B·S·L·(kv_lora_rank + qk_rope_head_dim) / 1024^3   (MLA)
 *   Vkv_cache = B·S·L·[ 2·(kv_lora_rank + qk_rope_head_dim) + index_head_dim ] / 1024^3   (DSA)
 *   Voverhead = 2.0 + Vweights × 10%
 *
 * 【通用解法】KV Cache 优先从「已解析的 safetensors 张量形状」推导
 *   （见 kv/ 目录：detect.js 推断架构 + 各架构模块 mha/mla/dsa 计算）：每家
 *   config.json 的注意力超参字段名都不一致（num_hidden_layers / num_layers；
 *    num_key_value_heads / multi_query_group_num；head_dim / kv_channels …），但
 *   「每层每 token 存入显存的 KV 元素数」恰好等于该层 K/V 投影权重的输出维度之和
 *   ——形状是厂商无关的 ground truth，层数 L 也从张量名层号推出。只有 DeepSeek DSA
 *   的「索引器额外缓存」仍需借一个稳定字段。张量推导失败（如融合 QKV 的 ChatGLM /
 *   GPT-NeoX）时，回退到 config 超参推导。
 *
 * 设计原则：计算器只统计「落入显存」的张量，不把「参与计算」的量（如 DSA top-k）计入。
 * ------------------------------------------------------------ */

const GB = 1024 ** 3;

const BPP = { fp16: 2, bf16: 2, int8: 1, int4: 0.5 };

import { computeKV } from './kv/index.js';
import { tensorParams } from '../tree/buildTree.js';

export function bytesPerParam(precision) {
  return BPP[precision] ?? 2;
}

/* ------------------------------------------------------------
 * 磁盘实际 dtype → 每参数字节（权重显存的真值来源）
 * safetensors 元数据中每个张量都带着自己的 dtype；若模型本就以
 * INT4/FP4/FP8 等预量化格式发布，其真实占用即由该 dtype 决定，
 * 不应被推理精度滑杆「再压缩」或「膨胀」。
 * ------------------------------------------------------------ */
const DTYPE_BYTES = {
  F64: 8, F32: 4, F16: 2, BF16: 2,
  F8_E4M3FN: 1, F8_E5M2: 1, F8_E4M3FNUZ: 1, F8_E5M2FNUZ: 1,
  INT8: 1, UINT8: 1, INT4: 0.5, UINT4: 0.5,
  F4: 0.5, NF4: 0.5, F4E2M3FNUZ: 0.5,
  I16: 2, I32: 4, I64: 8, BOOL: 0.125,
};

/** 解析 safetensors dtype 字符串为每参数字节；未知 dtype 回退到 fp16(2B) */
export function bytesForDtype(dtype, fallback = 2) {
  if (typeof dtype === 'string' && DTYPE_BYTES[dtype] != null) return DTYPE_BYTES[dtype];
  if (typeof dtype === 'string') {
    if (/(INT|UINT|F4|NF4)/.test(dtype) && /4/.test(dtype)) return 0.5;
    if (/8/.test(dtype)) return 1;
    if (/16/.test(dtype)) return 2;
    if (/32/.test(dtype)) return 4;
    if (/64/.test(dtype)) return 8;
  }
  return fallback;
}

/** 受保护张量：量化策略为「保留 FP16」时不参与量化 */
function isProtectedTensor(name) {
  return /embed_tokens|lm_head|norm/i.test(name);
}

/* ------------------------------------------------------------
 * 张量归类（用于总览的细粒度组成拆解）
 * 与 buildTree 的归类口径保持一致，但更细——把稠密基础层进一步
 * 拆为 嵌入层 / 注意力 / MLP / 归一化 / 其他 / LM Head，便于总览展开。
 * ------------------------------------------------------------ */
const EXPERT_TOKEN_RE = /experts\.\d+/i;
function categorizeTensor(name) {
  if (/embed_tokens/i.test(name)) return 'embedding';
  if (/lm_head/i.test(name)) return 'lmhead';
  if (EXPERT_TOKEN_RE.test(name)) return 'expert';
  const lm = name.match(/\.(\d+)\./);
  if (lm) {
    const remainder = name.slice(lm.index + lm[0].length);
    // 注意顺序：norm 先于 attention，否则 post_attention_layernorm 会被误判为注意力
    if (/norm/i.test(remainder)) return 'norm';
    if (/(?:^|[._])(?:mlp|ffn)/i.test(remainder)) return 'mlp';
    if (/(?:^|[._])(?:self_attn|attention)/i.test(remainder)) return 'attn';
    return 'other';
  }
  if (/norm/i.test(name)) return 'norm';
  return 'other';
}

/**
 * 单个张量的「有效每参数字节」——综合磁盘 dtype 与量化策略：
 *   - native：       直接用磁盘实际 dtype（忽略滑杆，展示真实文件占用）
 *   - keep-fp16：    受保护张量(Embedding/Norm/LMHead)保留 ≥FP16；其余取 min(原生, 目标)
 *   - uniform：      所有张量取 min(原生, 目标) —— 既模拟量化、又尊重已预量化的磁盘
 * 核心：min(原生, 目标) 保证「本就是 FP4 就按 FP4 算」，不会因滑杆膨胀或重复压缩。
 */
function effBppFor(t, { targetPrecision, strategy }) {
  const nativeB = bytesForDtype(t.dtype, 2);
  const targetB = BPP[targetPrecision] ?? 2;
  if (strategy === 'native') return nativeB;
  if (strategy === 'keep-fp16' && isProtectedTensor(t.name)) {
    return nativeB > 2 ? 2 : nativeB; // 保留 FP16 或更好
  }
  return Math.min(nativeB, targetB);
}

/**
 * 逐张量计算权重字节（同时产出 effBppMap 供树视图对账、byCategory 供总览细粒度拆解）。
 * @returns {{totalBytes:number, baseBytes:number, expertBytes:number, effBppMap:Map, byCategory:Map}|null}
 */
export function computeWeightBytes(tensors, opts = {}) {
  if (!Array.isArray(tensors) || !tensors.length) return null;
  const map = new Map();
  const byCategory = new Map();
  let totalBytes = 0, baseBytes = 0, expertBytes = 0;
  for (const t of tensors) {
    const params = t.params != null ? t.params : tensorParams(t.shape);
    const eff = effBppFor(t, opts);
    map.set(t.name, eff);
    const b = params * eff;
    totalBytes += b;
    const cat = categorizeTensor(t.name);
    byCategory.set(cat, (byCategory.get(cat) || 0) + b);
    if (cat === 'expert') expertBytes += b;
    else baseBytes += b;
  }
  return { totalBytes, baseBytes, expertBytes, effBppMap: map, byCategory };
}

export function buildEffBppMap(tensors, opts) {
  return computeWeightBytes(tensors, opts)?.effBppMap ?? new Map();
}


/**
 * @param {object} config   解析出的模型 config.json
 * @param {object} tree     buildTree 的结果
 * @param {object} opts     { precision, batch, seq, tensors? }
 *        tensors：已解析的扁平张量列表（analyze 返回），用于通用 KV 推导
 */
export function estimateVRAM(config, tree, { precision = 'fp16', batch = 1, seq = 8192, tensors = null, strategy = 'uniform' } = {}) {
  const bpp = bytesPerParam(precision);
  const totalParams = tree.totalParams;

  // 权重显存：优先按「每个张量的磁盘 dtype + 量化策略」逐张量计算
  // （磁盘实际精度为真值；推理精度滑杆仅对全精度张量模拟量化，已预量化层按磁盘计）
  const w = tensors && tensors.length ? computeWeightBytes(tensors, { targetPrecision: precision, strategy }) : null;
  let vWeights, baseWeightsGB, moeWeightsGB, weightNote;
  if (w) {
    vWeights = w.totalBytes / GB;
    baseWeightsGB = w.baseBytes / GB;
    moeWeightsGB = w.expertBytes / GB;
  } else {
    // 兜底：无张量明细时退化为「全模型均匀 × 滑杆精度」
    vWeights = (totalParams * bpp) / GB;
    baseWeightsGB = (tree.baseParams * bpp) / GB;
    moeWeightsGB = (tree.expertParams * bpp) / GB;
  }
  if (strategy === 'native') {
    weightNote = '权重按磁盘实际 dtype 逐张量计算（已含任何预量化，滑杆精度不生效）';
  } else if (strategy === 'keep-fp16') {
    weightNote = '仅 Linear 量化到目标精度；Embedding / Norm / LM Head 保留 FP16';
  } else {
    weightNote = '全模型均匀量化到目标精度；已预量化层按磁盘实际精度计（不重复压缩）';
  }

  let attnArch = 'mha';
  let vKV = null;
  let kvFormulaLabel = '';
  let kvNote = '';
  let kvUnknown = false;
  let kvMethod = 'config';

  // KV Cache 推算统一交由 kv/ 模块：主路径张量推导，失败回退 config 超参
  const kv = computeKV({ config, tensors, precision, batch, seq });
  vKV = kv.vKV;
  attnArch = kv.attnArch ?? 'mha';
  kvFormulaLabel = kv.formulaLabel || '';
  kvNote = kv.note || '';
  kvUnknown = !!kv.kvUnknown;
  kvMethod = kv.kvMethod || 'config';

  const vOverhead = 2.0 + vWeights * 0.1;
  const vKVsafe = vKV ?? 0;
  const vTotal = vWeights + vKVsafe + vOverhead;

  // 细粒度组成（供总览拆解）：按张量类别拆分权重，叠加 KV / 开销
  const CAT_META = {
    embedding: { label: '嵌入层 Embedding', group: 'weight' },
    attn: { label: '注意力层（稠密）', group: 'weight' },
    mlp: { label: 'MLP / FFN（稠密）', group: 'weight' },
    norm: { label: '归一化层（稠密）', group: 'weight' },
    other: { label: '其他（稠密）', group: 'weight' },
    lmhead: { label: 'LM Head', group: 'weight' },
    expert: { label: 'MoE 专家层', group: 'moe' },
  };
  const composition = [];
  if (w) {
    for (const [key, meta] of Object.entries(CAT_META)) {
      // 仅当该分类在模型中真实存在（byCategory 命中）才列出；
      // 稠密模型不含专家 → 不显示「MoE 专家层」；存在但极小的分类仍如实展示。
      if (!w.byCategory.has(key)) continue;
      const gb = (w.byCategory.get(key) || 0) / GB;
      composition.push({ key, label: meta.label, group: meta.group, gb });
    }
  } else {
    composition.push({ key: 'weight', label: '权重（未细分）', group: 'weight', gb: vWeights });
  }
  composition.push({ key: 'kv', label: 'KV Cache', group: 'kv', gb: vKVsafe });
  composition.push({ key: 'overhead', label: '固有开销', group: 'overhead', gb: vOverhead });

  // 图表分解（已使用逐张量有效字节）
  return {
    precision,
    bpp,
    totalParams,
    vWeights,
    weightNote,
    vKV,
    kvUnknown,
    kvFormulaLabel,
    kvNote,
    kvMethod,
    attnArch,
    isMLA: attnArch === 'mla',
    vOverhead,
    vTotal,
    composition,
    breakdown: {
      baseWeightsGB,
      moeWeightsGB,
      kvGB: vKV,
      overheadGB: vOverhead,
    },
  };
}
