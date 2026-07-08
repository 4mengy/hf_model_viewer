/* vram/kv/detect.js — 注意力架构识别 + 张量几何解析（共享工具）
 * ------------------------------------------------------------
 * 本文件与 mha.js / mla.js / dsa.js 同处一个目录，专司两件事：
 *   1) 推断架构：detectAttnArch（基于 config.json）/ detectArchFromTensors（基于张量名）
 *   2) 把原始张量列表解析成「代表层」的注意力张量清单，供各架构模块消费。
 *
 * 设计原则：KV Cache 只统计「落入显存」的张量；层数 L 直接由张量名层号推出
 *   （不依赖 config 的 num_hidden_layers / num_layers 字段差异）。
 * ------------------------------------------------------------ */

const GB = 1024 ** 3;

export function num(v, d = 1) {
  return Number.isFinite(v) ? v : d;
}

/** 从张量名解析层号；支持 layers.N / layer.N / h.N / blocks.N / transformer.layers.N */
export function layerOf(name) {
  const m = name.match(/(?:^|[._])(?:layers?|h|blocks?)\.(\d+)\./);
  if (m) return +m[1];
  return -1;
}

/** 是否注意力相关张量（排除 mlp / experts / ffn / moe 等非注意力块；含 DSA 索引器） */
export function isAttnTensor(name) {
  if (/(?:mlp|expert|ffn|moe)/i.test(name)) return false;
  return /(?:attn|attention|self_attn|kv|k_proj|v_proj|q_proj|query_key_value|qkv|indexer|index_key|index_k)/i.test(name);
}

/** Linear 权重 [out, in] 的输出维度 = shape[0] */
export function outDim(meta) {
  const s = meta && meta.shape;
  return Array.isArray(s) && s.length >= 1 ? s[0] : NaN;
}

/**
 * 解析张量列表，抽取「代表层」的注意力张量。
 * 代表层选择：注意力张量数量最多的层（平局取层号最大者），比单纯取最大层号更稳健
 *   （避免末层结构缺失导致 K/V 投影不完整）。
 * @returns {{L:number, sampleLayer:number, attnNames:string[], byName:Map}|null}
 *          无层号信息（如纯 embedding）返回 null，交由 config 回退。
 */
export function extractLayerTensors(tensors) {
  if (!Array.isArray(tensors) || tensors.length === 0) return null;
  const layerSet = new Set();
  const attnCount = new Map();
  for (const t of tensors) {
    const L = layerOf(t.name);
    if (L < 0) continue;
    layerSet.add(L);
    if (isAttnTensor(t.name)) attnCount.set(L, (attnCount.get(L) || 0) + 1);
  }
  if (layerSet.size === 0) return null;
  const L = Math.max(...layerSet) + 1;

  // 代表层 = 注意力张量最多者；平局取层号最大
  let repLayer = -1;
  let repCount = -1;
  for (const lyr of layerSet) {
    const c = attnCount.get(lyr) || 0;
    if (c > repCount || (c === repCount && lyr > repLayer)) {
      repCount = c;
      repLayer = lyr;
    }
  }

  const byName = new Map(tensors.map((t) => [t.name, t]));
  const attnNames = tensors
    .filter((t) => layerOf(t.name) === repLayer && isAttnTensor(t.name))
    .map((t) => t.name);
  return { L, sampleLayer: repLayer, attnNames, byName };
}

/**
 * 基于 config.json 推断架构，优先级：dsa > mla > mha。
 * 仅用于「张量无法纯形状拆分」时的 config 超参回退路径。
 */
export function detectAttnArch(config = {}) {
  const arch = Array.isArray(config.architectures) ? config.architectures.join(' ') : '';
  const modelType = config.model_type || '';
  const hasDSA =
    config.index_head_dim !== undefined ||
    config.index_topk !== undefined ||
    /deepseekv3\.?2/i.test(arch) ||
    modelType === 'deepseek_v32';
  if (hasDSA) return 'dsa';
  if (config.kv_lora_rank !== undefined && config.kv_lora_rank !== null) return 'mla';
  return 'mha';
}

/**
 * 基于代表层注意力张量名推断架构：dsa > mla > mha。
 * 返回 'dsa' | 'mla' | 'mha' | null（null 表示融合 QKV 等无法纯形状识别的情形）。
 */
export function detectArchFromTensors(attnNames = []) {
  const has = (re) => attnNames.some((n) => re.test(n));
  if (has(/indexer|lightning|index_key|index_k/i)) return 'dsa';
  if (has(/kv_a_proj|kv_b_proj|kv_proj|kv_a_layernorm/i)) return 'mla';
  if (has(/(^|[._])k_proj/i) && has(/(^|[._])v_proj/i)) return 'mha';
  return null;
}

export { GB };
