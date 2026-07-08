/* tree/buildTree.js — 智能参数树构建（支持 MoE）
 * ------------------------------------------------------------
 * 输入：扁平张量列表 [{ name, dtype, shape, shard }]
 * 输出：结构化树，含逐层归类、MoE 专家坍缩、参数/字节统计。
 * 约定：参数量与精度无关，字节数由 UI 按所选精度实时换算。
 * ------------------------------------------------------------ */

export function tensorParams(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return 1;
  return shape.reduce((a, b) => a * b, 1);
}

// 匹配层级索引：绝大多数现代模型含 ".数字." 段作为 layer 下标
const LAYER_RE = /\.(\d+)\./;
const EXPERT_RE = /experts\.(\d+)\./;

function classifyLayer(remainder) {
  if (remainder.startsWith('self_attn') || remainder.startsWith('attention')) return 'attn';
  if (remainder.includes('experts.')) return 'expert';
  if (remainder.startsWith('mlp')) return 'mlp';
  if (/norm/.test(remainder)) return 'norm';
  return 'other';
}

function classifyNonLayer(name) {
  if (/embed_tokens/.test(name)) return 'Embedding';
  if (/lm_head/.test(name)) return 'LM Head';
  if (/norm/.test(name)) return 'Norm';
  return 'Other';
}

export function buildTree(tensors) {
  const layers = new Map(); // index -> { index, attn:[], mlp:[], norm:[], other:[], experts:Map }
  const nonLayer = new Map(); // group -> [tensors]

  let baseParams = 0; // 非专家（稠密）参数
  let expertParams = 0; // 全部 MoE 专家参数

  for (const t of tensors) {
    const params = tensorParams(t.shape);
    t.params = params;

    const lm = t.name.match(LAYER_RE);
    if (lm) {
      const li = parseInt(lm[1], 10);
      const remainder = t.name.slice(lm.index + lm[0].length);
      const kind = classifyLayer(remainder);

      if (!layers.has(li)) {
        layers.set(li, { index: li, attn: [], mlp: [], norm: [], other: [], experts: new Map() });
      }
      const layer = layers.get(li);

      if (kind === 'expert') {
        const em = remainder.match(EXPERT_RE);
        const ei = em ? parseInt(em[1], 10) : 0;
        if (!layer.experts.has(ei)) layer.experts.set(ei, []);
        layer.experts.get(ei).push(t);
        expertParams += params; // 计入专家总量（展示时按 ×N 坍缩）
      } else {
        layer[kind].push(t);
        baseParams += params;
      }
    } else {
      const g = classifyNonLayer(t.name);
      if (!nonLayer.has(g)) nonLayer.set(g, []);
      nonLayer.get(g).push(t);
      baseParams += params;
    }
  }

  // 将专家 map 坍缩为单一代表（取专家 0），记录总数
  let numExperts = 0;
  let isMoe = false;
  const layerArr = [];
  const maxIdx = layers.size ? Math.max(...layers.keys()) : -1;
  for (let i = 0; i <= maxIdx; i++) {
    const layer = layers.get(i);
    if (!layer) {
      layerArr.push(null);
      continue;
    }
    let layerParams = 0;
    for (const k of ['attn', 'mlp', 'norm', 'other']) {
      for (const t of layer[k]) layerParams += t.params;
    }

    let experts = null;
    const expertIdxs = [...layer.experts.keys()].sort((a, b) => a - b);
    if (expertIdxs.length) {
      isMoe = true;
      numExperts = Math.max(numExperts, expertIdxs.length);
      const rep = layer.experts.get(expertIdxs[0]); // 代表专家（形状与其他专家一致）
      let perExpertParams = 0;
      for (const t of rep) perExpertParams += t.params;
      const count = expertIdxs.length;
      const totalParams = perExpertParams * count;
      experts = { count, perExpertParams, totalParams, representative: rep };
      layerParams += totalParams;
    }

    layerArr.push({ ...layer, layerParams, experts });
  }

  const totalParams = baseParams + expertParams;
  const nonLayerOrdered = orderNonLayer(nonLayer);

  return {
    numLayers: maxIdx + 1,
    layers: layerArr,
    nonLayer: nonLayerOrdered,
    totalParams,
    baseParams,
    expertParams,
    isMoe,
    numExperts,
  };
}

function orderNonLayer(map) {
  const order = ['Embedding', 'LM Head', 'Norm', 'Other'];
  const out = [];
  for (const g of order) {
    if (map.has(g)) out.push({ group: g, tensors: map.get(g) });
  }
  for (const [g, ts] of map) {
    if (!order.includes(g)) out.push({ group: g, tensors: ts });
  }
  return out;
}
