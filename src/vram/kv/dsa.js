/* vram/kv/dsa.js — DSA（DeepSeek Sparse Attention，V3.2）KV Cache 计算
 * ------------------------------------------------------------
 * DSA = MLA latent + FP8 索引器 K 缓存：
 *   Vkv = B · S · L · [ 2·(kv_lora_rank + qk_rope_head_dim) + index_head_dim ] / 1024^3
 *
 * 架构事实（重要）：DSA 的核心收益是「计算量」 O(L²)→O(L·topk) 与「显存带宽」
 *   （每步自回归仅加载 top-k 个 KV），**并非**削减 KV Cache 容量。其 VRAM 中的
 *   KV Cache ≈ 稠密 MLA 全量 latent + 索引器，容量基本持平、略增约
 *   index_head_dim / (kv_lora_rank + qk_rope_head_dim) ≈ 11%。
 *   —— top-k 选择是计算期加载行为，不占显存，绝不计入 Vkv（见显存口径原则）。
 *
 * 显存口径：MLA latent 按 BF16(2B)；索引器 K 固定 FP8(1B)。
 * ------------------------------------------------------------ */

import { num, outDim } from './detect.js';

const GB = 1024 ** 3;

/** DSA 说明（供 UI 提示，也用于 MLA→DSA 升级补计时的 note） */
export const DSA_NOTE =
  'DSA 主要削减计算量 O(L²)→O(L·topk) 与加载带宽；KV 容量≈稠密 MLA+索引器，几乎不降。';

const MLA_KV_RE = /kv_a_proj|kv_b_proj|kv_proj|kv_a_layernorm/i;
const INDEXER_RE = /indexer|lightning|index_key|index_k/i;

export default {
  name: 'dsa',

  /** 张量主路径：MLA latent + 索引器 K（FP8）。索引器张量缺失则返回 null 交升级逻辑 */
  computeFromTensors({ batch, seq, L, attnNames, byName }) {
    const mlaKv = attnNames.find((n) => MLA_KV_RE.test(n));
    const indexer = attnNames.find((n) => INDEXER_RE.test(n));
    if (!mlaKv || !indexer) return null; // 缺索引器 → 由 index.js 走 MLA+config 升级
    const kvElements = outDim(byName.get(mlaKv));
    const indexerDim = outDim(byName.get(indexer));
    const vKV = (batch * seq * L * (kvElements * 2 + indexerDim * 1)) / GB;
    return {
      vKV,
      attnArch: 'dsa',
      formulaLabel: 'DSA 稀疏注意力（张量推导 latent + FP8 索引器）',
      note: DSA_NOTE,
    };
  },

  /** config 回退路径 */
  computeFromConfig({ batch, seq, config }) {
    const L = num(config.num_hidden_layers ?? config.n_layers ?? config.num_layers, NaN);
    if (!Number.isFinite(L)) throw new Error('缺 num_hidden_layers');
    const Dcomp = num(config.kv_lora_rank, NaN);
    const Drope = num(config.qk_rope_head_dim ?? 0, 0);
    const Didx = num(config.index_head_dim ?? 128, 128);
    if (!Number.isFinite(Dcomp)) throw new Error('缺 kv_lora_rank');
    const vKV = (batch * seq * L * (2 * (Dcomp + Drope) + Didx)) / GB;
    return {
      vKV,
      attnArch: 'dsa',
      formulaLabel: 'DSA 稀疏注意力 (MLA latent + FP8 索引器K)',
      note: DSA_NOTE,
    };
  },
};
