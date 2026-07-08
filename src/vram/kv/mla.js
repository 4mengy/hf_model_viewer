/* vram/kv/mla.js — MLA（Multi-head Latent Attention，DeepSeek 系）KV Cache 计算
 * ------------------------------------------------------------
 * 公式（Spec §4.2.2）：
 *   Vkv = 2 · B · S · L · (kv_lora_rank + qk_rope_head_dim) / 1024^3
 * 关键点：MLA 不存逐头 K/V，只存一份压缩 latent（kv_lora_rank + qk_rope_head_dim），
 *   由 kv_a_proj_with_mqa 的输出维度给出，latent 共享计一次。
 *
 * 显存口径：latent 按 BF16(2B) 计。DSA 在此基础上额外塞一份 FP8 索引器（见 dsa.js）。
 * ------------------------------------------------------------ */

import { num, outDim } from './detect.js';

const GB = 1024 ** 3;

const MLA_KV_RE = /kv_a_proj|kv_b_proj|kv_proj|kv_a_layernorm/i;

export default {
  name: 'mla',

  /** 张量主路径：kvElements = out(kv_a_proj_with_mqa) */
  computeFromTensors({ batch, seq, L, attnNames, byName }) {
    const mlaKv = attnNames.find((n) => MLA_KV_RE.test(n));
    if (!mlaKv) return null;
    const kvElements = outDim(byName.get(mlaKv));
    const vKV = (batch * seq * L * kvElements * 2) / GB;
    return { vKV, attnArch: 'mla', formulaLabel: 'MLA 压缩架构（张量推导 latent）', note: '' };
  },

  /** config 回退路径 */
  computeFromConfig({ batch, seq, config }) {
    const L = num(config.num_hidden_layers ?? config.n_layers ?? config.num_layers, NaN);
    if (!Number.isFinite(L)) throw new Error('缺 num_hidden_layers');
    const Dcomp = num(config.kv_lora_rank, NaN);
    const Drope = num(config.qk_rope_head_dim ?? 0, 0);
    if (!Number.isFinite(Dcomp)) throw new Error('缺 kv_lora_rank');
    const vKV = (2 * batch * seq * L * (Dcomp + Drope)) / GB;
    return {
      vKV,
      attnArch: 'mla',
      formulaLabel: 'MLA 压缩架构 (kv_lora_rank + qk_rope_head_dim)',
      note: '',
    };
  },
};
