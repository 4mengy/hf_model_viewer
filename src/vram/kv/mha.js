/* vram/kv/mha.js — 标准注意力（MHA / GQA / MQA）KV Cache 计算
 * ------------------------------------------------------------
 * 公式（Spec §4.2.2）：
 *   Vkv = 4 · B · S · L · Hkv · Dhead / 1024^3
 * 其中 Hkv 为 KV 头数（MHA 时 = nHeads，GQA 时 < nHeads，MQA 时 = 1）。
 *
 * 两条输入路径：
 *   - computeFromTensors：优先，从 K/V 投影权重形状推导（厂商无关）。
 *   - computeFromConfig：回退，从 config.json 超参推导（含融合 QKV 等字段 fallback）。
 *
 * 显存口径：KV 按 BF16(2B) 计（与 Spec 一致，不随精度滑杆缩放）。
 * ------------------------------------------------------------ */

import { num, outDim } from './detect.js';

const GB = 1024 ** 3;

function labelOf(Hkv, nHeads) {
  if (!Number.isFinite(nHeads) || nHeads <= 0) return 'mha';
  if (Hkv === nHeads) return 'mha';
  if (Hkv === 1) return 'mqa';
  return 'gqa';
}

export default {
  name: 'mha',

  /** 张量主路径：kvElements = out(k_proj) + out(v_proj)；L 由调用方给出 */
  computeFromTensors({ batch, seq, L, attnNames, byName, config }) {
    const kProj = attnNames.find((n) => /(^|[._])k_proj/i.test(n));
    const vProj = attnNames.find((n) => /(^|[._])v_proj/i.test(n));
    if (!kProj || !vProj) return null; // 融合 QKV → 交 config 回退

    const kvElements = outDim(byName.get(kProj)) + outDim(byName.get(vProj));
    const vKV = (batch * seq * L * kvElements * 2) / GB;

    // 用最稳定的 num_attention_heads 仅做标签细分（数值仍来自张量）
    const kOut = outDim(byName.get(kProj));
    const qProj = attnNames.find((n) => /(^|[._])q_proj/i.test(n));
    const qOut = qProj ? outDim(byName.get(qProj)) : NaN;
    const nHeads = num(config.num_attention_heads ?? config.n_heads, NaN);

    let attnArch = 'mha';
    let formulaLabel = 'MHA/GQA 架构（张量推导 K/V 投影维度）';
    if (Number.isFinite(qOut) && Number.isFinite(nHeads) && nHeads > 0) {
      const Dhead = qOut / nHeads;
      const Hkv = Number.isFinite(Dhead) && Dhead > 0 ? Math.round(kOut / Dhead) : kOut;
      attnArch = labelOf(Hkv, nHeads);
      formulaLabel = `${attnArch.toUpperCase()} 架构（张量推导 K/V 维度，Hkv≈${Hkv}, Dhead≈${Math.round(Dhead)}）`;
    }
    return { vKV, attnArch, formulaLabel, note: '' };
  },

  /** config 回退路径：多源字段 fallback，覆盖各家命名差异 */
  computeFromConfig({ batch, seq, config }) {
    const L = num(config.num_hidden_layers ?? config.n_layers ?? config.num_layers, NaN);
    if (!Number.isFinite(L)) throw new Error('缺 num_hidden_layers');

    const nHeads = config.num_attention_heads ?? config.n_heads;
    let Hkv = config.num_key_value_heads ?? config.num_kv_heads;
    if (Hkv === undefined && config.multi_query_attention) {
      Hkv = config.multi_query_group_num ?? 1;
    }
    if (Hkv === undefined) Hkv = nHeads;
    const hidden = config.hidden_size ?? config.d_model;
    const Dhead = config.head_dim ?? config.kv_channels ?? Math.floor(hidden / nHeads);
    if (![Hkv, Dhead, nHeads].every(Number.isFinite)) throw new Error('缺注意力头配置');

    const vKV = (4 * batch * seq * L * Hkv * Dhead) / GB;
    const attnArch = labelOf(Hkv, nHeads);
    return {
      vKV,
      attnArch,
      formulaLabel: `${attnArch.toUpperCase()} 架构 (Hkv=${Hkv}, Dhead=${Dhead})`,
      note: '',
    };
  },
};
