/* vram/kv/index.js — KV Cache 调度中心（注册表 + 双路径分发）
 * ------------------------------------------------------------
 * 每种架构一个独立文件（mha.js / mla.js / dsa.js），推断架构的逻辑在 detect.js，
 * 三者同处本目录，便于扩展：新增架构只需在目录下加一个文件、在此注册即可。
 *
 * 调度策略：
 *   1) 主路径「张量推导」：从 safetensors 张量形状算 KV（厂商无关，免字段名猜测）。
 *      由 detectArchFromTensors 判定架构，调用对应模块的 computeFromTensors。
 *   2) 回退「config 推导」：仅当张量无法纯形状拆分（如融合 QKV）或缺失时，
 *      由 detectAttnArch 判定架构，调用对应模块的 computeFromConfig。
 *   3) DSA 升级：张量识别为 MLA latent、但 config 暴露 index_head_dim → 补计 FP8 索引器。
 *
 * 返回：{ vKV, attnArch, formulaLabel, note, kvUnknown?, kvMethod }
 *   attnArch ∈ { 'mha' | 'gqa' | 'mqa' | 'mla' | 'dsa' }（小写，供 UI 徽章直接映射）
 *   kvMethod ∈ { 'tensors' | 'config' }
 * ------------------------------------------------------------ */

import mha from './mha.js';
import mla from './mla.js';
import dsa from './dsa.js';
import { DSA_NOTE } from './dsa.js';
import {
  detectAttnArch,
  detectArchFromTensors,
  extractLayerTensors,
  num,
  GB,
} from './detect.js';

const REGISTRY = { mha, mla, dsa };

/**
 * 统一入口：计算 KV Cache。
 * @param {object} opts { config, tensors?, precision?, batch?, seq? }
 */
export function computeKV({ config = {}, tensors = null, precision = 'fp16', batch = 1, seq = 8192 } = {}) {
  // ── 主路径：张量形状推导（通用、免字段名猜测） ──
  if (Array.isArray(tensors) && tensors.length) {
    const parsed = extractLayerTensors(tensors);
    if (parsed) {
      const archName = detectArchFromTensors(parsed.attnNames);
      if (archName) {
        const mod = REGISTRY[archName];
        const r = mod.computeFromTensors({ ...parsed, config, batch, seq });
        if (r && Number.isFinite(r.vKV)) {
          // DSA 升级：张量识别为 MLA latent，但 config 暴露 index_head_dim → 补计 FP8 索引器
          if (archName === 'mla' && detectAttnArch(config) === 'dsa') {
            const Didx = num(config.index_head_dim ?? 128, 128);
            r.vKV += (batch * seq * parsed.L * Didx * 1) / GB;
            r.attnArch = 'dsa';
            r.formulaLabel = 'DSA 稀疏注意力（张量推导 latent + FP8 索引器）';
            r.note = DSA_NOTE;
          }
          return { ...r, kvMethod: 'tensors' };
        }
      }
    }
  }

  // ── 回退：config 超参推导（覆盖融合 QKV 等张量无法拆分的情形） ──
  const archName = detectAttnArch(config);
  const mod = REGISTRY[archName] || REGISTRY.mha;
  try {
    const r = mod.computeFromConfig({ config, batch, seq });
    return { ...r, kvMethod: 'config' };
  } catch {
    return {
      vKV: null,
      attnArch: archName,
      formulaLabel: '',
      note: '',
      kvUnknown: true,
      kvMethod: 'config',
    };
  }
}

export { detectAttnArch, detectArchFromTensors };
