import { makeBuffer, verifiedResult } from '../profile-result.js';

const COMPRESS_RATIOS = [
  128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
  4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
  4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
  4, 0,
];
const HCA_LAYERS = COMPRESS_RATIOS.slice(0, 61)
  .map((ratio, layer) => ({ ratio, layer }))
  .filter(({ ratio }) => ratio === 128)
  .map(({ layer }) => layer);
const CSA_LAYERS = COMPRESS_RATIOS.slice(0, 61)
  .map((ratio, layer) => ({ ratio, layer }))
  .filter(({ ratio }) => ratio === 4)
  .map(({ layer }) => layer);

const PROFILE = Object.freeze({
  id: 'deepseek-v4-pro-instruct-b5968e9',
  version: '1.0.0',
  label: 'DeepSeek V4 Pro',
  modelClassIdentifier: 'DeepseekV4ForCausalLM',
  layout: Object.freeze({ id: 'deepseek-v4-pro-csa-hca-bf16-v1', version: '1.0.0' }),
  verification: 'verified',
  evidence: Object.freeze([
    {
      id: 'deepseek-v4-pro-config',
      label: 'DeepSeek V4 Pro official config',
      revision: 'b5968e9190ef611bbf34a7229255be88a0e937c1',
      url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json',
    },
    {
      id: 'deepseek-v4-pro-reference',
      label: 'DeepSeek V4 Pro reference inference',
      revision: 'b5968e9190ef611bbf34a7229255be88a0e937c1',
      url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py',
    },
  ]),
});

const REQUIRED_CONFIG = {
  model_type: 'deepseek_v4',
  num_hidden_layers: 61,
  num_nextn_predict_layers: 1,
  hidden_size: 7168,
  num_attention_heads: 128,
  num_key_value_heads: 1,
  head_dim: 512,
  qk_rope_head_dim: 64,
  sliding_window: 128,
  max_position_embeddings: 1048576,
  use_cache: true,
  torch_dtype: 'bfloat16',
  index_n_heads: 64,
  index_head_dim: 128,
  index_topk: 1024,
  compress_rope_theta: 160000,
  expert_dtype: 'fp4',
};

function sameArray(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

function tensorMatches(byName, name, shape, dtype) {
  const tensor = byName.get(name);
  return tensor && tensor.dtype === dtype && sameArray(tensor.shape, shape);
}

function match({ config, tensors }) {
  const mismatches = [];
  if (!sameArray(config.architectures, ['DeepseekV4ForCausalLM'])) mismatches.push('architectures');
  for (const [key, expected] of Object.entries(REQUIRED_CONFIG)) {
    if (config[key] !== expected) mismatches.push(`config.${key}`);
  }
  if (!sameArray(config.compress_ratios, COMPRESS_RATIOS)) mismatches.push('config.compress_ratios');
  for (const key of [
    'attention_chunk_size', 'layer_types',
    'kv_cache_dtype', 'cache_dtype', 'kv_cache_config', 'cache_implementation',
  ]) {
    if (config[key] != null) mismatches.push(`config.${key}`);
  }
  if (config.use_cla === true) mismatches.push('config.use_cla');
  if (config.cla_share_factor != null && config.cla_share_factor !== 1) {
    mismatches.push('config.cla_share_factor');
  }

  const byName = new Map((Array.isArray(tensors) ? tensors : []).map((tensor) => [tensor.name, tensor]));
  for (let layer = 0; layer < 61; layer++) {
    const prefix = `layers.${layer}.attn`;
    const ratio = COMPRESS_RATIOS[layer];
    if (!tensorMatches(byName, `${prefix}.wkv.weight`, [512, 7168], 'F8_E4M3')) {
      mismatches.push(`tensor.layers.${layer}.wkv`);
    }
    if (ratio === 128) {
      if (!tensorMatches(byName, `${prefix}.compressor.ape`, [128, 512], 'F32')) {
        mismatches.push(`tensor.layers.${layer}.compressor_ape`);
      }
      for (const name of ['wkv', 'wgate']) {
        if (!tensorMatches(byName, `${prefix}.compressor.${name}.weight`, [512, 7168], 'BF16')) {
          mismatches.push(`tensor.layers.${layer}.compressor_${name}`);
        }
      }
      if ([...byName.keys()].some((name) => name.startsWith(`${prefix}.indexer.`))) {
        mismatches.push(`tensor.layers.${layer}.unexpected_indexer`);
      }
    } else {
      if (!tensorMatches(byName, `${prefix}.compressor.ape`, [4, 1024], 'F32')) {
        mismatches.push(`tensor.layers.${layer}.compressor_ape`);
      }
      for (const name of ['wkv', 'wgate']) {
        if (!tensorMatches(byName, `${prefix}.compressor.${name}.weight`, [1024, 7168], 'BF16')) {
          mismatches.push(`tensor.layers.${layer}.compressor_${name}`);
        }
        if (!tensorMatches(byName, `${prefix}.indexer.compressor.${name}.weight`, [256, 7168], 'BF16')) {
          mismatches.push(`tensor.layers.${layer}.indexer_compressor_${name}`);
        }
      }
      if (!tensorMatches(byName, `${prefix}.indexer.compressor.ape`, [4, 256], 'F32')) {
        mismatches.push(`tensor.layers.${layer}.indexer_compressor_ape`);
      }
      if (!tensorMatches(byName, `${prefix}.indexer.weights_proj.weight`, [64, 7168], 'BF16')) {
        mismatches.push(`tensor.layers.${layer}.indexer_weights_proj`);
      }
    }
  }
  if (!tensorMatches(byName, 'mtp.0.attn.wkv.weight', [512, 7168], 'F8_E4M3')) {
    mismatches.push('tensor.mtp.0');
  }
  if ([...byName.keys()].some((name) => /^mtp\.0\.attn\.(?:compressor|indexer)\./.test(name))) {
    mismatches.push('tensor.mtp.0.unexpected_cache_topology');
  }
  if (!tensorMatches(
    byName,
    'layers.0.ffn.experts.0.w1.weight',
    [3072, 3584],
    'I8',
  )) {
    mismatches.push('tensor.fp4_expert_identity');
  }
  if (!tensorMatches(
    byName,
    'layers.0.ffn.experts.0.w1.scale',
    [3072, 224],
    'F8_E8M0',
  )) {
    mismatches.push('tensor.fp4_expert_scale_identity');
  }
  for (const name of byName.keys()) {
    const layer = name.match(/^layers\.(\d+)\./)?.[1];
    if (layer != null && Number(layer) > 60) {
      mismatches.push(`tensor.unexpected_layer_${layer}`);
      break;
    }
  }
  if ([...byName.keys()].some((name) => /^mtp\.(?!0\.)\d+\./.test(name))) {
    mismatches.push('tensor.unexpected_mtp_layer');
  }
  return { matched: mismatches.length === 0, mismatches };
}

function layerGroup(label, indices) {
  return { label, count: indices.length, indices };
}

function compute({ batch, seq }) {
  if (!Number.isInteger(batch) || batch < 0 || !Number.isInteger(seq) || seq < 0 || seq > 1048576) {
    return { error: 'profile_input_out_of_range', details: { batch, seq, maxContext: 1048576 } };
  }

  const local = Math.min(seq, 128);
  const compressed128 = Math.floor(seq / 128);
  const remainder128 = seq % 128;
  const compressed4 = Math.floor(seq / 4);
  const active4 = seq < 4 ? seq : 4 + (seq % 4);
  const hca = layerGroup('HCA layers (compression ratio 128)', HCA_LAYERS);
  const csa = layerGroup('CSA layers (compression ratio 4)', CSA_LAYERS);
  const buffers = [
    makeBuffer({
      id: 'hca-kv', label: 'HCA local and compressed KV', layerGroup: hca,
      elements: batch * 31 * 512 * (local + compressed128),
      dtype: 'BF16', bytesPerElement: 2,
      formula: 'B × 31 × 512 × (min(S,128) + floor(S/128))',
      evidenceIds: ['deepseek-v4-pro-config', 'deepseek-v4-pro-reference'],
    }),
    makeBuffer({
      id: 'hca-kv-state', label: 'HCA compressor KV state', layerGroup: hca,
      elements: batch * 31 * 512 * remainder128,
      dtype: 'F32', bytesPerElement: 4,
      formula: 'B × 31 × 512 × (S mod 128)', evidenceIds: ['deepseek-v4-pro-reference'],
    }),
    makeBuffer({
      id: 'hca-score-state', label: 'HCA compressor score state', layerGroup: hca,
      elements: batch * 31 * 512 * remainder128,
      dtype: 'F32', bytesPerElement: 4,
      formula: 'B × 31 × 512 × (S mod 128)', evidenceIds: ['deepseek-v4-pro-reference'],
    }),
    makeBuffer({
      id: 'csa-kv', label: 'CSA local and compressed KV', layerGroup: csa,
      elements: batch * 30 * 512 * (local + compressed4),
      dtype: 'BF16', bytesPerElement: 2,
      formula: 'B × 30 × 512 × (min(S,128) + floor(S/4))',
      evidenceIds: ['deepseek-v4-pro-config', 'deepseek-v4-pro-reference'],
    }),
    makeBuffer({
      id: 'csa-indexer-kv', label: 'CSA compressed indexer KV', layerGroup: csa,
      elements: batch * 30 * 128 * compressed4,
      dtype: 'BF16', bytesPerElement: 2,
      formula: 'B × 30 × 128 × floor(S/4)',
      evidenceIds: ['deepseek-v4-pro-config', 'deepseek-v4-pro-reference'],
    }),
    makeBuffer({
      id: 'csa-kv-state', label: 'CSA compressor KV state', layerGroup: csa,
      elements: batch * 30 * 1024 * active4,
      dtype: 'F32', bytesPerElement: 4,
      formula: 'B × 30 × 1024 × A4(S)', evidenceIds: ['deepseek-v4-pro-reference'],
    }),
    makeBuffer({
      id: 'csa-score-state', label: 'CSA compressor score state', layerGroup: csa,
      elements: batch * 30 * 1024 * active4,
      dtype: 'F32', bytesPerElement: 4,
      formula: 'B × 30 × 1024 × A4(S)', evidenceIds: ['deepseek-v4-pro-reference'],
    }),
    makeBuffer({
      id: 'csa-indexer-kv-state', label: 'CSA indexer compressor KV state', layerGroup: csa,
      elements: batch * 30 * 256 * active4,
      dtype: 'F32', bytesPerElement: 4,
      formula: 'B × 30 × 256 × A4(S)', evidenceIds: ['deepseek-v4-pro-reference'],
    }),
    makeBuffer({
      id: 'csa-indexer-score-state', label: 'CSA indexer compressor score state', layerGroup: csa,
      elements: batch * 30 * 256 * active4,
      dtype: 'F32', bytesPerElement: 4,
      formula: 'B × 30 × 256 × A4(S)', evidenceIds: ['deepseek-v4-pro-reference'],
    }),
  ];

  return verifiedResult({
    profile: PROFILE,
    buffers,
    note: 'MTP and runtime capacity are excluded; index_topk limits reads, not cache capacity.',
  });
}

export default Object.freeze({
  id: PROFILE.id,
  modelClassIdentifiers: Object.freeze(['DeepseekV4ForCausalLM']),
  match,
  compute,
});
