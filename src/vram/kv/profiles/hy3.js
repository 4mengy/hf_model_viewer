import { makeBuffer, verifiedResult } from '../profile-result.js';

const PROFILE = Object.freeze({
  id: 'hy3-instruct-semantic-bf16-v1',
  version: '1.0.0',
  label: 'Hunyuan 3 (Hy3)',
  modelClassIdentifier: 'HYV3ForCausalLM',
  layout: Object.freeze({ id: 'hy3-full-gqa-bf16-v1', version: '1.0.0' }),
  verification: 'verified',
  verifiedMetadataAliases: Object.freeze([
    { repoId: 'tencent/Hy3', revision: '716aa7241bd6d95896be4ebfc761162a9c4d49ef' },
    { repoId: 'tencent/Hy3-preview', revision: 'b53bd705bef15f0a9e52eade60a4353eaaa6c6b8' },
  ]),
  evidence: Object.freeze([
    {
      id: 'hy3-config',
      label: 'Hy3 official config',
      revision: '716aa7241bd6d95896be4ebfc761162a9c4d49ef',
      url: 'https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/config.json',
    },
    {
      id: 'hy3-model-card',
      label: 'Hy3 official model card',
      revision: '716aa7241bd6d95896be4ebfc761162a9c4d49ef',
      url: 'https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/README.md',
    },
    {
      id: 'hy3-vllm',
      label: 'vLLM HYV3 implementation',
      revision: 'd0009ddb0b96e95bcfae6038e9b8673bd2263058',
      url: 'https://github.com/vllm-project/vllm/blob/d0009ddb0b96e95bcfae6038e9b8673bd2263058/vllm/model_executor/models/hy_v3.py',
    },
    {
      id: 'hy3-preview-config',
      label: 'Metadata-equivalent Hy3 preview config alias',
      revision: 'b53bd705bef15f0a9e52eade60a4353eaaa6c6b8',
      url: 'https://huggingface.co/tencent/Hy3-preview/blob/b53bd705bef15f0a9e52eade60a4353eaaa6c6b8/config.json',
    },
  ]),
});

const REQUIRED_CONFIG = {
  model_type: 'hy_v3',
  use_cache: true,
  num_hidden_layers: 80,
  num_nextn_predict_layers: 1,
  hidden_size: 4096,
  num_attention_heads: 64,
  num_key_value_heads: 8,
  head_dim: 128,
  max_position_embeddings: 262144,
  qk_norm: true,
  vocab_size: 120832,
  intermediate_size: 13312,
  first_k_dense_replace: 1,
  num_experts: 192,
  num_experts_per_tok: 8,
  num_shared_experts: 1,
  expert_hidden_dim: 1536,
  moe_intermediate_size: 1536,
  router_scaling_factor: 2.826,
  tie_word_embeddings: false,
  transformers_version: '5.6.0',
};

function sameArray(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

function tensorMatches(byName, name, shape) {
  const tensor = byName.get(name);
  return tensor && tensor.dtype === 'BF16' && sameArray(tensor.shape, shape);
}

function match({ config, tensors }) {
  const mismatches = [];
  if (!sameArray(config.architectures, ['HYV3ForCausalLM'])) mismatches.push('architectures');
  for (const [key, expected] of Object.entries(REQUIRED_CONFIG)) {
    if (config[key] !== expected) mismatches.push(`config.${key}`);
  }
  if (
    config.rope_parameters?.rope_type !== 'default'
    || config.rope_parameters?.rope_theta !== 11158840.0
    || Object.keys(config.rope_parameters || {}).length !== 2
  ) {
    mismatches.push('config.rope_parameters');
  }
  for (const key of [
    'sliding_window', 'layer_types', 'attention_chunk_size', 'kv_lora_rank',
    'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim', 'index_head_dim', 'index_n_heads',
  ]) {
    if (config[key] != null) mismatches.push(`config.${key}`);
  }
  if (config.use_cla === true) mismatches.push('config.use_cla');
  if (config.cla_share_factor != null && config.cla_share_factor !== 1) {
    mismatches.push('config.cla_share_factor');
  }
  for (const key of [
    'quantization_config', 'kv_cache_dtype', 'cache_dtype',
    'kv_cache_config', 'cache_implementation',
  ]) {
    if (config[key] != null) mismatches.push(`config.${key}`);
  }
  if (config.num_attention_heads % config.num_key_value_heads !== 0) {
    mismatches.push('config.gqa_identity');
  }
  if (config.num_attention_heads * config.head_dim !== 8192) mismatches.push('config.q_projection_identity');
  if (config.num_key_value_heads * config.head_dim !== 1024) mismatches.push('config.kv_projection_identity');

  const byName = new Map((Array.isArray(tensors) ? tensors : []).map((tensor) => [tensor.name, tensor]));
  for (let layer = 0; layer <= 80; layer++) {
    const prefix = `model.layers.${layer}.self_attn`;
    if (!tensorMatches(byName, `${prefix}.q_proj.weight`, [8192, 4096])) {
      mismatches.push(`tensor.layers.${layer}.q_proj`);
    }
    if (!tensorMatches(byName, `${prefix}.k_proj.weight`, [1024, 4096])) {
      mismatches.push(`tensor.layers.${layer}.k_proj`);
    }
    if (!tensorMatches(byName, `${prefix}.v_proj.weight`, [1024, 4096])) {
      mismatches.push(`tensor.layers.${layer}.v_proj`);
    }
    if (!tensorMatches(byName, `${prefix}.o_proj.weight`, [4096, 8192])) {
      mismatches.push(`tensor.layers.${layer}.o_proj`);
    }
    if (!tensorMatches(byName, `${prefix}.q_norm.weight`, [128])) {
      mismatches.push(`tensor.layers.${layer}.q_norm`);
    }
    if (!tensorMatches(byName, `${prefix}.k_norm.weight`, [128])) {
      mismatches.push(`tensor.layers.${layer}.k_norm`);
    }
    for (const norm of ['input_layernorm', 'post_attention_layernorm']) {
      if (!tensorMatches(byName, `model.layers.${layer}.${norm}.weight`, [4096])) {
        mismatches.push(`tensor.layers.${layer}.${norm}`);
      }
    }
  }
  for (const [name, shape] of [
    ['model.layers.80.eh_proj.weight', [4096, 8192]],
    ['model.layers.80.enorm.weight', [4096]],
    ['model.layers.80.hnorm.weight', [4096]],
    ['model.layers.80.final_layernorm.weight', [4096]],
  ]) {
    if (!tensorMatches(byName, name, shape)) mismatches.push(`tensor.mtp.${name.split('.').at(-2)}`);
  }
  for (const name of byName.keys()) {
    const layer = name.match(/^model\.layers\.(\d+)\./)?.[1];
    if (layer != null && Number(layer) > 80) {
      mismatches.push(`tensor.unexpected_layer_${layer}`);
      break;
    }
  }
  return { matched: mismatches.length === 0, mismatches };
}

function compute({ batch, seq }) {
  if (!Number.isInteger(batch) || batch < 0 || !Number.isInteger(seq) || seq < 0 || seq > 262144) {
    return { error: 'profile_input_out_of_range', details: { batch, seq, maxContext: 262144 } };
  }
  const elementsPerBuffer = batch * seq * 80 * 8 * 128;
  const layerGroup = { label: 'all backbone layers', count: 80, range: [0, 79] };
  const buffers = [
    makeBuffer({
      id: 'main.key', label: 'Full-context normalized and RoPE key', layerGroup,
      elements: elementsPerBuffer, dtype: 'BF16', bytesPerElement: 2,
      formula: 'B × S × 80 × 8 × 128', evidenceIds: ['hy3-config', 'hy3-vllm'],
    }),
    makeBuffer({
      id: 'main.value', label: 'Full-context value', layerGroup,
      elements: elementsPerBuffer, dtype: 'BF16', bytesPerElement: 2,
      formula: 'B × S × 80 × 8 × 128', evidenceIds: ['hy3-config', 'hy3-vllm'],
    }),
  ];
  return verifiedResult({
    profile: PROFILE,
    buffers,
    note: 'The MTP layer is excluded as an optional speculative runtime. Hy3 preview is an explicitly audited metadata-equivalent layout alias.',
  });
}

export default Object.freeze({
  id: PROFILE.id,
  modelClassIdentifiers: Object.freeze(['HYV3ForCausalLM']),
  match,
  compute,
});
