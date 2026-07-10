import { makeBuffer, verifiedResult } from '../profile-result.js';

const FULL_INDEXER_LAYERS = [
  0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
];
const FULL_INDEXER_SET = new Set(FULL_INDEXER_LAYERS);
const INDEXER_TYPES = Array.from({ length: 78 }, (_, layer) =>
  FULL_INDEXER_SET.has(layer) ? 'full' : 'shared');

const PROFILE = Object.freeze({
  id: 'glm-5.2-semantic-bf16-v1',
  version: '1.0.0',
  label: 'GLM 5.2',
  modelClassIdentifier: 'GlmMoeDsaForCausalLM',
  layout: Object.freeze({ id: 'glm-5.2-indexshare-bf16-v1', version: '1.0.0' }),
  verification: 'verified',
  verifiedMetadataAliases: Object.freeze([
    { repoId: 'zai-org/GLM-5.2', revision: 'b4734de4facf877f85769a911abafc5283eab3d9' },
    { repoId: 'zai-org/GLM-5.2-FP8', revision: 'ba978f7d347eaf65d22f1a86833408afdb953541' },
  ]),
  evidence: Object.freeze([
    {
      id: 'glm52-config',
      label: 'GLM 5.2 official config',
      revision: 'b4734de4facf877f85769a911abafc5283eab3d9',
      url: 'https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/config.json',
    },
    {
      id: 'glm52-fp8-checkpoint',
      label: 'GLM 5.2 FP8 official checkpoint index',
      revision: 'ba978f7d347eaf65d22f1a86833408afdb953541',
      url: 'https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/model.safetensors.index.json',
    },
    {
      id: 'glm52-transformers',
      label: 'Transformers GLM MoE DSA implementation',
      revision: 'e0e7504bca2bfd1b85bb0eedb148f7b250226f06',
      url: 'https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py',
    },
  ]),
});

const REQUIRED_CONFIG = {
  model_type: 'glm_moe_dsa',
  dtype: 'bfloat16',
  use_cache: true,
  num_hidden_layers: 78,
  hidden_size: 6144,
  num_attention_heads: 64,
  num_key_value_heads: 64,
  q_lora_rank: 2048,
  kv_lora_rank: 512,
  qk_head_dim: 256,
  qk_nope_head_dim: 192,
  qk_rope_head_dim: 64,
  v_head_dim: 256,
  index_n_heads: 32,
  index_head_dim: 128,
  index_topk: 2048,
  index_topk_freq: 4,
  index_skip_topk_offset: 3,
  index_topk_pattern: null,
  indexer_rope_interleave: true,
  index_share_for_mtp_iteration: true,
  max_position_embeddings: 1048576,
  num_nextn_predict_layers: 1,
};

function sameArray(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

function tensorMatches(byName, name, shape, dtype) {
  const tensor = byName.get(name);
  return tensor
    && sameArray(tensor.shape, shape)
    && tensor.dtype === dtype;
}

function match({ config, tensors }) {
  const mismatches = [];
  if (!sameArray(config.architectures, ['GlmMoeDsaForCausalLM'])) mismatches.push('architectures');
  for (const [key, expected] of Object.entries(REQUIRED_CONFIG)) {
    if (config[key] !== expected) mismatches.push(`config.${key}`);
  }
  if (!sameArray(config.indexer_types, INDEXER_TYPES)) mismatches.push('config.indexer_types');
  for (const key of [
    'sliding_window', 'attention_chunk_size', 'layer_types',
    'kv_cache_dtype', 'cache_dtype', 'kv_cache_config', 'cache_implementation',
  ]) {
    if (config[key] != null) mismatches.push(`config.${key}`);
  }
  if (config.use_cla === true) mismatches.push('config.use_cla');
  if (config.cla_share_factor != null && config.cla_share_factor !== 1) {
    mismatches.push('config.cla_share_factor');
  }
  if (config.qk_head_dim !== config.qk_nope_head_dim + config.qk_rope_head_dim) {
    mismatches.push('config.qk_head_dim_identity');
  }

  const byName = new Map((Array.isArray(tensors) ? tensors : []).map((tensor) => [tensor.name, tensor]));
  const firstDtype = byName.get('model.layers.0.self_attn.kv_a_proj_with_mqa.weight')?.dtype;
  const checkpointVariant = firstDtype === 'BF16' ? 'bf16' : firstDtype === 'F8_E4M3' ? 'fp8' : null;
  if (!checkpointVariant) mismatches.push('tensor.checkpoint_dtype_alias');

  const checkLinear = (name, shape, scaleShape, mismatchKey) => {
    const dtype = checkpointVariant === 'fp8' ? 'F8_E4M3' : 'BF16';
    if (!tensorMatches(byName, name, shape, dtype)) mismatches.push(mismatchKey);
    const scaleName = `${name}_scale_inv`;
    if (checkpointVariant === 'fp8') {
      if (!tensorMatches(byName, scaleName, scaleShape, 'F32')) {
        mismatches.push(`${mismatchKey}_scale`);
      }
    } else if (byName.has(scaleName)) {
      mismatches.push(`${mismatchKey}_unexpected_scale`);
    }
  };

  for (let layer = 0; layer < 78; layer++) {
    const prefix = `model.layers.${layer}.self_attn`;
    checkLinear(
      `${prefix}.kv_a_proj_with_mqa.weight`, [576, 6144], [5, 48],
      `tensor.layers.${layer}.kv_a`,
    );
    checkLinear(
      `${prefix}.kv_b_proj.weight`, [28672, 512], [224, 4],
      `tensor.layers.${layer}.kv_b`,
    );
    const indexerNames = [
      [`${prefix}.indexer.wk.weight`, [128, 6144], [1, 48], 'indexer_wk'],
      [`${prefix}.indexer.wq_b.weight`, [4096, 2048], [32, 16], 'indexer_wq_b'],
    ];
    for (const [name, shape, scaleShape, suffix] of indexerNames) {
      if (FULL_INDEXER_SET.has(layer)) {
        checkLinear(name, shape, scaleShape, `tensor.layers.${layer}.${suffix}`);
      } else if (byName.has(name) || byName.has(`${name}_scale_inv`)) {
        mismatches.push(`tensor.layers.${layer}.unexpected_${suffix}`);
      }
    }
    const weightsProj = `${prefix}.indexer.weights_proj.weight`;
    if (FULL_INDEXER_SET.has(layer)) {
      if (!tensorMatches(byName, weightsProj, [32, 6144], 'BF16')) {
        mismatches.push(`tensor.layers.${layer}.indexer_weights_proj`);
      }
      if (byName.has(`${weightsProj}_scale_inv`)) {
        mismatches.push(`tensor.layers.${layer}.indexer_weights_proj_unexpected_scale`);
      }
    } else if (byName.has(weightsProj) || byName.has(`${weightsProj}_scale_inv`)) {
      mismatches.push(`tensor.layers.${layer}.unexpected_indexer_weights_proj`);
    }
  }
  checkLinear(
    'model.layers.78.self_attn.kv_a_proj_with_mqa.weight',
    [576, 6144],
    [5, 48],
    'tensor.mtp.0',
  );
  for (const name of byName.keys()) {
    const layer = name.match(/^model\.layers\.(\d+)\./)?.[1];
    if (layer != null && Number(layer) > 78) {
      mismatches.push(`tensor.unexpected_layer_${layer}`);
      break;
    }
  }
  return { matched: mismatches.length === 0, mismatches };
}

function compute({ batch, seq }) {
  if (!Number.isInteger(batch) || batch < 0 || !Number.isInteger(seq) || seq < 0 || seq > 1048576) {
    return { error: 'profile_input_out_of_range', details: { batch, seq, maxContext: 1048576 } };
  }

  const tokenCount = batch * seq;
  const buffers = [
    makeBuffer({
      id: 'mla-latent',
      label: 'MLA compressed latent',
      layerGroup: { label: 'all backbone layers', count: 78, range: [0, 77] },
      elements: tokenCount * 78 * 512,
      dtype: 'BF16',
      bytesPerElement: 2,
      formula: 'B × S × 78 × 512',
      evidenceIds: ['glm52-config', 'glm52-transformers'],
    }),
    makeBuffer({
      id: 'mla-rope-key',
      label: 'MLA RoPE key',
      layerGroup: { label: 'all backbone layers', count: 78, range: [0, 77] },
      elements: tokenCount * 78 * 64,
      dtype: 'BF16',
      bytesPerElement: 2,
      formula: 'B × S × 78 × 64',
      evidenceIds: ['glm52-config', 'glm52-transformers'],
    }),
    makeBuffer({
      id: 'indexer-key',
      label: 'IndexShare full-layer indexer key',
      layerGroup: { label: 'full indexer layers', count: 21, indices: FULL_INDEXER_LAYERS },
      elements: tokenCount * 21 * 128,
      dtype: 'BF16',
      bytesPerElement: 2,
      formula: 'B × S × 21 × 128',
      evidenceIds: ['glm52-config', 'glm52-transformers'],
    }),
  ];

  return verifiedResult({
    profile: PROFILE,
    buffers,
    note: 'All history is retained; index_topk limits reads, not cache capacity. BF16 and FP8 weight checkpoints are separately signature-verified aliases of the same BF16 KV semantics.',
  });
}

export default Object.freeze({
  id: PROFILE.id,
  modelClassIdentifiers: Object.freeze(['GlmMoeDsaForCausalLM']),
  match,
  compute,
});
