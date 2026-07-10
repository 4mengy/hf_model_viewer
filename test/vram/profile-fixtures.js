const GLM52_FULL_INDEXER_LAYERS = new Set([
  0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74,
]);

export function glm52Fixture({ fp8 = false } = {}) {
  const indexerTypes = Array.from({ length: 78 }, (_, layer) =>
    GLM52_FULL_INDEXER_LAYERS.has(layer) ? 'full' : 'shared');
  const tensors = [];
  const addLinear = (name, shape, scaleShape = null) => {
    tensors.push({ name, shape, dtype: fp8 ? 'F8_E4M3' : 'BF16' });
    if (fp8 && scaleShape) {
      tensors.push({ name: `${name}_scale_inv`, shape: scaleShape, dtype: 'F32' });
    }
  };
  for (let layer = 0; layer < 78; layer++) {
    const prefix = `model.layers.${layer}.self_attn`;
    addLinear(`${prefix}.kv_a_proj_with_mqa.weight`, [576, 6144], [5, 48]);
    addLinear(`${prefix}.kv_b_proj.weight`, [28672, 512], [224, 4]);
    if (GLM52_FULL_INDEXER_LAYERS.has(layer)) {
      addLinear(`${prefix}.indexer.wk.weight`, [128, 6144], [1, 48]);
      addLinear(`${prefix}.indexer.wq_b.weight`, [4096, 2048], [32, 16]);
      tensors.push({
        name: `${prefix}.indexer.weights_proj.weight`, shape: [32, 6144], dtype: 'BF16',
      });
    }
  }
  addLinear('model.layers.78.self_attn.kv_a_proj_with_mqa.weight', [576, 6144], [5, 48]);

  return {
    config: {
      architectures: ['GlmMoeDsaForCausalLM'],
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
      indexer_types: indexerTypes,
    },
    tensors,
  };
}

const DEEPSEEK_V4_PRO_COMPRESS_RATIOS = [
  128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
  4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
  4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
  4, 0,
];

export function deepseekV4ProFixture() {
  const tensors = [];
  for (let layer = 0; layer < 61; layer++) {
    const prefix = `layers.${layer}.attn`;
    const ratio = DEEPSEEK_V4_PRO_COMPRESS_RATIOS[layer];
    tensors.push({ name: `${prefix}.wkv.weight`, shape: [512, 7168], dtype: 'F8_E4M3' });
    if (ratio === 128) {
      tensors.push(
        { name: `${prefix}.compressor.ape`, shape: [128, 512], dtype: 'F32' },
        { name: `${prefix}.compressor.wkv.weight`, shape: [512, 7168], dtype: 'BF16' },
        { name: `${prefix}.compressor.wgate.weight`, shape: [512, 7168], dtype: 'BF16' },
      );
    } else {
      tensors.push(
        { name: `${prefix}.compressor.ape`, shape: [4, 1024], dtype: 'F32' },
        { name: `${prefix}.compressor.wkv.weight`, shape: [1024, 7168], dtype: 'BF16' },
        { name: `${prefix}.compressor.wgate.weight`, shape: [1024, 7168], dtype: 'BF16' },
        { name: `${prefix}.indexer.compressor.ape`, shape: [4, 256], dtype: 'F32' },
        { name: `${prefix}.indexer.compressor.wkv.weight`, shape: [256, 7168], dtype: 'BF16' },
        { name: `${prefix}.indexer.compressor.wgate.weight`, shape: [256, 7168], dtype: 'BF16' },
        { name: `${prefix}.indexer.weights_proj.weight`, shape: [64, 7168], dtype: 'BF16' },
      );
    }
  }
  tensors.push(
    { name: 'mtp.0.attn.wkv.weight', shape: [512, 7168], dtype: 'F8_E4M3' },
    { name: 'layers.0.ffn.experts.0.w1.weight', shape: [3072, 3584], dtype: 'I8' },
    { name: 'layers.0.ffn.experts.0.w1.scale', shape: [3072, 224], dtype: 'F8_E8M0' },
  );

  return {
    config: {
      architectures: ['DeepseekV4ForCausalLM'],
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
      compress_ratios: [...DEEPSEEK_V4_PRO_COMPRESS_RATIOS],
    },
    tensors,
  };
}

export function hy3Fixture() {
  const tensors = [];
  for (let layer = 0; layer <= 80; layer++) {
    const prefix = `model.layers.${layer}.self_attn`;
    tensors.push(
      { name: `${prefix}.q_proj.weight`, shape: [8192, 4096], dtype: 'BF16' },
      { name: `${prefix}.k_proj.weight`, shape: [1024, 4096], dtype: 'BF16' },
      { name: `${prefix}.v_proj.weight`, shape: [1024, 4096], dtype: 'BF16' },
      { name: `${prefix}.o_proj.weight`, shape: [4096, 8192], dtype: 'BF16' },
      { name: `${prefix}.q_norm.weight`, shape: [128], dtype: 'BF16' },
      { name: `${prefix}.k_norm.weight`, shape: [128], dtype: 'BF16' },
      { name: `model.layers.${layer}.input_layernorm.weight`, shape: [4096], dtype: 'BF16' },
      { name: `model.layers.${layer}.post_attention_layernorm.weight`, shape: [4096], dtype: 'BF16' },
    );
  }
  tensors.push(
    { name: 'model.layers.80.eh_proj.weight', shape: [4096, 8192], dtype: 'BF16' },
    { name: 'model.layers.80.enorm.weight', shape: [4096], dtype: 'BF16' },
    { name: 'model.layers.80.hnorm.weight', shape: [4096], dtype: 'BF16' },
    { name: 'model.layers.80.final_layernorm.weight', shape: [4096], dtype: 'BF16' },
  );
  return {
    config: {
      architectures: ['HYV3ForCausalLM'],
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
      rope_parameters: { rope_type: 'default', rope_theta: 11158840.0 },
      transformers_version: '5.6.0',
    },
    tensors,
  };
}
