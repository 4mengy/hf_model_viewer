import test from 'node:test';
import assert from 'node:assert/strict';

import { computeKV } from '../../src/vram/kv/index.js';
import { deepseekV4ProFixture, glm52Fixture, hy3Fixture } from './profile-fixtures.js';

test('unsupported Model Class Identifier fails closed without a heuristic estimate', () => {
  const result = computeKV({
    config: {
      architectures: ['UnknownForCausalLM'],
      num_hidden_layers: 2,
      num_attention_heads: 4,
      num_key_value_heads: 2,
      hidden_size: 16,
    },
    tensors: [
      { name: 'model.layers.0.self_attn.k_proj.weight', shape: [8, 16], dtype: 'BF16' },
      { name: 'model.layers.0.self_attn.v_proj.weight', shape: [8, 16], dtype: 'BF16' },
      { name: 'model.layers.1.self_attn.k_proj.weight', shape: [8, 16], dtype: 'BF16' },
      { name: 'model.layers.1.self_attn.v_proj.weight', shape: [8, 16], dtype: 'BF16' },
    ],
    batch: 1,
    seq: 1024,
  });

  assert.equal(result.status, 'unsupported');
  assert.equal(result.kvUnknown, true);
  assert.equal(result.vKV, null);
  assert.equal(result.profile, null);
  assert.deepEqual(result.buffers, []);
  assert.deepEqual(result.diagnostic, {
    code: 'unsupported_model_architecture',
    modelClassIdentifiers: ['UnknownForCausalLM'],
  });
});

test('missing Model Class Identifier fails closed instead of using config fallback', () => {
  const result = computeKV({
    config: {
      num_hidden_layers: 2,
      num_attention_heads: 4,
      num_key_value_heads: 2,
      hidden_size: 16,
    },
    batch: 1,
    seq: 1024,
  });

  assert.equal(result.status, 'unsupported');
  assert.equal(result.vKV, null);
  assert.deepEqual(result.diagnostic, {
    code: 'missing_model_architecture',
    modelClassIdentifiers: [],
  });
});

test('GLM 5.2 returns its verified IndexShare buffer breakdown', () => {
  const fixture = glm52Fixture();
  const result = computeKV({ ...fixture, batch: 1, seq: 1 });

  assert.equal(result.status, 'verified');
  assert.equal(result.kvUnknown, false);
  assert.equal(result.profile.id, 'glm-5.2-semantic-bf16-v1');
  assert.equal(result.profile.layout.id, 'glm-5.2-indexshare-bf16-v1');
  assert.equal(result.totalBytes, 95_232);
  assert.deepEqual(
    result.buffers.map(({ id, bytes, dtype }) => ({ id, bytes, dtype })),
    [
      { id: 'mla-latent', bytes: 79_872, dtype: 'BF16' },
      { id: 'mla-rope-key', bytes: 9_984, dtype: 'BF16' },
      { id: 'indexer-key', bytes: 5_376, dtype: 'BF16' },
    ],
  );
});

test('GLM 5.2 accepts only the explicitly audited FP8 checkpoint signature alias', () => {
  const fixture = glm52Fixture({ fp8: true });
  const result = computeKV({ ...fixture, batch: 1, seq: 1 });

  assert.equal(result.status, 'verified');
  assert.equal(result.profile.id, 'glm-5.2-semantic-bf16-v1');
  assert.equal(result.totalBytes, 95_232);

  fixture.tensors = fixture.tensors.filter(
    (tensor) => tensor.name !== 'model.layers.34.self_attn.kv_b_proj.weight_scale_inv',
  );
  const mismatch = computeKV({ ...fixture, batch: 1, seq: 1 });
  assert.equal(mismatch.status, 'unsupported');
  assert.equal(mismatch.vKV, null);
  assert.ok(mismatch.diagnostic.details.mismatches.includes('tensor.layers.34.kv_b_scale'));
});

test('DeepSeek V4 Pro returns its verified HCA and CSA stateful buffer breakdown', () => {
  const fixture = deepseekV4ProFixture();
  const result = computeKV({ ...fixture, batch: 1, seq: 128 });

  assert.equal(result.status, 'verified');
  assert.equal(result.profile.id, 'deepseek-v4-pro-instruct-b5968e9');
  assert.equal(result.profile.layout.id, 'deepseek-v4-pro-csa-hca-bf16-v1');
  assert.equal(result.totalBytes, 10_484_736);
  assert.deepEqual(
    result.buffers.map(({ id, bytes, dtype }) => ({ id, bytes, dtype })),
    [
      { id: 'hca-kv', bytes: 4_094_976, dtype: 'BF16' },
      { id: 'hca-kv-state', bytes: 0, dtype: 'F32' },
      { id: 'hca-score-state', bytes: 0, dtype: 'F32' },
      { id: 'csa-kv', bytes: 4_915_200, dtype: 'BF16' },
      { id: 'csa-indexer-kv', bytes: 245_760, dtype: 'BF16' },
      { id: 'csa-kv-state', bytes: 491_520, dtype: 'F32' },
      { id: 'csa-score-state', bytes: 491_520, dtype: 'F32' },
      { id: 'csa-indexer-kv-state', bytes: 122_880, dtype: 'F32' },
      { id: 'csa-indexer-score-state', bytes: 122_880, dtype: 'F32' },
    ],
  );
});

test('Hunyuan 3 returns its verified full-context GQA buffer breakdown', () => {
  const fixture = hy3Fixture();
  const result = computeKV({ ...fixture, batch: 1, seq: 1 });

  assert.equal(result.status, 'verified');
  assert.equal(result.profile.id, 'hy3-instruct-semantic-bf16-v1');
  assert.equal(result.profile.layout.id, 'hy3-full-gqa-bf16-v1');
  assert.equal(result.totalBytes, 327_680);
  assert.deepEqual(
    result.buffers.map(({ id, bytes, dtype }) => ({ id, bytes, dtype })),
    [
      { id: 'main.key', bytes: 163_840, dtype: 'BF16' },
      { id: 'main.value', bytes: 163_840, dtype: 'BF16' },
    ],
  );
});

test('GLM 5.2 keeps full history across the index_topk boundary', () => {
  const fixture = glm52Fixture();
  const atTopK = computeKV({ ...fixture, batch: 1, seq: 2048 });
  const afterTopK = computeKV({ ...fixture, batch: 1, seq: 2049 });
  const atMax = computeKV({ ...fixture, batch: 1, seq: 1048576 });
  const beyondMax = computeKV({ ...fixture, batch: 1, seq: 1048577 });

  assert.equal(atTopK.totalBytes, 195_035_136);
  assert.equal(afterTopK.totalBytes, 195_130_368);
  assert.equal(afterTopK.totalBytes - atTopK.totalBytes, 95_232);
  assert.equal(atMax.totalBytes, 99_857_989_632);
  assert.equal(beyondMax.status, 'unsupported');
  assert.equal(beyondMax.diagnostic.code, 'profile_input_out_of_range');
});

test('DeepSeek V4 Pro compressor state transitions match verified boundaries', () => {
  const fixture = deepseekV4ProFixture();
  const vectors = new Map([
    [3, 1_489_920],
    [4, 2_024_960],
    [7, 3_514_880],
    [8, 2_821_120],
    [127, 27_399_680],
    [128, 10_484_736],
    [129, 10_918_912],
    [1048576, 10_335_600_640],
  ]);

  for (const [seq, expectedBytes] of vectors) {
    assert.equal(computeKV({ ...fixture, batch: 1, seq }).totalBytes, expectedBytes, `S=${seq}`);
  }
});

test('DeepSeek V4 Pro fails closed when its audited FP4 checkpoint identity drifts', () => {
  const fixture = deepseekV4ProFixture();
  fixture.tensors = fixture.tensors.filter(
    (tensor) => tensor.name !== 'layers.0.ffn.experts.0.w1.scale',
  );

  const result = computeKV({ ...fixture, batch: 1, seq: 128 });

  assert.equal(result.status, 'unsupported');
  assert.equal(result.vKV, null);
  assert.equal(result.diagnostic.code, 'profile_signature_mismatch');
  assert.ok(result.diagnostic.details.mismatches.includes('tensor.fp4_expert_scale_identity'));
});

test('Hunyuan 3 full-context GQA reaches 80 GiB at its verified maximum', () => {
  const fixture = hy3Fixture();
  const at4K = computeKV({ ...fixture, batch: 1, seq: 4096 });
  const atMax = computeKV({ ...fixture, batch: 1, seq: 262144 });
  const beyondMax = computeKV({ ...fixture, batch: 1, seq: 262145 });

  assert.equal(at4K.totalBytes, 1_342_177_280);
  assert.equal(atMax.totalBytes, 85_899_345_920);
  assert.equal(beyondMax.diagnostic.code, 'profile_input_out_of_range');
});

test('Hunyuan 3 fails closed when an audited config predicate drifts', () => {
  const fixture = hy3Fixture();
  fixture.config.router_scaling_factor = 3;

  const result = computeKV({ ...fixture, batch: 1, seq: 1 });

  assert.equal(result.status, 'unsupported');
  assert.equal(result.vKV, null);
  assert.equal(result.diagnostic.code, 'profile_signature_mismatch');
  assert.ok(result.diagnostic.details.mismatches.includes('config.router_scaling_factor'));
});

test('Hunyuan 3 fails closed when a cache-related tensor signature drifts', () => {
  const fixture = hy3Fixture();
  fixture.tensors = fixture.tensors.filter(
    (tensor) => tensor.name !== 'model.layers.37.self_attn.o_proj.weight',
  );

  const result = computeKV({ ...fixture, batch: 1, seq: 1 });

  assert.equal(result.status, 'unsupported');
  assert.equal(result.vKV, null);
  assert.equal(result.diagnostic.code, 'profile_signature_mismatch');
  assert.ok(result.diagnostic.details.mismatches.includes('tensor.layers.37.o_proj'));
});

test('shared buffer arithmetic fails closed instead of returning an unsafe integer', () => {
  const fixture = hy3Fixture();

  const result = computeKV({ ...fixture, batch: Number.MAX_SAFE_INTEGER, seq: 262144 });

  assert.equal(result.status, 'unsupported');
  assert.equal(result.vKV, null);
  assert.equal(result.diagnostic.code, 'profile_calculation_out_of_range');
});

test('verified profiles reject unreviewed extra backbone layers', () => {
  const cases = [
    {
      fixture: glm52Fixture(),
      tensor: {
        name: 'model.layers.79.self_attn.kv_a_proj_with_mqa.weight',
        shape: [576, 6144], dtype: 'BF16',
      },
      mismatch: 'tensor.unexpected_layer_79',
    },
    {
      fixture: deepseekV4ProFixture(),
      tensor: { name: 'layers.61.attn.wkv.weight', shape: [512, 7168], dtype: 'F8_E4M3' },
      mismatch: 'tensor.unexpected_layer_61',
    },
    {
      fixture: hy3Fixture(),
      tensor: {
        name: 'model.layers.82.self_attn.k_proj.weight',
        shape: [1024, 4096], dtype: 'BF16',
      },
      mismatch: 'tensor.unexpected_layer_82',
    },
  ];

  for (const { fixture, tensor, mismatch } of cases) {
    fixture.tensors.push(tensor);
    const result = computeKV({ ...fixture, batch: 1, seq: 1 });
    assert.equal(result.status, 'unsupported', tensor.name);
    assert.ok(result.diagnostic.details.mismatches.includes(mismatch), tensor.name);
  }
});

test('verified profiles reject unreviewed KV cache dtype overrides', () => {
  for (const fixture of [glm52Fixture(), deepseekV4ProFixture(), hy3Fixture()]) {
    fixture.config.kv_cache_dtype = 'fp8';
    const result = computeKV({ ...fixture, batch: 1, seq: 128 });
    assert.equal(result.status, 'unsupported', fixture.config.architectures[0]);
    assert.ok(
      result.diagnostic.details.mismatches.includes('config.kv_cache_dtype'),
      fixture.config.architectures[0],
    );
  }
});
