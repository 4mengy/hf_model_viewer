import test from 'node:test';
import assert from 'node:assert/strict';

import { estimateVRAM } from '../../src/vram/estimate.js';
import { glm52Fixture } from './profile-fixtures.js';

test('Complete VRAM Estimate stays unknown when KV Cache is unsupported', () => {
  const result = estimateVRAM(
    { architectures: ['UnknownForCausalLM'] },
    { totalParams: 100, baseParams: 100, expertParams: 0 },
    {
      precision: 'fp16',
      batch: 1,
      seq: 1024,
      tensors: [
        { name: 'model.embed_tokens.weight', shape: [10, 10], dtype: 'BF16' },
      ],
    },
  );

  assert.equal(result.complete, false);
  assert.equal(result.kvUnknown, true);
  assert.equal(result.vKV, null);
  assert.equal(result.vTotal, null);
  assert.equal(result.breakdown.kvGB, null);
  assert.equal(result.composition.some((item) => item.key === 'kv'), false);
  assert.equal(result.kvStatus, 'unsupported');
  assert.deepEqual(result.kvBuffers, []);
  assert.deepEqual(result.kvDiagnostic, {
    code: 'unsupported_model_architecture',
    modelClassIdentifiers: ['UnknownForCausalLM'],
  });
});

test('Verified Profile details flow through the complete VRAM Estimate and ignore weight precision', () => {
  const fixture = glm52Fixture();
  const tree = { totalParams: 1, baseParams: 1, expertParams: 0 };
  const fp16 = estimateVRAM(fixture.config, tree, {
    precision: 'fp16', batch: 1, seq: 1, tensors: fixture.tensors,
  });
  const int4 = estimateVRAM(fixture.config, tree, {
    precision: 'int4', batch: 1, seq: 1, tensors: fixture.tensors,
  });

  assert.equal(fp16.complete, true);
  assert.equal(fp16.kvUnknown, false);
  assert.equal(fp16.kvProfile.id, 'glm-5.2-semantic-bf16-v1');
  assert.equal(fp16.kvBuffers.reduce((sum, buffer) => sum + buffer.bytes, 0), 95_232);
  assert.equal(fp16.vKV, 95_232 / (1024 ** 3));
  assert.equal(int4.vKV, fp16.vKV);
  assert.ok(Number.isFinite(fp16.vTotal));
});

test('Complete VRAM Estimate forwards ragged sequence lengths to the Profile', () => {
  const fixture = glm52Fixture();
  const result = estimateVRAM(
    fixture.config,
    { totalParams: 1, baseParams: 1, expertParams: 0 },
    { tensors: fixture.tensors, sequenceLengths: [1, 2, 3] },
  );

  assert.equal(result.complete, true);
  assert.equal(result.kvBuffers.reduce((sum, buffer) => sum + buffer.bytes, 0), 6 * 95_232);
});

test('weight composition merges Tensor Name Patterns and sorts them by size', () => {
  const result = estimateVRAM(
    { architectures: ['UnknownForCausalLM'] },
    { totalParams: 10, baseParams: 10, expertParams: 0 },
    {
      tensors: [
        {
          name: 'language_model.model.layers.0.self_attn.k_norm.weight',
          shape: [2],
          dtype: 'BF16',
        },
        {
          name: 'language_model.model.layers.1.self_attn.k_norm.weight',
          shape: [3],
          dtype: 'BF16',
        },
        {
          name: 'language_model.model.layers.0.self_attn.q_proj.weight',
          shape: [4],
          dtype: 'BF16',
        },
      ],
    },
  );

  const weights = result.composition.filter((item) => item.group === 'weight');
  assert.deepEqual(
    weights.map((item) => [item.label, item.gb]),
    [
      ['language_model.model.layers.*.self_attn.k_norm.weight', 10 / (1024 ** 3)],
      ['language_model.model.layers.*.self_attn.q_proj.weight', 8 / (1024 ** 3)],
    ],
  );
});
