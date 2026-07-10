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
