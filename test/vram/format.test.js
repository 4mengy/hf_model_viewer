import test from 'node:test';
import assert from 'node:assert/strict';

import { fmtBytesAuto, fmtBytesGB } from '../../src/ui/format.js';

test('tree byte values are converted to GiB before display', () => {
  assert.equal(fmtBytesGB(1024 ** 3), '1.00 GB');
  assert.equal(fmtBytesGB(512 * 1024 ** 2), '0.50 GB');
});

test('overview byte values select a readable binary unit', () => {
  assert.equal(fmtBytesAuto(512), '512 B');
  assert.equal(fmtBytesAuto(1536), '1.50 KB');
  assert.equal(fmtBytesAuto(512 * 1024 ** 2), '512.00 MB');
  assert.equal(fmtBytesAuto(1024 ** 3), '1.00 GB');
  assert.equal(fmtBytesAuto(Number.NaN), '—');
});
