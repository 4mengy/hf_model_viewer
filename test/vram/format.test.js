import test from 'node:test';
import assert from 'node:assert/strict';

import { fmtBytesGB } from '../../src/ui/format.js';

test('tree byte values are converted to GiB before display', () => {
  assert.equal(fmtBytesGB(1024 ** 3), '1.00 GB');
  assert.equal(fmtBytesGB(512 * 1024 ** 2), '0.50 GB');
});
