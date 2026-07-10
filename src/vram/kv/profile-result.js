const GB = 1024 ** 3;

export function modelClassIdentifiers(config = {}) {
  return Array.isArray(config.architectures)
    ? config.architectures.filter((name) => typeof name === 'string' && name.length > 0)
    : [];
}

export function makeBuffer({
  id,
  label,
  layerGroup,
  elements,
  dtype,
  bytesPerElement,
  formula,
  evidenceIds = [],
}) {
  const bytes = elements * bytesPerElement;
  if (
    !Number.isSafeInteger(elements)
    || !Number.isSafeInteger(bytesPerElement)
    || !Number.isSafeInteger(bytes)
    || elements < 0
    || bytesPerElement <= 0
  ) {
    throw new Error(`Invalid KV buffer size for ${id}`);
  }
  return {
    id,
    label,
    layerGroup,
    elements,
    dtype,
    bytesPerElement,
    bytes,
    gb: bytes / GB,
    formula,
    evidenceIds,
  };
}

export function verifiedResult({ profile, buffers, note = '' }) {
  const totalBytes = buffers.reduce((sum, buffer) => sum + buffer.bytes, 0);
  if (!Number.isSafeInteger(totalBytes) || totalBytes < 0) {
    throw new Error(`Invalid KV total for ${profile.id}`);
  }
  return {
    status: 'verified',
    kvUnknown: false,
    vKV: totalBytes / GB,
    totalBytes,
    profile,
    buffers,
    diagnostic: null,
    note,
  };
}

export function unknownResult({ code, modelClassIds, details = null, status = 'unsupported' }) {
  return {
    status,
    kvUnknown: true,
    vKV: null,
    totalBytes: null,
    profile: null,
    buffers: [],
    diagnostic: {
      code,
      modelClassIdentifiers: modelClassIds,
      ...(details ? { details } : {}),
    },
    note: '',
  };
}
