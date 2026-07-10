/* vram/kv/index.js — Verified Architecture Profile dispatcher.
 *
 * Production KV calculation is fail-closed: exact model-class aliases select
 * a curated profile candidate, then that profile validates its own config and
 * safetensors signature before its dedicated layout may run. There is no
 * tensor-name heuristic or generic MHA/MLA fallback.
 */

import { profileCandidates } from './catalog.js';
import { modelClassIdentifiers, unknownResult } from './profile-result.js';

export function computeKV({ config = {}, tensors = null, batch = 1, seq = 8192 } = {}) {
  const modelClassIds = modelClassIdentifiers(config);
  if (modelClassIds.length === 0) {
    return unknownResult({ code: 'missing_model_architecture', modelClassIds });
  }

  const candidates = profileCandidates(modelClassIds);
  if (candidates.length === 0) {
    return unknownResult({ code: 'unsupported_model_architecture', modelClassIds });
  }
  if (candidates.length > 1) {
    return unknownResult({
      code: 'conflicting_architecture_profiles',
      modelClassIds,
      status: 'conflict',
      details: { profileIds: candidates.map((candidate) => candidate.id) },
    });
  }

  const profile = candidates[0];
  const match = profile.match({ config, tensors });
  if (!match.matched) {
    return unknownResult({
      code: 'profile_signature_mismatch',
      modelClassIds,
      details: { profileId: profile.id, mismatches: match.mismatches },
    });
  }

  let result;
  try {
    result = profile.compute({ config, tensors, batch, seq });
  } catch (error) {
    return unknownResult({
      code: 'profile_calculation_out_of_range',
      modelClassIds,
      details: { profileId: profile.id, message: error.message },
    });
  }
  if (result.error) {
    return unknownResult({
      code: result.error,
      modelClassIds,
      details: result.details,
    });
  }
  return result;
}
