import glm52 from './profiles/glm_5_2.js';
import deepseekV4Pro from './profiles/deepseek_v4_pro.js';
import hy3 from './profiles/hy3.js';

const PROFILES = Object.freeze([glm52, deepseekV4Pro, hy3]);
const BY_MODEL_CLASS = new Map();
for (const profile of PROFILES) {
  for (const modelClassId of profile.modelClassIdentifiers) {
    const entries = BY_MODEL_CLASS.get(modelClassId) || [];
    entries.push(profile);
    BY_MODEL_CLASS.set(modelClassId, entries);
  }
}

export function profileCandidates(modelClassIds) {
  const unique = new Map();
  for (const modelClassId of modelClassIds) {
    for (const profile of BY_MODEL_CLASS.get(modelClassId) || []) unique.set(profile.id, profile);
  }
  return [...unique.values()];
}
