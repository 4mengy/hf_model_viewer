/* engine/index.js — 解析编排（analyze）
 * ------------------------------------------------------------
 * 闭环：
 *   config.json -> 全局超参
 *   model.safetensors.index.json / 单文件 -> 分片清单
 *   并行 HTTP Range -> 各分片头部 JSON -> 扁平张量列表
 * ------------------------------------------------------------ */

import { makeNet } from '../platform/net.js';
import { readSafetensorsHeader } from './safetensors.js';
import { mapLimit } from './util.js';

const CONFIG_URL = (repo) => `https://huggingface.co/${repo}/resolve/main/config.json`;
const INDEX_URL = (repo) => `https://huggingface.co/${repo}/resolve/main/model.safetensors.index.json`;
const SHARD_BASE = (repo) => `https://huggingface.co/${repo}/resolve/main`;

const NOT_SAFETENSORS =
  '该模型未提供 Safetensors 格式，无法进行远程 Range 碎片化解析，请更换现代大模型仓库。';

/**
 * 解析一个 Hugging Face 仓库，返回 { repoId, config, tensors, shardFiles, shardCount }。
 * @param {string} repoId 形如 org/repo
 * @param {object} [opts]
 * @param {string} [opts.token] 可选 Hugging Face Access Token（受限/私有模型）
 * @param {(done:number,total:number,file:string)=>void} [opts.onShard] 分片进度回调
 */
export async function analyze(repoId, { token, onShard } = {}) {
  if (!repoId || !/^[\w.-]+\/[\w.-]+/.test(repoId)) {
    throw new Error('请输入合法的仓库 ID，形如 org/repo');
  }

  const net = makeNet();
  const auth = token ? { Authorization: `Bearer ${token}` } : {};

  // 1) config.json
  let config;
  try {
    config = JSON.parse(await net.text(CONFIG_URL(repoId), auth));
  } catch (e) {
    throw new Error(`无法获取 config.json（仓库是否存在或网络异常）：${e.message}`);
  }

  // 2) 分片发现：优先 index.json，回退单文件 model.safetensors
  const base = SHARD_BASE(repoId);
  let shardFiles = [];
  try {
    const idx = JSON.parse(await net.text(INDEX_URL(repoId), auth));
    shardFiles = [...new Set(Object.values(idx.weight_map))];
    if (idx.metadata) config.__hf_index_meta = idx.metadata;
  } catch {
    try {
      await readSafetensorsHeader(net, base, 'model.safetensors', auth);
      shardFiles = ['model.safetensors'];
    } catch {
      throw new Error(NOT_SAFETENSORS);
    }
  }

  if (!shardFiles.length) throw new Error(NOT_SAFETENSORS);

  // 3) 并行拉取所有分片头部（并发上限 16，满足超大 MoE <3s）
  const headersMap = {};
  let done = 0;
  await mapLimit(shardFiles, 16, async (file) => {
    headersMap[file] = await readSafetensorsHeader(net, base, file, auth);
    done += 1;
    onShard?.(done, shardFiles.length, file);
  });

  // 4) 合并为扁平张量列表
  const tensors = [];
  for (const file of shardFiles) {
    const h = headersMap[file];
    for (const [name, meta] of Object.entries(h)) {
      if (name === '__metadata__') continue;
      tensors.push({ name, dtype: meta.dtype, shape: meta.shape, shard: file });
    }
  }

  return { repoId, config, tensors, shardFiles, shardCount: shardFiles.length };
}
