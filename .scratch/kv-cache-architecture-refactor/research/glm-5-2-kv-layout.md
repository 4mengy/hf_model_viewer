# GLM 5.2 KV Cache 布局调查

## 结论

`GLM 5.2` 在当前公开的一手资料中没有型号歧义：本调查对象是 Z.ai 官方的 [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2/tree/b4734de4facf877f85769a911abafc5283eab3d9)（BF16 权重，固定 revision `b4734de4facf877f85769a911abafc5283eab3d9`）和官方量化别名 [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8/tree/ba978f7d347eaf65d22f1a86833408afdb953541)（固定 revision `ba978f7d347eaf65d22f1a86833408afdb953541`）。两者具有相同的模型类标识、结构字段和 BF16 计算 dtype；FP8 只描述权重 checkpoint，不自动改变本报告的 KV Cache 语义 dtype。[BF16 config](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/config.json)，[FP8 config](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/config.json)

本项目已约定估算的是跨框架稳定的 **Effective KV Cache Payload**，而非某个推理框架的页分配。按该口径，GLM 5.2 的专用完整布局为：

- 78 个 backbone 层各保存一个 576 元素的 BF16 MLA 压缩向量（`512` 维 latent KV + `64` 维 RoPE key），不是分别保存完整 K 和 V。
- 21 个 `full` IndexShare 层各保存一个 128 元素的 BF16 indexer key；其余 57 个 `shared` 层复用最近一次完整 indexer 的 top-k 结果，不拥有历史 indexer key cache。
- 所有历史 token 均保留；`index_topk = 2048` 只减少每次注意力计算读取的历史位置，不是 2048-token 滑窗。
- 当前 query 的 top-k indices、logits/workspace、allocator page padding、prefix-cache 预留、offload 和可选 MTP 推测解码缓存均不计入语义 payload。

因此，在 `B >= 0`、`0 <= S <= 1,048,576` 且每个 batch item 都有 `S` 个实际 token 时：

```text
MLA_bytes(B,S)     = B * S * 78 * (512 + 64) * 2
                   = B * S * 89,856

Indexer_bytes(B,S) = B * S * 21 * 128 * 2
                   = B * S * 5,376

KV_bytes(B,S)      = B * S * 95,232
```

即每个实际 token 恰好 `95,232 bytes = 93 KiB`。`S > 1,048,576` 应报 profile 输入越界，不能静默截断；不同序列长度的 batch 应将公式中的 `B*S` 换成 `sum(sequence_lengths)`。

该 Profile 可以作为 `glm-5.2-semantic-bf16-v1` 落地。资料足以支持这一语义布局，但不足以仅凭模型文件唯一推断用户将采用哪种框架 FP8 cache 编码；后者必须作为独立、显式验证的运行时场景，不能覆盖本 Profile。

## 第一方证据链

### 官方模型身份

- 官方 model card 把 GLM 5.2 描述为 1M context，并明确 IndexShare 每四个 sparse-attention 层复用 indexer；同时列出 SGLang、vLLM、Transformers 等官方支持路径。[固定 model card](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/README.md)
- 两个官方 checkpoint 的 `architectures` 都是精确数组 `["GlmMoeDsaForCausalLM"]`，`model_type` 都是 `"glm_moe_dsa"`，`dtype` 都是 `"bfloat16"`。[BF16 config](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/config.json)，[FP8 config](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/config.json)
- BF16 safetensors 的实际 cache 相关权重确认了投影尺寸：`kv_a_proj_with_mqa.weight = [576, 6144]`、`kv_b_proj.weight = [28672, 512]`、`indexer.wk.weight = [128, 6144]`、`indexer.wq_b.weight = [4096, 2048]`、`indexer.weights_proj.weight = [32, 6144]`。[固定 BF16 shard](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/model-00001-of-00282.safetensors)，[固定 weight index](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/model.safetensors.index.json)
- FP8 checkpoint 保留相同逻辑 shape；`kv_a`、`kv_b`、`indexer.wk`、`indexer.wq_b` 为 `F8_E4M3` 权重并带 F32 scale，`indexer.weights_proj` 保持 BF16。这是 checkpoint 身份特征，不是 KV Cache dtype 证据。[固定 FP8 shard](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/model-00001-of-00141.safetensors)，[固定 FP8 weight index](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/model.safetensors.index.json)

### 缓存拓扑与保留规则

- Transformers v5.12.0 的固定实现先产生 `compressed_kv`，再按 `[kv_lora_rank, qk_rope_head_dim]` 分成 `512 + 64`；vLLM v0.23.0 的 `MLAAttention` 也把语义 `head_size` 定义为 `kv_lora_rank + qk_rope_head_dim`，其 MLA cache spec 是一个向量而不是 K/V 两份。[Transformers fixed source, lines 427-441](https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L427-L441)，[vLLM fixed source](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/model_executor/layers/attention/mla_attention.py)
- Transformers 按 `config.indexer_types[layer_idx]` 决定创建完整 indexer 还是 `shared` 层；只有完整 indexer 会调用 `update_indexer`。[fixed source, lines 351-407](https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L351-L407)，[fixed source, lines 235-245](https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L235-L245)
- `DynamicIndexedLayer.update_indexer` 接受 `[batch_size, seq_len, index_head_dim]` 并沿 sequence 维拼接，没有窗口淘汰。因此 indexer cache 的语义宽度为 128，保留长度为完整 `S`。[fixed cache source, lines 287-322](https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/cache_utils.py#L287-L322)
- 官方 config 的 78 项 `indexer_types` 中，`full` 层精确为 `{0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74}`，共 21 层；其余 57 层为 `shared`。相同模式由 `index_topk_freq=4`、`index_skip_topk_offset=3` 定义。官方权重索引也只在这些 backbone 层包含 indexer 投影（另有 layer 78 的可选 MTP 权重）。[fixed config](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/config.json)，[fixed weight index](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/model.safetensors.index.json)
- config 没有 `sliding_window`，所有 78 个 layer type 都是 `deepseek_sparse_attention`；indexer 的 top-k 是从完整历史中选择，不能把 `index_topk=2048` 当作 cache 长度。[fixed configuration implementation, lines 131-157](https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py#L131-L157)，[fixed model implementation, lines 226-249](https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L226-L249)

## 专用 runtime resolver 签名

Resolver 不得使用仓库名模式、`model_type` 单独命中或通用 MLA/DSA 猜测。只有下面的完整、人工登记签名通过时才返回 `glm-5.2-semantic-bf16-v1`。

### 必需 config predicates

```json
{
  "architectures": ["GlmMoeDsaForCausalLM"],
  "model_type": "glm_moe_dsa",
  "dtype": "bfloat16",
  "use_cache": true,
  "num_hidden_layers": 78,
  "hidden_size": 6144,
  "num_attention_heads": 64,
  "num_key_value_heads": 64,
  "q_lora_rank": 2048,
  "kv_lora_rank": 512,
  "qk_head_dim": 256,
  "qk_nope_head_dim": 192,
  "qk_rope_head_dim": 64,
  "v_head_dim": 256,
  "index_n_heads": 32,
  "index_head_dim": 128,
  "index_topk": 2048,
  "index_topk_freq": 4,
  "index_skip_topk_offset": 3,
  "index_topk_pattern": null,
  "indexer_rope_interleave": true,
  "index_share_for_mtp_iteration": true,
  "max_position_embeddings": 1048576,
  "num_nextn_predict_layers": 1
}
```

还必须要求：

- `indexer_types` 精确等于官方 78 项数组；不要只重新计算 pattern 后忽略显式数组。
- config 不含非空 `sliding_window`、`attention_chunk_size` 或任何未登记的 cache 压缩字段。
- `architectures` 若含另一个已知且指向不同 Profile 的模型类标识，报告冲突；缺失、为空或只含未知值均 fail closed。
- 所有算术恒等式成立：`qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 256`，`576 = kv_lora_rank + qk_rope_head_dim`，`28672 = num_attention_heads * (qk_nope_head_dim + v_head_dim)`。

### 必需 safetensors predicates

对 backbone 层 `i = 0..77`：

- 每层存在 `model.layers.i.self_attn.kv_a_proj_with_mqa.weight`，逻辑 shape 为 `[576, 6144]`。
- 每层存在 `model.layers.i.self_attn.kv_b_proj.weight`，逻辑 shape 为 `[28672, 512]`。
- 仅当 `i` 属于 full 集合 `{0,1,2,6,10,...,74}` 时，必须存在：
  - `indexer.wk.weight [128,6144]`
  - `indexer.wq_b.weight [4096,2048]`
  - `indexer.weights_proj.weight [32,6144]`
- 对全部 57 个 shared 层，上述 indexer 权重必须不存在；存在即说明 checkpoint 与审核过的 IndexShare 拓扑冲突，应 fail closed。
- 因 config 声明一个 MTP layer，严格的官方 checkpoint 身份还应验证 layer 78 的 MTP/cache 相关权重存在。它只用于 checkpoint 身份验证，不加入 base-model KV 公式。

只允许两个已登记的权重 dtype alias：

1. `zai-org/GLM-5.2@b4734de...`：上述投影为 BF16。
2. `zai-org/GLM-5.2-FP8@ba978f7d...`：cache 相关线性权重保持相同逻辑 shape，主要权重为 `F8_E4M3` 并具有相应 F32 inverse-scale tensors，而 `indexer.weights_proj` 为 BF16。

任意其他量化、改名、缺层、额外 indexer 层或 shape/dtype 组合即使看起来“类似 GLM 5.2”，也不能自动归入本 Profile；应新增人工审核 alias。

## Buffer 明细

| Buffer | 层组 | 每 token shape | 语义 dtype | 保留长度 | 每 token 字节 |
|---|---:|---:|---:|---:|---:|
| MLA compressed latent (`c_KV`) | 78 | `[512]` | BF16 | `S` | `78 * 512 * 2 = 79,872` |
| MLA RoPE key (`k_PE`) | 78 | `[64]` | BF16 | `S` | `78 * 64 * 2 = 9,984` |
| DSA indexer key | 21 full indexer layers | `[128]` | BF16 | `S` | `21 * 128 * 2 = 5,376` |
| top-k indices | 跨相邻层的当前 query 临时值 | `[index_topk]` | int32 | 非历史 cache | 不计入 |

合计 `79,872 + 9,984 + 5,376 = 95,232 bytes/token`。

## Golden vectors

以下值从上面的三个独立 buffer 项逐项求和，测试应使用整数 bytes 断言，GB/GiB 只用于展示。

| B | S | MLA bytes | Indexer bytes | Total bytes | 展示值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 GiB |
| 1 | 1 | 89,856 | 5,376 | 95,232 | 93 KiB |
| 1 | 2,048 | 184,025,088 | 11,010,048 | 195,035,136 | 186 MiB |
| 1 | 2,049 | 184,114,944 | 11,015,424 | 195,130,368 | 0.1817293167 GiB |
| 2 | 4,096 | 736,100,352 | 44,040,192 | 780,140,544 | 0.7265625 GiB |
| 4 | 131,072 | 47,110,422,528 | 2,818,572,288 | 49,928,994,816 | 46.5 GiB |
| 1 | 1,048,576 | 94,220,845,056 | 5,637,144,576 | 99,857,989,632 | 99.857989632 GB = 93 GiB |

必须覆盖的边界断言：

- `S=2048` 与 `S=2049` 的差恰好仍为 `95,232` bytes；top-k 阈值不会截断 cache。
- `S=1,048,576` 可计算；`S=1,048,577` 必须返回 profile input error/unknown，而不是 clamp 或继续外推。
- `B=0` 或空的 sequence-length list 返回零；负数 batch/context 输入报错。
- ragged batch `[1, 2048, 2049]` 应按总 token 数 `4098` 计算为 `390,260,736` bytes，即 `95,232 * sum(lengths)`；不能先补齐到最大长度。若产品语义是 allocator capacity，须另建场景，不能改这个 golden。

## 运行时实现差异（不进入语义公式）

- Transformers v5.12.0 的通用 `DynamicCache` 路径在当前实现中把展开后的 per-head K/V 传给 cache；这是框架实现分配，不是 MLA 能表达的最小模型语义 payload。[fixed model source, lines 427-441](https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L427-L441)
- vLLM 官方 GLM 5.2 recipe 显式传入 `--kv-cache-dtype fp8_e4m3`；所以 cache dtype 是部署场景选择，不能从 BF16/FP8 权重文件名或 safetensor dtype 自动推出。[fixed recipe](https://github.com/vllm-project/recipes/blob/548c6a2a4f3b46345254db8ca3a2317a535447dc/models/zai-org/GLM-5.2.yaml)
- vLLM v0.23.0 的 NVIDIA FlashMLA sparse FP8 格式每 token、每 MLA 层使用 656 bytes（512 FP8 NoPE + 4 个 F32 scale + 64 个 BF16 RoPE）；ROCm AITER 路径则可使用 576-wide FP8 表示。两者是 backend-specific 编码，不应冒充单一的模型语义布局。[fixed FlashMLA source](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/attention/backends/mla/flashmla_sparse.py)，[fixed ROCm source](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py)
- vLLM v0.23.0 的 indexer cache 使用每 token `128` 个 FP8 bytes 加一个 F32 scale，即 132 bytes；同一稳定版本的构造路径还可能为 shared/skip 层注册 indexer cache object。这些都属于实现分配或优化格式，不能替换“21 个 full layer × 128 BF16”的语义答案。[fixed indexer source](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/model_executor/models/deepseek_v2.py)
- `num_nextn_predict_layers=1` 和 layer 78 权重支持可选 MTP speculative decoding。是否启用、循环几次、是否为 draft cache 分配额外页属于运行时策略；base Profile 不计入。如果产品以后要展示 “vLLM FP8 + MTP” 实占，应新建独立场景和 golden tests。

## 不确定性与 fail-closed 后果

1. **实际框架显存并不由 config+safetensors 唯一决定。** 同一官方 checkpoint 可采用 BF16、标准 FP8 或 FlashMLA 的混合 FP8/BF16 cache 编码。当前 runtime resolver 没有 backend/cache-mode 输入，因此只能返回本报告明确命名的语义 BF16 Profile，不能声称预测某个框架实际 allocation。
2. **权重 FP8 不等于 KV FP8。** 两个官方 config 都声明 BF16；仅观察 safetensor 权重 dtype 后把 KV bytes 除以二会违反已确认的产品约束。
3. **Transformers 展开缓存与优化 MLA 缓存不同。** 本报告采用模型自身压缩所需的最小持久语义状态；若产品目标改成“Transformers 实测显存”，必须建立新的 framework-specific Profile。
4. **MTP 不在 base 公式中。** 将来若产品口径改为默认启用官方 recipe 的 MTP，须专项验证其持久 cache、共享行为和额外内存；不可简单再加一层。
5. **未登记 checkpoint 一律未知。** 第三方量化或 fork 即使使用同样的 `GlmMoeDsaForCausalLM`，只要 config/tensor signature 未完全命中就应显示“不支持的模型架构”，并使完整显存估算保持未知。

## 实施建议

- Catalog 条目固定为 `glm-5.2-semantic-bf16-v1`，显式列出两个官方 checkpoint alias 和上述严格 predicates。
- Layout 实现返回三条可审计 buffer 明细（latent、RoPE key、indexer key），最终用整数 bytes 汇总。
- 错误应区分：模型类未命中、GLM 5.2 config signature 冲突、indexer pattern/权重拓扑冲突、tensor shape/dtype alias 未审核、context 超过 1M。
- 不提供启发式 GLM/MLA/DSA fallback；不根据权重量化 slider 改变 KV payload。
