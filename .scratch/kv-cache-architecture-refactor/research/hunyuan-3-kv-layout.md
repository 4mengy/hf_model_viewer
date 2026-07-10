# Tencent Hy3 KV Cache 布局调查

## 结论

本票的唯一主目标是腾讯正式版 [`tencent/Hy3`](https://huggingface.co/tencent/Hy3/tree/716aa7241bd6d95896be4ebfc761162a9c4d49ef)，不是 `tencent/Hy3-preview`，也不是 Hunyuan3D。调查固定到 2026-07-10 时官方 revision `716aa7241bd6d95896be4ebfc761162a9c4d49ef`。该 revision 的 `architectures` 精确等于 `['HYV3ForCausalLM']`，`model_type` 为 `hy_v3`。[固定 config](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/config.json)

建议目录项：

```text
profile_id: hy3-instruct-semantic-bf16-v1
layout_id: hy3-full-gqa-bf16-v1
primary_checkpoint: tencent/Hy3@716aa7241bd6d95896be4ebfc761162a9c4d49ef
model_class_identifier: HYV3ForCausalLM
```

Hy3 的主体 KV Cache 布局为 **80 层、全上下文、独立 Key/Value 的 GQA**。每层有 8 个 KV heads，每个 head 128 维，Key 和 Value 都是 BF16；没有 sliding window、CLA、MLA latent、indexer、压缩池或混合层组。腾讯官方表格独立列出 80 个主体层、1 个 MTP 层、64 query heads、8 KV heads、head dim 128、256K context 和 BF16 supported precision。[固定官方 model card](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/README.md#model-introduction)

在每个 batch item 都有 `S` 个有效 token 时，主体 Effective KV Cache Payload 为：

```text
K_elements(B,S) = B * S * 80 * 8 * 128
V_elements(B,S) = B * S * 80 * 8 * 128

K_bytes(B,S) = B * S * 80 * 8 * 128 * 2
             = B * S * 163,840

V_bytes(B,S) = B * S * 163,840

KV_bytes(B,S) = K_bytes + V_bytes
              = B * S * 80 * 2 * 8 * 128 * 2
              = B * S * 327,680 bytes
```

也就是每个有效 token 恰好 `327,680 bytes = 320 KiB`。输入必须是非负整数，`S <= 262,144`；超出 256K 必须返回 Profile 输入越界/unknown，不能 clamp 或继续外推。ragged batch 用 `sum(sequence_lengths)` 代替 `B*S`。

## 第一方证据链

### 正式版模型身份

- 正式版 model card 明确说 Hy3 是在 Hy3 Preview 之后扩大高质量 post-training 数据和 RL 训练得到的新发布版本；因此不能把 Preview 当成主调查对象。[固定正式版 model card](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/README.md#model-introduction)
- 固定 config 精确声明 `HYV3ForCausalLM`、`hy_v3`、80 个主体层、1 个 next-token predictor layer、64/8 Q/KV heads、128 head dim、4096 hidden size、262144 maximum positions、`use_cache=true` 和 `qk_norm=true`。[固定 config](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/config.json)
- 官方 Transformers 版本为 5.6.0；其固定 `HYV3Config` 把 `num_key_value_heads=8` 定义为 GQA 的 KV head 数，把 `head_dim=128` 定义为单 head 维度，并把 `use_cache` 默认为 true。[Transformers v5.6.0 fixed config](https://github.com/huggingface/transformers/blob/3e80155a968c1080f11b2710e8b31741ac5ab0ed/src/transformers/models/hy_v3/configuration_hy_v3.py#L76-L103)

### 缓存拓扑与保留策略

- Transformers 5.6.0 的 `HYV3Attention` 分别投影 Q、K、V；K/V 输出宽度都是 `num_key_value_heads * head_dim = 8 * 128 = 1024`。它先对 K 做 QK norm 和 RoPE，再把 K/V 传给 cache `update`，Query 不进入 cache。[fixed HYV3 attention](https://github.com/huggingface/transformers/blob/3e80155a968c1080f11b2710e8b31741ac5ab0ed/src/transformers/models/hy_v3/modeling_hy_v3.py#L223-L283)
- `HYV3Model` 精确构建 `range(config.num_hidden_layers)`，即主体层 `0..79`，并在启用 cache 时使用 `DynamicCache`；源码还明确注释 `No sliding window`。[fixed HYV3 model](https://github.com/huggingface/transformers/blob/3e80155a968c1080f11b2710e8b31741ac5ab0ed/src/transformers/models/hy_v3/modeling_hy_v3.py#L468-L541)
- Transformers 的 `DynamicLayer` 把每层 K 和 V 保存为 `[batch_size, num_heads, seq_len, head_dim]`，每次沿 sequence 维拼接；`is_sliding=false`。因此每层逻辑 shape 精确为两份 `[B,8,S,128]`，保留完整历史 `S`。[fixed DynamicLayer](https://github.com/huggingface/transformers/blob/3e80155a968c1080f11b2710e8b31741ac5ab0ed/src/transformers/cache_utils.py#L88-L121)
- 当前固定 config 没有 `sliding_window`、`layer_types`、`use_cla`、MLA 维度或 indexer 字段；官方源码也没有为 HYV3 选择混合/特殊 cache layer。因此不能把它按名称猜成其他 Hunyuan 布局，也不需要拼装多个布局。

### Dtype

- 正式版 model card 只列 BF16 为 supported precision；固定 checkpoint 的 safetensors 汇总是 `298,786,140,416` 个 BF16 参数和 `15,360` 个 F32 expert-bias 参数。所有 cache 相关投影与 norm 都是 BF16。[固定 model card](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/README.md#model-introduction)，[固定 weight index](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/model.safetensors.index.json)
- Transformers `DynamicLayer` 从首次 `key_states` 继承 dtype 并用相同 dtype 创建 K/V cache。官方 BF16 模型路径因此保存 BF16 K/V。[fixed DynamicLayer](https://github.com/huggingface/transformers/blob/3e80155a968c1080f11b2710e8b31741ac5ab0ed/src/transformers/cache_utils.py#L88-L121)
- `config.json` 本身没有一个可供用户覆盖的 KV cache dtype 字段。本 Profile 明确固定 BF16；Hy3-FP8、显式 vLLM FP8 cache 或其他量化模式都不是本目录项的自动 alias。

## 专用 runtime resolver

Resolver 不得使用仓库名模式、`model_type` 单独命中或通用 GQA fallback。只有完整人工登记签名通过时才返回 `hy3-instruct-semantic-bf16-v1`。

### 必需 config predicates

```json
{
  "architectures": ["HYV3ForCausalLM"],
  "model_type": "hy_v3",
  "transformers_version": "5.6.0",
  "use_cache": true,
  "num_hidden_layers": 80,
  "num_nextn_predict_layers": 1,
  "hidden_size": 4096,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "max_position_embeddings": 262144,
  "qk_norm": true,
  "vocab_size": 120832,
  "intermediate_size": 13312,
  "first_k_dense_replace": 1,
  "num_experts": 192,
  "num_experts_per_tok": 8,
  "num_shared_experts": 1,
  "expert_hidden_dim": 1536,
  "moe_intermediate_size": 1536,
  "router_scaling_factor": 2.826,
  "tie_word_embeddings": false,
  "rope_parameters": {
    "rope_type": "default",
    "rope_theta": 11158840.0
  }
}
```

还必须要求：

- `architectures` 为精确 singleton；多已知模型类且指向不同 Profile 时报告冲突。
- `sliding_window`、`layer_types`、`attention_chunk_size`、`kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim`、`index_head_dim`、`index_n_heads` 缺失或为 `null`。
- `use_cla` 缺失或为 `false`，`cla_share_factor` 缺失或为 `null/1`。未来同名 HYV3 若加入跨层 KV 共享，必须新建 Profile。
- `quantization_config` 与显式模型级 KV cache dtype 缺失/空。权重量化或 cache override 不得改变本 Profile 的字节数。
- 恒等式 `64 % 8 == 0`、`8 * 128 == 1024`、`64 * 128 == 8192` 成立；它们只用于冲突诊断，不能替代精确字段。

### 必需 safetensors predicates

正式版 [`model.safetensors.index.json`](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/model.safetensors.index.json) 有 47,138 个 tensor 名称，`metadata.total_size = 597572342272`。它被打成 99 个 shard，但 shard 数只能作为来源审计信息，不能作为 Architecture Profile predicate。

对每个主体层 `i = 0..79`，精确存在：

| 完整 tensor 名 pattern | dtype | shape |
|---|---:|---:|
| `model.layers.{i}.self_attn.q_proj.weight` | BF16 | `[8192,4096]` |
| `model.layers.{i}.self_attn.k_proj.weight` | BF16 | `[1024,4096]` |
| `model.layers.{i}.self_attn.v_proj.weight` | BF16 | `[1024,4096]` |
| `model.layers.{i}.self_attn.o_proj.weight` | BF16 | `[4096,8192]` |
| `model.layers.{i}.self_attn.q_norm.weight` | BF16 | `[128]` |
| `model.layers.{i}.self_attn.k_norm.weight` | BF16 | `[128]` |
| `model.layers.{i}.input_layernorm.weight` | BF16 | `[4096]` |
| `model.layers.{i}.post_attention_layernorm.weight` | BF16 | `[4096]` |

MTP checkpoint layer `i=80` 精确存在以下非-MLP identity/cache-related tensors：

| 完整 tensor 名 | dtype | shape |
|---|---:|---:|
| `model.layers.80.eh_proj.weight` | BF16 | `[4096,8192]` |
| `model.layers.80.enorm.weight` | BF16 | `[4096]` |
| `model.layers.80.hnorm.weight` | BF16 | `[4096]` |
| `model.layers.80.final_layernorm.weight` | BF16 | `[4096]` |
| `model.layers.80.input_layernorm.weight` | BF16 | `[4096]` |
| `model.layers.80.post_attention_layernorm.weight` | BF16 | `[4096]` |
| `model.layers.80.self_attn.q_proj.weight` | BF16 | `[8192,4096]` |
| `model.layers.80.self_attn.k_proj.weight` | BF16 | `[1024,4096]` |
| `model.layers.80.self_attn.v_proj.weight` | BF16 | `[1024,4096]` |
| `model.layers.80.self_attn.o_proj.weight` | BF16 | `[4096,8192]` |
| `model.layers.80.self_attn.q_norm.weight` | BF16 | `[128]` |
| `model.layers.80.self_attn.k_norm.weight` | BF16 | `[128]` |

不得存在 `model.layers.81.*`。代表性 tensor metadata 位于 [layer 0 shard](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/model-00090-of-00099.safetensors)、[layers 79/80 QKV shard](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/model-00066-of-00099.safetensors)、[MTP projection shard](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/model-00018-of-00099.safetensors) 和 [MTP norm shard](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/model-00091-of-00099.safetensors)。

本次实际读取了固定 revision 全部 99 个 safetensors header：81 个 Q、81 个 K、81 个 V、81 个 output projection、162 个 Q/K norm 和 162 个 input/post-attention norm 均命中上述 shape/dtype，零例外。主体层必须连续且恰为 `0..79`；缺层、额外主体层或任意 cache-related shape/dtype 冲突均 fail closed。

## Buffer 明细

| Buffer | 层组 | 每层逻辑 shape | 语义 dtype | 保留长度 | 全模型每 token bytes |
|---|---:|---:|---:|---:|---:|
| normalized + RoPE Key | 主体 `0..79`（80 层） | `[B,8,S,128]` | BF16 | 完整 `S` | `80 * 8 * 128 * 2 = 163,840` |
| Value | 主体 `0..79`（80 层） | `[B,8,S,128]` | BF16 | 完整 `S` | `80 * 8 * 128 * 2 = 163,840` |

合计 `327,680 bytes/token`。Query、Q/K norm 权重、RoPE cos/sin、MoE router/expert 状态、attention logits、softmax workspace 和当前 token 临时 tensor 都不是历史 KV buffer。

## MTP：存在但明确排除主体 payload

主体 Profile **不计 layer 80 的 MTP cache**。这是明确的范围决定：

1. 腾讯把模型描述为“80 layers excluding MTP + 1 MTP layer”；正式版 vLLM 命令写明 `with MTP enabled`，必须显式传 `--speculative-config.method mtp`。因此 MTP 是可选 speculative runtime strategy。[固定结构表和部署命令](https://huggingface.co/tencent/Hy3/blob/716aa7241bd6d95896be4ebfc761162a9c4d49ef/README.md#deployment)
2. Transformers `HYV3ForCausalLM` 只构建和执行 `num_hidden_layers=80` 个主体层；它的普通生成 cache 不含 layer 80。[fixed Transformers model](https://github.com/huggingface/transformers/blob/3e80155a968c1080f11b2710e8b31741ac5ab0ed/src/transformers/models/hy_v3/modeling_hy_v3.py#L468-L605)
3. vLLM 的普通 `HYV3ForCausalLM` 同样只构建 80 个主体层，并在普通模型 weight loader 中过滤 MTP 权重。[fixed vLLM main model](https://github.com/vllm-project/vllm/blob/2d814a00820daec7082599bea75ae1d0959a346c/vllm/model_executor/models/hy_v3.py#L445-L465)，[fixed MTP filter](https://github.com/vllm-project/vllm/blob/2d814a00820daec7082599bea75ae1d0959a346c/vllm/model_executor/models/hy_v3.py#L653-L735)
4. vLLM 用独立 `HYV3MultiTokenPredictor`/`HYV3MTP` 构建 layer 80；正式版 recipe 要求 2 个 speculative tokens，但 checkpoint 仍只有 1 个 MTP layer，源码通过 modulo 重复选择该层。这进一步说明 speculative steps 不能直接当作额外主体层数。[fixed vLLM MTP](https://github.com/vllm-project/vllm/blob/2d814a00820daec7082599bea75ae1d0959a346c/vllm/model_executor/models/hy_v3_mtp.py#L89-L204)

MTP 模块确实包含同 shape 的 GQA attention，所以不能说它“没有 cache”。但其 draft token 生命周期、复用和分页属于 runtime scenario；本地图已排除这类策略。审计输出应显示 `MTP present: 1; core payload inclusion: excluded (optional speculative runtime)`。以后若支持 MTP 实占，必须单独调查和建 golden，不能无条件把公式从 80 层改成 81 层，更不能因为 `num_speculative_tokens=2` 改成 82 层。

## Golden vectors

测试以整数 bytes 断言；GB/GiB 仅用于显示。

| B | S | K bytes | V bytes | Total bytes | 展示值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 GiB |
| 1 | 1 | 163,840 | 163,840 | 327,680 | 320 KiB |
| 3 | 17 | 8,355,840 | 8,355,840 | 16,711,680 | 15.9375 MiB |
| 1 | 4,096 | 671,088,640 | 671,088,640 | 1,342,177,280 | 1.25 GiB |
| 2 | 32,768 | 10,737,418,240 | 10,737,418,240 | 21,474,836,480 | 20 GiB |
| 1 | 262,144 | 42,949,672,960 | 42,949,672,960 | 85,899,345,920 | 85.89934592 GB = 80 GiB |

必须覆盖：

- `KV(B,S+1) - KV(B,S) == B * 327,680`；没有窗口或压缩边界。
- `S=262,144` 可计算；`S=262,145` 返回越界/unknown。
- ragged `[1,4096,8192]` 以 `sum=12,289` 计算为 `4,026,859,520 bytes`，不能按 padding capacity 计算。
- 负数、非整数和整数溢出报错；bytes 汇总使用整数/BigInt。
- layer 80 MTP 存在时核心结果仍是 80 层；`331,776 bytes/token` 或最大上下文 `81 GiB` 都是误把 MTP 加入主体的失败结果。

## 与 Hy3 Preview 的人工对照

Preview 只用于验证差异，不是本报告的主证据：

| 项目 | `tencent/Hy3` final | `tencent/Hy3-preview` |
|---|---|---|
| 固定 revision | `716aa724...` | `b53bd705...` |
| 发布身份 | 正式 post-trained Hy3 | Preview |
| config | 与 Preview 逐字段相同 | 与 final 逐字段相同 |
| tensor 名称数 | 47,138 | 47,138 |
| cache-related names/shapes/dtypes | 与 Preview 相同 | 与 final 相同 |
| index `metadata.total_size` | `597572342272` | `597572342272` |
| shard count | 99 | 112 |
| 官方 vLLM speculative recipe | 2 speculative tokens | 1 speculative token |

人工专项比较确认两者的 **KV 语义布局相同**，但发布身份、训练后权重内容、tokenizer/template 和推荐 runtime recipe 不是一回事。正式版来源仍必须记录为本 Profile 的 primary checkpoint。

### 运行时无法仅靠当前允许元数据稳定区分两者

config、tensor name set、cache-related shape/dtype 和 `metadata.total_size` 都相同。唯一显眼差异是 99 vs 112 shards，但 shard count 是可无损改变的打包细节：重新分片、合并 safetensors 或本地转换都会改变它，而 KV 布局不变；第三方也能复制相同分片数。因此 **不得把 shard count 当 Architecture Profile discriminator**。

按当前“运行时只读 config + safetensors tensor metadata”的边界，有两种诚实行为：

1. 面向 KV 布局（推荐）：在 catalog 中人工记录 final 为 primary，并把已专项验证、metadata-equivalent 的 Preview 记录为同一 Profile/layout 的显式证据 alias。运行时不必区分，因为结果相同；这不是启发式复用。
2. 若产品要求“只接受 final，必须拒绝 Preview”：现有输入不足，必须增加可信 `repoId + revision` 或 checkpoint 内容哈希。没有它们时精确 checkpoint 身份应 fail closed；不能偷偷使用 shard count。

`tencent/Hy3-FP8` 未在本票内验证，不得因为模型类相同或官方 model card 提到它就自动成为 alias。

## Effective payload 与 runtime allocation

- 本公式是全模型逻辑有效元素，不是某张 GPU 的 reserved memory。vLLM block capacity、最后一页空槽、prefix cache 预留、allocator alignment/碎片与 CUDA graph buffers 都不计入。
- tensor parallel 会分片 8 个 KV heads；当 TP 大于 8 时框架可能复制 heads。pipeline parallel 会按层分布。它们影响逐设备物理 allocation，不改变逻辑 `80 * 8 heads`；没有明确部署输入时不得平均猜测。[fixed vLLM TP logic](https://github.com/vllm-project/vllm/blob/2d814a00820daec7082599bea75ae1d0959a346c/vllm/model_executor/models/hy_v3.py#L223-L262)
- Transformers `DynamicCache` 的有效 K/V tensor shape 可作为公式交叉核对，但其 `torch.cat` 中间分配不属于持久 payload；vLLM paged cache 的预留容量也不属于。
- 显式 FP8 KV、offload、prefix sharing、MTP 和 backend-specific storage 都应建立独立 runtime scenario，不能覆盖 `hy3-instruct-semantic-bf16-v1`。

## Fail-closed 清单与实施建议

- Catalog 固定 `hy3-instruct-semantic-bf16-v1 -> hy3-full-gqa-bf16-v1`；primary evidence 指向 `tencent/Hy3@716aa724...`。
- Layout 输出两条 buffer 明细：`main.key` 和 `main.value`，层组精确为 `0..79`、BF16、full retention。
- Evidence 输出 config predicate、全量 cache-related tensor 验证、固定 source revisions，以及 MTP `present but excluded`。
- 错误至少区分：模型类未命中、config signature 冲突、cache-semantic 字段冲突、tensor range/shape/dtype 冲突、context 超过 256K、严格 checkpoint 身份不可判定。
- 未验证 Hy3-FP8、第三方量化/转换、未来 revision、显式非-BF16 cache 或 MTP runtime 时一律 unknown；KV unknown 时完整总显存也保持 unknown。
- 不使用 GQA fallback，不根据权重量化 slider 修改 KV bytes，不把未知 KV 按零加入总显存。
