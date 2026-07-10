# DeepSeek V4 Pro KV Cache 布局调查

## 结论

DeepSeek V4 Pro 有足够的第一方公开证据建立一个专用 Architecture Profile，但只应支持已审计的官方 instruct checkpoint；不能只凭 `DeepseekV4ForCausalLM` 或 `model_type=deepseek_v4` 接受其他 V4、Base、DSpark 或第三方转换。

建议目录项：

```text
profile_id: deepseek-v4-pro-instruct-b5968e9
layout_id: deepseek-v4-pro-csa-hca-bf16-v1
official_repo: deepseek-ai/DeepSeek-V4-Pro
official_revision: b5968e9190ef611bbf34a7229255be88a0e937c1
```

该 revision 是 2026-07-10 调查时官方仓库的 HEAD；所有结论固定到这个 snapshot，而不是浮动的 `main`。[官方 revision](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/commit/b5968e9190ef611bbf34a7229255be88a0e937c1)

官方配置与参考推理实现共同表明：61 个主层中，31 层是压缩率 128 的 HCA，30 层是压缩率 4、带独立 indexer cache 的 CSA；每层还保留 128-token 本地滑窗。主 KV、压缩 KV 与 indexer KV 都以 BF16 保存；每个 compressor 的增量压缩状态是 FP32。[固定配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json) [固定参考实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py)

## 身份与 fail-closed resolver

官方 `config.json` 中唯一的模型类标识是精确字符串 `DeepseekV4ForCausalLM`，`model_type` 是 `deepseek_v4`。API 名 `deepseek-v4-pro` 不是 `config.json` 中的模型类标识，不能作为运行时匹配键。[固定配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json)

Resolver 必须同时满足以下条件；任一缺失或冲突即 fail closed：

1. `architectures` 精确等于 `["DeepseekV4ForCausalLM"]`，且 `model_type == "deepseek_v4"`。
2. `num_hidden_layers == 61`、`num_nextn_predict_layers == 1`、`hidden_size == 7168`、`num_attention_heads == 128`、`num_key_value_heads == 1`、`head_dim == 512`、`qk_rope_head_dim == 64`。
3. `sliding_window == 128`、`max_position_embeddings == 1048576`、`use_cache == true`、`torch_dtype == "bfloat16"`。
4. `index_n_heads == 64`、`index_head_dim == 128`、`index_topk == 1024`、`compress_rope_theta == 160000`。
5. `expert_dtype == "fp4"`。这不是 KV 布局参数，而是把本目录项限定到已审计 instruct checkpoint 的身份条件。
6. `compress_ratios` 长度必须为 62，且精确等于：

   ```text
   [128,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,
    4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,
    4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,
    4,0]
   ```

7. safetensors 元数据必须验证 checkpoint 拓扑，而不只验证配置：
   - 精确存在主层 `layers.0` 至 `layers.60`，以及一个 `mtp.0`。
   - 所有主层的 `attn.wkv.weight` 为 `F8_E4M3 [512,7168]`。
   - HCA 层的 `attn.compressor.ape` 为 `F32 [128,512]`，`wkv/wgate.weight` 为 `BF16 [512,7168]`，并且不存在该层的 `attn.indexer.*`。
   - CSA 层的主 compressor `ape` 为 `F32 [4,1024]`，`wkv/wgate.weight` 为 `BF16 [1024,7168]`；indexer compressor `ape` 为 `F32 [4,256]`，`wkv/wgate.weight` 为 `BF16 [256,7168]`，且 `indexer.weights_proj.weight` 为 `BF16 [64,7168]`。
   - `mtp.0.attn` 存在，但不存在 `mtp.0.attn.compressor.*` 或 `mtp.0.attn.indexer.*`。
   - instruct 身份还应抽查/验证 FP4-packed expert 元数据，例如 `layers.0.ffn.experts.0.w1.weight` 为 `I8 [3072,3584]`、对应 scale 为 `F8_E8M0 [3072,224]`。

这些 tensor 名称来自固定的官方 [`model.safetensors.index.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/model.safetensors.index.json)，shape/dtype 来自同一 revision 的官方 safetensors header（HCA 样本在 `model-00002-of-00064.safetensors`，CSA 样本在 `model-00004-of-00064.safetensors`）。

不要把以下官方仓库自动当作别名：

- `deepseek-ai/DeepSeek-V4-Pro-Base`：模型类与主体配置相同，但 `expert_dtype=fp8`，是独立 checkpoint；本票未给它建立目录项。
- `deepseek-ai/DeepSeek-V4-Pro-DSpark`：同一模型类标识，但 `compress_ratios` 多出三个 `0`，并增加 `dspark_*` 字段，布局/执行场景不同。

因此当前 Profile 没有额外的模型类标识别名。若以后支持 Base 或 DSpark，必须另做人工审核后显式加入目录。

## 层组与保留语义

设 `W=128`、主 KV latent 维度 `D=512`、indexer KV 维度 `I=128`。

### HCA：31 层

层索引为 `0, 1, 3, 5, ..., 59`，压缩率 `R=128`。

每层持有：

- 本地滑窗 KV：最后 `min(S,W)` 个 token，每 token 一个共享的 512 维 K=V latent，不是独立 K 与 V 两份。
- HCA 压缩 KV：完整历史每 128 token 产生一个 512 维条目，因此保留 `floor(S/128)` 个条目。
- Compressor 增量状态：`kv_state` 与 `score_state` 两个 FP32 buffer；非 overlap，当前未满块有 `S mod 128` 个 live row。

HCA 没有 indexer；参考实现把所有当前可见的压缩条目加入 attention 索引。压缩池不会被 `index_topk` 截断。[Attention/Compressor 实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py#L279-L377)

### CSA：30 层

层索引为 `2, 4, 6, ..., 60`，压缩率 `R=4`。

每层持有：

- 本地滑窗 KV：最后 `min(S,W)` 个 token，每 token 512 维。
- CSA 压缩 KV：完整历史每 4 token 一个 512 维条目，共 `floor(S/4)` 个。
- 主 compressor overlap 状态：`kv_state` 与 `score_state`，每个 live row 宽 `2D=1024`、FP32。
- Indexer 压缩 KV：完整历史每 4 token 一个 128 维条目，共 `floor(S/4)` 个。
- Indexer compressor overlap 状态：另一个 `kv_state` 与 `score_state`，每个 live row 宽 `2I=256`、FP32。

`index_topk=1024` 只限制一次 query 读取多少压缩位置，不删除压缩池条目，因此不能用 `min(floor(S/4),1024)` 计算驻留 cache。[Indexer/Attention 实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py#L380-L533)

Overlap compressor 在首个 4-token 块前有 `S` 个 live row；之后保留上一个完整块的 4 行和当前块余数。因此定义：

```text
A4(S) = S                         , 0 <= S < 4
        4 + (S mod 4)             , S >= 4
```

为避免低估，payload 对每个 live row 按参考实现实际保存的完整行宽 `2D`/`2I` 计数；即使后续 kernel 只读取其中一部分维度，也不把已保存的半行假设为可消除。

## Dtype

- 主 local/compressed KV cache：BF16，2 bytes/element。
- Indexer compressed KV cache：BF16，2 bytes/element。
- 四类 compressor state（主/indexer 的 `kv_state` 与 `score_state`）：FP32，4 bytes/element。
- Cache 没有独立的持久 scale buffer。代码对非-RoPE 维度做 FP8 fake-quant、对 indexer 做 FP4 simulation，但随后仍保存在 BF16 tensor；官方注释明确说明当前实现使用 BF16，只是未来“could also use fp8”。[固定参考实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py#L415-L420) [主 KV 保存](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py#L501-L533)

`generate.py` 在构造模型前将默认 dtype 设为 BF16，而 cache 的 `torch.zeros` 没有覆盖 dtype；state 则显式指定 FP32。[固定生成入口](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/generate.py)

## Effective KV Cache Payload 公式

输入：

- `B`：活跃 batch 数，正整数。
- `S`：已经进入 cache 的 token 数，`0 <= S <= 1,048,576`。
- `L(S)=min(S,128)`。
- `C128(S)=floor(S/128)`、`r128(S)=S mod 128`。
- `C4(S)=floor(S/4)`。

下式计算单个 tensor-parallel rank/GPU 的语义有效 payload。参考实现没有按 `world_size` 切分 cache tensor，所以 TP rank 间是复制关系；总集群 payload 为 `TP * E_rank`。

```text
HCA_cache  = B * 31 * 2 * 512 * (L + C128)
HCA_state  = B * 31 * 2 * 4 * 512 * r128

CSA_main_cache  = B * 30 * 2 * 512 * (L + C4)
CSA_index_cache = B * 30 * 2 * 128 * C4
CSA_state       = B * 30 * 2 * 4 * (2*512 + 2*128) * A4

E_rank(B,S) = HCA_cache + HCA_state
            + CSA_main_cache + CSA_index_cache + CSA_state
```

合并后：

```text
E_rank(B,S) = B * {
  31 * [1024*(L+C128) + 4096*r128]
  + 30 * [1024*(L+C4) + 256*C4 + 10240*A4]
} bytes
```

### 为什么包含 compressor state

`kv_state`/`score_state` 跨 decode 调用保存未完成压缩块，下一 token 的结果依赖它们；它们不是一次 forward 的临时 activation。因此 Effective payload 必须包含 live rows。反之，未使用的预分配行、allocator 碎片、临时 `index_score`/`topk_idxs`、prefill 的临时 `torch.cat` 都不属于语义 payload。

### 与参考实现实际预分配的区别

官方参考实现按 `max_batch_size` 与 `max_seq_len` 一次性分配完整容量：

- HCA 主 cache `[Bmax,128+floor(M/128),512]`；两个 state 各 `[Bmax,128,512]`。
- CSA 主 cache `[Bmax,128+floor(M/4),512]`；两个 state 各 `[Bmax,8,1024]`。
- CSA index cache `[Bmax,floor(M/4),128]`；两个 index state 各 `[Bmax,8,256]`。

这是 runtime allocation/capacity，不应冒充 Effective payload。对 `Bmax=1, M=1,048,576`，再算上构造但默认生成未使用的 MTP sliding buffer，参考实现分配约 `9.6421813965 GiB`；同一点的语义 payload 是 `9.6257781982 GiB`。

## MTP 处理

`compress_ratios` 第 62 个值（索引 61）为 `0`，对应 `num_nextn_predict_layers=1` 的 `mtp.0`，不是第 61 个主层。官方 `Transformer.forward` 只遍历 `self.layers` 的 61 个主层并直接产生 logits；官方 `generate.py` 也不调用 `self.mtp`。[模型 forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py#L751-L814) [生成入口](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/generate.py)

因此默认生成场景的 Effective payload 不含 MTP cache。参考构造器仍会为 MTP Attention 分配一个 `[Bmax,128,512]` BF16 滑窗 buffer，这是未使用的参考 runtime allocation。若将来支持 MTP/speculative decoding，必须建立独立已验证场景；当前 resolver 不得静默把它加到语义 payload。

## Golden vectors

下列结果由上面的公式独立计算，单位使用二进制 MiB/GiB；测试应同时断言各分量与总数。

| B | S | HCA cache | HCA state | CSA main | CSA index | CSA state | Total bytes | MiB / GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 31,744 | 126,976 | 30,720 | 0 | 307,200 | 496,640 | 0.4736328125 MiB |
| 1 | 3 | 95,232 | 380,928 | 92,160 | 0 | 921,600 | 1,489,920 | 1.4208984375 MiB |
| 1 | 4 | 126,976 | 507,904 | 153,600 | 7,680 | 1,228,800 | 2,024,960 | 1.93115234375 MiB |
| 1 | 7 | 222,208 | 888,832 | 245,760 | 7,680 | 2,150,400 | 3,514,880 | 3.35205078125 MiB |
| 1 | 8 | 253,952 | 1,015,808 | 307,200 | 15,360 | 1,228,800 | 2,821,120 | 2.6904296875 MiB |
| 1 | 127 | 4,031,488 | 16,125,952 | 4,853,760 | 238,080 | 2,150,400 | 27,399,680 | 26.13037109375 MiB |
| 1 | 128 | 4,094,976 | 0 | 4,915,200 | 245,760 | 1,228,800 | 10,484,736 | 9.9990234375 MiB |
| 1 | 129 | 4,094,976 | 126,976 | 4,915,200 | 245,760 | 1,536,000 | 10,918,912 | 10.4130859375 MiB |
| 1 | 1,048,576 | 264,110,080 | 0 | 8,056,995,840 | 2,013,265,920 | 1,228,800 | 10,335,600,640 | 9.625778198242188 GiB |
| 4 | 1,048,576 | 1,056,440,320 | 0 | 32,227,983,360 | 8,053,063,680 | 4,915,200 | 41,342,402,560 | 38.50311279296875 GiB |

关键边界断言：

- `S=3 -> 4`：首次生成 CSA compressed/index 条目，并开始保留 overlap 的上一完整块。
- `S=7 -> 8`：CSA 当前块被压缩，live overlap state 从 7 行回到 4 行，因此 payload 可以下降；这不是计算错误。
- `S=127 -> 128`：HCA 产生首个压缩条目，128 行未完成状态归零，因此 payload 显著下降。
- `S=128 -> 129`：本地窗口保持 128，新增 HCA/CSA compressor state，不再增长 local cache。

## 排除项与不确定性

以下不应伪装为已验证能力：

- FP8/FP4 cache：源码只有 QAT simulation 与“could use”注释，没有第一方已发布的持久 cache dtype/layout；必须 fail closed。
- DeepSeek-V4-Pro-Base、DeepSeek-V4-Pro-DSpark 与第三方量化/转换：本报告未把它们验证为同一 Profile。
- MTP/speculative decoding cache、prefix sharing、offload、paged cache、allocator 预留与碎片：都是独立运行时场景或 out of scope。
- Pipeline parallel 的逐 GPU 分层分布：官方参考实现只展示 tensor parallel；若 UI 要给 PP 单卡数，必须另有部署拓扑输入，不能平均猜测。
- `freqs_cis` 是随最大长度预计算的 RoPE 常量，属于固定模型/runtime overhead，不是 token KV payload；`index_score`、`topk_idxs` 和 prefill 拼接 tensor 是临时工作区，也不属于本公式。

官方技术报告只提供“1M context 下相对 V3.2 为 10% KV cache”的系列级比较，不能替代逐 buffer dtype 与 shape 证据；本 Profile 应以固定配置、safetensors metadata 与固定参考实现为准。[第一方技术报告](https://arxiv.org/abs/2606.19348v1)
