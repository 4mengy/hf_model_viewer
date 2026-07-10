# `config.architectures` 与 KV Cache 布局是否一一对应

## 结论

**不一一对应。** 完全相同的 `config.architectures[]` 字符串可以合法地对应不同的 KV Cache 保留拓扑；因此不能把该字符串单独当作唯一、完备的 KV 布局标识。

已确认的一手反例是 Mistral：

- [`mistralai/Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/27d67f1b5f57dc0953326b2601d68371d40ea8da/config.json) 的 `architectures` 为 `["MistralForCausalLM"]`，`sliding_window` 为 `4096`。
- [`mistralai/Mistral-7B-v0.3`](https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/caa1feb0e54d415e2df31207e5f4e273e33509b1/config.json) 的 `architectures` 同样为 `["MistralForCausalLM"]`，但 `sliding_window` 为 `null`。

在当前 Transformers 的 config-aware cache 路径中，未显式提供 `layer_types` 时，`sliding_window != None` 会把所有层推导为 `sliding_attention`，否则推导为 `full_attention`；随后 `DynamicCache`/`StaticCache` 按这些 layer types 创建不同的缓存层（[推导与分派源码](https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/cache_utils.py#L1473-L1503)，[DynamicCache 分派](https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/cache_utils.py#L1506-L1562)）。普通 `DynamicLayer` 持续拼接全部历史 KV、没有长度上限；`DynamicSlidingWindowLayer` 只保留最近窗口并丢弃更老的 KV（[两种更新语义](https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/cache_utils.py#L118-L161)，[滑窗保留逻辑](https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/cache_utils.py#L190-L255)）。这不是 `num_key_value_heads` 等数值变化，而是保留策略和序列维内存增长方式不同。

## `architectures` 实际表示什么

Transformers 将它定义为“可用于这组预训练权重的模型架构（类）”列表（[`PreTrainedConfig` 定义](https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/configuration_utils.py#L207-L212)）。保存模型时，Transformers 直接把当前模型的 Python 类名写入该字段（[`save_pretrained` 源码](https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/modeling_utils.py#L3509-L3516)）。`AutoModel` 也只把它当作同一 config 类型下的候选类选择提示：匹配不到时会返回默认候选（[`_get_model_class`](https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/models/auto/auto_factory.py#L178-L191)）。

所以该字段是**权重可加载的模型类提示**，不是模型内部 attention/cache 结构的规范化指纹。

## 必须分开的三个维度

1. **布局拓扑 / buffer 语义**：每层是 full、sliding、chunked、混合层，或特殊状态；是否丢弃旧 KV。Mistral 反例证明同一 `architectures` 值在这一维也可能不同。
2. **几何参数**：`num_hidden_layers`、`num_key_value_heads`、`head_dim`、窗口大小等。它们决定各 buffer 的尺寸，即使拓扑相同也必须逐模型读取；cache 张量的基本形状及不同 layer 类型由官方缓存说明明确区分（[Caching 文档](https://huggingface.co/docs/transformers/main/en/cache_explanation#cache-storage-implementation)）。
3. **运行时 cache 实现**：Dynamic、Static、offloaded、quantized 等是调用方/生成配置的选择，会改变预分配、GPU 常驻和存储精度，但不等于模型自身的 attention 拓扑（[官方 KV cache 策略](https://huggingface.co/docs/transformers/main/en/kv_cache)）。显存估算必须声明目标运行时策略，不能从 `architectures` 推断。

## 对重构的约束

- 可以继续维护**人工审核的显式目录**，但目录键不能只有裸 `architectures` 字符串。
- 同一字符串存在多种已验证布局时，应建立不同的内部 architecture variant/layout 条目，并在显式清单中同时声明必要的 config 判别条件（例如 `sliding_window`、`layer_types`）；运行时只选择一个完整布局，不拼装多个布局。
- 未命中完整的“类名 + 已审核判别条件”时应 fail closed；不能把裸类名命中视为已经准确识别布局。
- 运行时 cache 实现应作为独立输入或估算场景，而不是布局目录的一部分隐式猜测。
