Type: research
Status: resolved
Blocked by:

## Question

DeepSeek V4 Pro 的第一方配置与推理实现定义了哪些常驻 KV Cache buffer、逐层混合模式、压缩/窗口/索引器语义、dtype 和尺寸参数；哪些 `config.json` 与 safetensors 元数据足以在不执行远程代码的情况下唯一识别其 Architecture Profile，并据此独立推导 Effective KV Cache Payload？

## Answer

见 [`DeepSeek V4 Pro KV Cache 布局调查`](../research/deepseek-v4-pro-kv-layout.md)。结论已落实为一个完整 HCA/CSA 专用布局，逐项统计 local/compressed KV、indexer KV 与 FP32 compressor live state，并验证 instruct FP4 checkpoint 身份。
