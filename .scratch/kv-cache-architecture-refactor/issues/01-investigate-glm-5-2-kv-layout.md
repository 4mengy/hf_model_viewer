Type: research
Status: resolved
Blocked by:

## Question

GLM 5.2 的第一方配置与推理实现定义了哪些常驻 KV Cache buffer、逐层模式、保留策略、dtype 和尺寸参数；哪些 `config.json` 与 safetensors 元数据足以在不执行远程代码的情况下唯一识别其 Architecture Profile，并据此独立推导 Effective KV Cache Payload？

## Answer

见 [`GLM 5.2 KV Cache 布局调查`](../research/glm-5-2-kv-layout.md)。结论已落实为 `glm-5.2-semantic-bf16-v1` 与专用 `glm-5.2-indexshare-bf16-v1`；BF16/FP8 官方权重别名分别验证完整 cache-related dtype/scale 签名。
