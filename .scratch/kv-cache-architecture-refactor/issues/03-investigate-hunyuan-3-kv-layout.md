Type: research
Status: resolved
Blocked by:

## Question

Hunyuan 3 的第一方配置与推理实现定义了哪些常驻 KV Cache buffer、逐层模式、保留策略、dtype 和尺寸参数；哪些 `config.json` 与 safetensors 元数据足以在不执行远程代码的情况下唯一识别其 Architecture Profile，并据此独立推导 Effective KV Cache Payload？

## Answer

见 [`Tencent Hy3 KV Cache 布局调查`](../research/hunyuan-3-kv-layout.md)。正式目标是 `tencent/Hy3@716aa724…`：80 层 full-context GQA，每 token `327,680` bytes；MTP 层仅用于 checkpoint 身份验证，核心 payload 明确排除。
