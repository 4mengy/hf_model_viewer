Label: wayfinder:map

## Destination

产出一份可直接实施的 KV Cache 计算重构规格：以人工审核的 Architecture Profile 目录替代通用启发式识别，并明确专用布局、验证准入、可审计输出与安全切换方案；本地图不修改 KV Cache 业务代码。

## Notes

- 领域：LLM 推理显存估算；每个 session 先读 [`CONTEXT.md`](../../CONTEXT.md)。
- 首批且仅首批支持 GLM 5.2、DeepSeek V4 Pro、Hunyuan 3；未验证模型 fail closed。
- 每个 Architecture Profile 唯一对应一个专用完整布局；多个纯模型类别名可归入同一 Profile。完整布局不跨 Profile 复用，只允许复用 buffer 原语、单位换算、逐层汇总与校验基础设施。
- 复用基础设施不构成正确性证据；每个 Profile 必须独立满足第一方依据、公式推导、端到端 golden tests，以及复杂布局的官方 cache shape 或实测显存交叉核对。
- 计算口径是指定 batch/context 下的 Effective KV Cache Payload；cache dtype 由已验证布局逐 buffer 定义，权重精度不影响 KV。
- 运行时 Profile 识别只读取 `config.json`、safetensors 张量元数据和内置人工目录；不得下载或执行远程模型代码。
- 未命中、冲突或证据不足时 KV 与总显存均为未知；权重和固定开销可单独展示。
- 正式输出必须包含 Profile/layout 版本、逐 buffer/层组元素数、dtype、字节数、GB、总 KV、证据与验证状态。
- 外部模型事实使用 `/research`；接口与行为原型使用 `/prototype`；领域词汇使用 `/domain-modeling`；需要产品取舍时使用 `/grilling`。
- 基础调查：[`architectures` 与 KV Cache 布局是否一一对应](research/architectures-vs-kv-layout.md)（结论：不一一对应；裸模型类标识不能作为完整 layout 身份）。

## Decisions so far

- [`01`](issues/01-investigate-glm-5-2-kv-layout.md): GLM 5.2 使用独立 IndexShare 布局；BF16 与官方 FP8 权重 checkpoint 是分别验签的显式 alias，KV 语义 dtype 均为 BF16。
- [`02`](issues/02-investigate-deepseek-v4-pro-kv-layout.md): DeepSeek V4 Pro 使用一个完整 HCA/CSA 专用布局，包含 local/compressed KV、indexer KV 与 FP32 compressor live state。
- [`03`](issues/03-investigate-hunyuan-3-kv-layout.md): 正式目标锁定 `tencent/Hy3`；主体为 80 层 full-context GQA，MTP 只作 checkpoint 身份验证并从核心 payload 排除。
- [`04`](issues/04-prototype-profile-catalog-and-resolver-contract.md): `config.architectures` 仅选择人工目录候选，每个 Profile 再独立验证 config 与 tensor metadata；未知、冲突或漂移均 fail closed。
- [`05`](issues/05-prototype-dedicated-layout-and-breakdown-contract.md): 每个 Profile 只调用自己的完整布局；共享层仅负责 buffer 结果结构、安全整数、单位换算与总和验证。
- [`06`](issues/06-prototype-profile-evidence-and-golden-test-gate.md): 固定 revision 证据、signature drift 测试、边界 golden vectors 与公开 API 端到端断言共同构成准入门槛。
- [`07`](issues/07-prototype-auditable-vram-result-and-ui.md): 结果公开 Profile/layout 版本、逐 buffer 明细与证据；KV unknown 时完整总显存保持 unknown。
- [`08`](issues/08-decide-cutover-to-verified-profile-calculation.md): 已移除生产启发式回退与旧 MHA/MLA/DSA 注册表，首批目录之外的模型不再估算 KV。

## Not yet specified

- 当前首批范围没有未决项。新的 checkpoint alias、KV dtype 或 runtime scenario 必须另做人工专项验证后显式加入目录。

## Implementation outcome

Wayfinder 阶段完成后，用户显式启动了实施阶段。实现位于 `src/vram/kv/`，公开入口为 `computeKV(...)` 与 `estimateVRAM(...)`，回归与 golden tests 位于 `test/vram/`。

## Out of scope

- 部署 KV Cache 重构。
- GLM 5.2、DeepSeek V4 Pro、Hunyuan 3 之外模型的 Profile 与布局支持。
- 推理框架预分配余量、allocator 碎片、offload、Dynamic/Static cache 等实际运行时内存策略。
- 全局或用户可选的 KV cache dtype 覆盖模式。
- 运行时下载或执行模型仓库的远程代码。
