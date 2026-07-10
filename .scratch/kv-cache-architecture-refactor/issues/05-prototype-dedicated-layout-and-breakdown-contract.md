Type: prototype
Status: resolved
Blocked by: 01, 02, 03

## Question

什么样的专用 KV Cache Layout 与计算结果 contract，能够完整表达三个目标 Profile 的逐 buffer/层组拓扑、元素数、dtype、字节数、证据和验证状态，同时把共享限制在不会掩盖 Profile 级公式与端到端正确性的基础设施？

## Answer

三个完整布局分别位于 `src/vram/kv/profiles/`。共享 `profile-result.js` 只构造 buffer、执行安全整数/字节换算、汇总与 verified/unknown 结果；所有层组、保留策略和公式都由 Profile 本地定义并由端到端 golden tests 覆盖。
