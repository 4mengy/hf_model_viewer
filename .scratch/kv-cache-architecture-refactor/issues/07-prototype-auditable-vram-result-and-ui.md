Type: prototype
Status: resolved
Blocked by: 04, 05, 06

## Question

计算 API 与界面应如何呈现 Profile/layout 版本、逐 buffer/层组明细、证据、完整显存估算，以及 unknown/conflict/unsupported 的诊断，使用户能审计结果且绝不会把未知 KV 按零计入总显存？

## Answer

`estimateVRAM(...)` 透传 Profile、buffer 与结构化诊断；KV 未验证时 `vKV`/`vTotal` 为 `null` 且组成图不添加 KV。界面新增审计表，展示 Profile/layout 版本、层组、元素、dtype、bytes、公式与固定证据链接。
