Type: prototype
Status: resolved
Blocked by: 04, 05

## Question

三个目标 Architecture Profile 各自需要哪些固定 revision 的第一方证据、独立推导、边界向量、端到端 golden tests、cache shape 或实测显存核对，才能通过 Verified Layout 准入；证据漂移时如何撤销支持并回到 fail closed？

## Answer

每个 Profile 内嵌固定 revision 证据，研究资产记录独立推导与边界向量；`test/vram/kv-profile.test.js` 通过公开 `computeKV(...)` 验证逐 buffer golden、窗口/压缩边界、最大 context、dtype/scale、额外层和签名漂移。任一 predicate 不符即自动回到 unknown。
