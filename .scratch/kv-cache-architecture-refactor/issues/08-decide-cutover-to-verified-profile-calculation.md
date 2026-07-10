Type: grilling
Status: resolved
Blocked by: 06, 07

## Question

应以什么依赖顺序、兼容边界和验收条件，把现有 tensor-first/config-fallback 启发式注册表切换为 Verified Architecture Profile 计算，并移除生产 fallback，形成可直接交给实现阶段的重构规格？

## Answer

切换已完成：先固定三份研究与 golden vectors，再建立目录/结果 contract、接入完整估算与 UI，最后删除旧的 MHA/MLA/DSA/DeepSeek heuristic modules。公开 seam 保持 `computeKV(...)`/`estimateVRAM(...)`，但未验证模型的 KV 与总显存现在明确 unknown。
