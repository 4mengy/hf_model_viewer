# LLM 架构与显存计算器

[English](./README.md) | [简体中文](./README.zh-CN.md)

> 零下载解析 Hugging Face 模型拓扑，并通过已验证的模型架构档案动态估算显存（VRAM）。首批档案覆盖 GLM 5.2、DeepSeek V4 Pro 与 Hy3。同一套代码同时发布为 Web 应用和 Manifest V3 浏览器扩展。

### 功能特性

- **零下载解析** — 通过 HTTP Range 请求仅读取 `safetensors` 文件头部的 JSON 元数据，无需下载权重数据，可在数秒内解析超大型 MoE 仓库。
- **动态 VRAM 估算** — 根据解析出的张量 Shape 与 DType 逐张量计算权重显存，再叠加已验证的 KV Cache payload。
- **已验证 KV Cache 档案** — 使用精确的模型类标识选择人工审核的候选档案，并验证完整配置和 safetensors 元数据签名。未知模型会安全失败，不使用通用公式猜测。
- **首批专用布局** — GLM 5.2 IndexShare、DeepSeek V4 Pro HCA/CSA（包含 indexer 和 compressor state），以及 Hy3 full-context GQA。每个档案拥有一套完整的专用布局。
- **可审计明细** — 展示 Profile/layout 版本，以及每类 buffer 的层组、元素数、DType、字节数、公式和固定 revision 证据。
- **按模块量化策略** — `uniform`（全部量化）、`keep-fp16`（仅 Linear）和 `native`（使用磁盘 DType）。已预量化权重会保持原始精度，例如 FP4 仍按 FP4 计算。
- **细粒度组成** — 总览按 Tensor Name Pattern 拆分显存，并叠加 KV。共享专家（始终激活，每层恰好一个）与路由专家（×N，每个 token 只激活少量专家）会明确区分。
- **双形态发布** — 同一份源码同时构建为 GitHub Pages 静态站点和浏览器扩展。
- **双语界面** — Web 应用和浏览器扩展均支持一键切换中文与英语，并在本地保存语言偏好。

### 设计原则

估算器只统计模型语义要求常驻 GPU 的有效 payload，不包含 top-k workspace、框架容量预留、内存分配器碎片、offload 和可选 speculative runtime。如果任何必要组成未知，总显存也保持未知。

### 数学模型

```
Vtotal   = Vweights + Vkv_cache
Vweights = Σ params × B_precision / 1024³
Vkv      = Σ verified-profile buffer bytes / 1024³
```

GLM 5.2 的已验证语义 payload 为 `B × S × 95,232 bytes`，Hy3 为 `B × S × 327,680 bytes`。DeepSeek V4 Pro 分别计算 HCA/CSA 的本地与压缩 KV、indexer KV 和 FP32 compressor live state；界面的审计视图和固定研究资产会展示完整的逐 buffer 公式。

公开的 `computeKV(...)` 和 `estimateVRAM(...)` API 也接受 `sequenceLengths: number[]`。对于 ragged batch，GLM 5.2 和 Hy3 使用 `Σ sequenceLengths`；DeepSeek V4 Pro 会对每条序列分别处理窗口与压缩边界，再汇总每个 buffer。

### 本地开发

```bash
npm install
npm run dev                           # 本地开发服务器（Web 形态）
BUILD_TARGET=web npm run build:web    # 构建 GitHub Pages 站点 -> dist-web/
BUILD_TARGET=ext npm run build:ext    # 构建浏览器扩展包 -> dist-ext/
```

### 部署到 GitHub Pages

推送提交后，GitHub Actions 会通过 `.github/workflows/deploy.yml` 自动部署：

```bash
git push -u origin main
```

随后在仓库的 `Settings → Pages → Source` 中选择 **GitHub Actions**。站点地址：
`https://<user>.github.io/hf_model_viewer/`

### 构建为浏览器扩展

```bash
BUILD_TARGET=ext npm run build:ext
```

在浏览器扩展管理页面中加载解压后的 `dist-ext/` 目录，并启用“开发者模式”。扩展通过 Background Service Worker 代理网络请求，以绕过网页端 CORS 限制。
