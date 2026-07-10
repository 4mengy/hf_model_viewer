# LLM Architecture & VRAM Calculator

> 零下载解析 Hugging Face 模型拓扑，并通过已验证的模型架构档案计算动态显存（VRAM）。首批支持 GLM 5.2、DeepSeek V4 Pro 与 Hy3。Web 与浏览器扩展（Manifest V3）同一份代码双形态分发。
>
> Zero-download parsing of Hugging Face model topology with dynamic VRAM estimation from verified Architecture Profiles. Initial profiles cover GLM 5.2, DeepSeek V4 Pro, and Hy3. One codebase ships as both a web app and a Manifest V3 browser extension.

[English](#english) | [中文](#中文)

---

## 中文

### 功能特性

- **零下载解析**：通过 HTTP Range 请求只读取 `safetensors` 文件头部的 JSON 元数据，不下载任何权重数据，秒级解析超大 MoE 仓库。
- **动态 VRAM 计算**：基于解析出的张量形状与 dtype，逐张量计算权重显存，并叠加 KV Cache 与固有开销。
- **已验证 KV Cache 档案**：模型类标识只用于查找人工审核的候选档案；档案必须同时验证完整 `config` 与 safetensors 元数据签名，未命中时 fail closed，不再用通用公式猜测。
- **首批专用布局**：GLM 5.2 IndexShare、DeepSeek V4 Pro HCA/CSA（含 indexer 与 compressor state）、Hy3 full-context GQA。每个档案只使用自己的完整布局。
- **可审计明细**：展示 Profile/layout 版本，以及每类 buffer 的层组、元素数、dtype、字节数、公式和固定 revision 证据。
- **按模块量化策略**：`uniform`（全量化）/` keep-fp16`（仅 Linear）/ `native`（按磁盘实际 dtype）。尊重已预量化的权重——本就是 FP4 就按 FP4 计。
- **细粒度组成明细**：总览按张量类别（嵌入 / 注意力 / MLP / 归一化 / LM Head / MoE 路由专家 / 共享专家）拆解显存，并叠加 KV 与开销。共享专家（始终激活，每模型 1 个）与路由专家（×N，每 token 仅激活少数）在界面与组成中清晰区分。
- **双形态分发**：同一份源码同时构建为 GitHub Pages 静态站点与浏览器扩展。
- **中英双语界面**：Web 与扩展均提供中文 / English 一键切换，选择记忆在本地。

### 架构原则

显存计算器只统计模型语义要求常驻 GPU 的有效 payload，不把 top-k workspace、框架预分配余量、allocator 碎片、offload 或可选 speculative runtime 混入结果。任何必要组成未知时，总显存也保持未知。

### 数学模型

```
Vtotal   = Vweights + Vkv_cache + Voverhead
Vweights = Σ params × B_precision / 1024³
Vkv      = Σ verified-profile buffer bytes / 1024³
Voverhead = 2.0 + Vweights × 10%
```

GLM 5.2 的已验证语义 payload 为 `B × S × 95,232 bytes`；Hy3 为 `B × S × 327,680 bytes`。DeepSeek V4 Pro 按 HCA/CSA 层组分别计算 local/compressed KV、indexer KV 和 FP32 compressor live state，完整逐项公式在界面审计明细与固定研究资产中展示。

### 本地开发

```bash
npm install
npm run dev            # 本地开发服务器（web 形态）
BUILD_TARGET=web npm run build:web   # 构建 GitHub Pages 站点 → dist-web/
BUILD_TARGET=ext npm run build:ext   # 构建浏览器扩展包 → dist-ext/
```

### 部署到 GitHub Pages

推送后由 GitHub Actions 自动部署（` .github/workflows/deploy.yml`）：

```bash
git push -u origin main
```

随后在仓库 `Settings → Pages → Source` 选择 **GitHub Actions**。站点地址：
`https://<user>.github.io/hf_model_viewer/`

### 构建为浏览器扩展

```bash
BUILD_TARGET=ext npm run build:ext
```

在浏览器扩展管理页加载解压的 `dist-ext/` 目录（需开启「开发者模式」）。扩展通过 Background Service Worker 代理网络请求，突破网页端 CORS 限制。

---

## English

### Features

- **Zero-download parsing** — reads only the `safetensors` header JSON via HTTP Range requests, never downloading weight data; parses huge MoE repos in seconds.
- **Dynamic VRAM estimation** — computes per-tensor weight VRAM from parsed shapes and dtypes, then adds KV Cache and fixed overhead.
- **Verified KV Cache Profiles** — exact model-class identifiers select manually reviewed candidates, which must validate their complete config and safetensors metadata signatures. Unknown models fail closed; no generic formula is guessed.
- **Initial dedicated layouts** — GLM 5.2 IndexShare, DeepSeek V4 Pro HCA/CSA (including indexer and compressor state), and Hy3 full-context GQA. Each Profile owns one complete layout.
- **Auditable breakdown** — exposes Profile/layout versions and every buffer group's layers, element count, dtype, bytes, formula, and fixed-revision evidence.
- **Per-module quantization strategy** — `uniform` (quantize all) / `keep-fp16` (Linear only) / `native` (use on-disk dtype). Respects pre-quantized weights — already FP4 stays FP4.
- **Fine-grained composition** — overview breaks VRAM down by tensor category (embedding / attention / MLP / norm / LM Head / MoE routed experts / shared expert), plus KV and overhead. The shared expert (always active, exactly 1 per layer) is clearly separated from the routed experts (×N, only a few active per token).
- **Dual-form delivery** — one source tree builds both a GitHub Pages static site and a browser extension.
- **Bilingual UI** — web and extension both offer one-click Chinese / English switching, persisted locally.

### Design principle

The estimator counts the effective payload that model semantics require to remain GPU-resident. It excludes top-k workspaces, framework capacity padding, allocator fragmentation, offload, and optional speculative runtimes. If any required component is unknown, total VRAM remains unknown too.

### Math model

```
Vtotal   = Vweights + Vkv_cache + Voverhead
Vweights = Σ params × B_precision / 1024³
Vkv      = Σ verified-profile buffer bytes / 1024³
Voverhead = 2.0 + Vweights × 10%
```

The verified semantic payload is `B × S × 95,232 bytes` for GLM 5.2 and `B × S × 327,680 bytes` for Hy3. DeepSeek V4 Pro separately accounts for HCA/CSA local and compressed KV, indexer KV, and FP32 compressor live state; the UI audit view and fixed research assets expose the complete per-buffer formulas.

### Local development

```bash
npm install
npm run dev                           # local dev server (web form)
BUILD_TARGET=web npm run build:web    # build GitHub Pages site -> dist-web/
BUILD_TARGET=ext npm run build:ext    # build browser extension bundle -> dist-ext/
```

### Deploy to GitHub Pages

Pushed commits deploy automatically via GitHub Actions (`.github/workflows/deploy.yml`):

```bash
git push -u origin main
```

Then in the repo `Settings → Pages → Source` choose **GitHub Actions**. Site URL:
`https://<user>.github.io/hf_model_viewer/`

### Build as a browser extension

```bash
BUILD_TARGET=ext npm run build:ext
```

Load the unpacked `dist-ext/` directory in your browser's extension manager (enable "Developer mode"). The extension proxies network requests through a Background Service Worker to bypass page-side CORS.
