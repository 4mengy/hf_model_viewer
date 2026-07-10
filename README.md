# LLM Architecture & VRAM Calculator

[English](./README.md) | [简体中文](./README.zh-CN.md)

> Zero-download parsing of Hugging Face model topology with dynamic VRAM estimation from verified Architecture Profiles. Initial profiles cover GLM 5.2, DeepSeek V4 Pro, and Hy3. One codebase ships as both a web app and a Manifest V3 browser extension.

### Features

- **Zero-download parsing** — reads only the `safetensors` header JSON via HTTP Range requests, never downloading weight data; parses huge MoE repos in seconds.
- **Dynamic VRAM estimation** — computes per-tensor weight VRAM from parsed shapes and dtypes, then adds the verified KV Cache payload.
- **Verified KV Cache Profiles** — exact model-class identifiers select manually reviewed candidates, which must validate their complete config and safetensors metadata signatures. Unknown models fail closed; no generic formula is guessed.
- **Initial dedicated layouts** — GLM 5.2 IndexShare, DeepSeek V4 Pro HCA/CSA (including indexer and compressor state), and Hy3 full-context GQA. Each Profile owns one complete layout.
- **Auditable breakdown** — exposes Profile/layout versions and every buffer group's layers, element count, dtype, bytes, formula, and fixed-revision evidence.
- **Per-module quantization strategy** — `uniform` (quantize all) / `keep-fp16` (Linear only) / `native` (use on-disk dtype). Respects pre-quantized weights — already FP4 stays FP4.
- **Fine-grained composition** — overview breaks VRAM down by Tensor Name Pattern plus KV. The shared expert (always active, exactly 1 per layer) is clearly separated from the routed experts (×N, only a few active per token).
- **Dual-form delivery** — one source tree builds both a GitHub Pages static site and a browser extension.
- **Bilingual UI** — web and extension both offer one-click Chinese / English switching, persisted locally.

### Design principle

The estimator counts the effective payload that model semantics require to remain GPU-resident. It excludes top-k workspaces, framework capacity padding, allocator fragmentation, offload, and optional speculative runtimes. If any required component is unknown, total VRAM remains unknown too.

### Math model

```
Vtotal   = Vweights + Vkv_cache
Vweights = Σ params × B_precision / 1024³
Vkv      = Σ verified-profile buffer bytes / 1024³
```

The verified semantic payload is `B × S × 95,232 bytes` for GLM 5.2 and `B × S × 327,680 bytes` for Hy3. DeepSeek V4 Pro separately accounts for HCA/CSA local and compressed KV, indexer KV, and FP32 compressor live state; the UI audit view and fixed research assets expose the complete per-buffer formulas.

The public `computeKV(...)` / `estimateVRAM(...)` APIs also accept `sequenceLengths: number[]`. For ragged batches, GLM 5.2 and Hy3 use `Σ sequenceLengths`; DeepSeek V4 Pro evaluates window and compression boundaries independently for every sequence before summing each buffer.

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
