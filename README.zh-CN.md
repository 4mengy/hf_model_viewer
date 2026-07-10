# LLM VRAM 计算器 — 零下载 GPU 显存估算工具

[English](./README.md) | [简体中文](./README.zh-CN.md)

> 无需下载模型权重，即可解析任意 Hugging Face 模型的拓扑结构，并结合已验证的架构档案（Architecture Profile）估算 GPU 显存。首批档案覆盖 GLM 5.2、DeepSeek V4 Pro 和 Hunyuan 3。

### 功能特性

- **零下载解析** — 通过 HTTP Range 请求仅读取 `safetensors` 文件头部的 JSON 元数据，无需下载权重数据，可在数秒内解析超大型 MoE 仓库。
- **动态显存估算** — 根据张量的磁盘 DType 与 Shape 逐张量计算权重显存，再叠加来自已验证架构档案的 KV Cache 有效载荷。
- **已验证 KV Cache 档案** — 精确的模型类标识选择人工审核的候选档案，并验证完整配置与 safetensors 元数据签名。未知模型安全失败，不使用启发式回退。
- **首批专用布局** — GLM 5.2 IndexShare、DeepSeek V4 Pro HCA/CSA（含 indexer 和 compressor state），以及 Hunyuan 3 full-context GQA。每个档案拥有一套完整布局。
- **可审计明细** — 展示 Profile/layout 版本，以及每类 buffer 的层组、元素数、DType、字节数、公式和固定 revision 证据。
- **细粒度组成** — 总览图按 Tensor Name Pattern 拆分显存组成并叠加 KV Cache，每个条目旁标注其磁盘 DType。
- **双语界面** — 一键切换中文 / 英语，语言偏好保存在本地。

### 设计原则

估算器仅统计模型语义要求常驻 GPU 的有效载荷，不包含 top-k 工作区、框架容量预留、内存分配器碎片、offload 及可选 speculative runtime。若任一必要组成未知，总显存也保持未知。

### 数学模型

```
Vtotal   = Vweights + Vkv_cache
Vweights = Σ params × B_dtype / 1024³
Vkv      = Σ verified-profile buffer bytes / 1024³
```

GLM 5.2 的已验证语义载荷为 `B × S × 95,232 bytes`，Hunyuan 3 为 `B × S × 327,680 bytes`。DeepSeek V4 Pro 分别计算 HCA/CSA 的本地与压缩 KV、indexer KV 和 F32 compressor live state；界面审计视图展示完整的逐 buffer 公式。

公开的 `computeKV(...)` / `estimateVRAM(...)` API 也接受 `sequenceLengths: number[]`。对于 ragged batch，GLM 5.2 和 Hunyuan 3 使用 `Σ sequenceLengths`；DeepSeek V4 Pro 会对每条序列独立计算窗口与压缩边界，再汇总每个 buffer。

### 本地开发

```bash
npm install
npm run dev        # 本地开发服务器
npm run build      # 构建静态站点 -> dist-web/
npm test           # 运行测试套件
```

### 部署到 GitHub Pages

推送提交后，GitHub Actions 会通过 `.github/workflows/deploy.yml` 自动部署：

```bash
git push -u origin main
```

随后在仓库 `Settings → Pages → Source` 中选择 **GitHub Actions**。站点地址：
`https://<user>.github.io/hf_view/`
