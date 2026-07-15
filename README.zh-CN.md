# LLM VRAM 计算器 — 零下载 GPU 显存估算工具

[English](./README.md) | [简体中文](./README.zh-CN.md)

> 无需下载模型权重，即可解析任意 Hugging Face 模型的拓扑结构，并结合已验证的架构档案（Architecture Profile）或明确标记为近似值的通用 KV Cache 回退估算 GPU 显存。首批档案覆盖 GLM 5.2、DeepSeek V4 Pro、Hunyuan 3 和 Qwen 3.6 35B A3B。

## 使用 skill 贡献 KV Cache Profile

仓库提供 [`add-kv-cache-profile`](skills/add-kv-cache-profile/SKILL.md) agent skill，可以把一个模型接入请求转化为可审计的代码贡献。它会固定上游 revision、追踪所有持久化 decode state、推导独立 golden vectors、实现 Architecture Profile，并运行要求的 assurance 与 workload 测试。

先 fork 并 clone 仓库、安装 JavaScript 依赖，再把 skill 链接到个人 skills 目录，让 Codex 可以发现它：

```bash
git clone https://github.com/<your-account>/hf_model_viewer.git
cd hf_model_viewer
npm install

SKILLS_HOME="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$SKILLS_HOME"
ln -s "$(pwd)/skills/add-kv-cache-profile" "$SKILLS_HOME/add-kv-cache-profile"
```

从仓库根目录启动 Codex task，提供规范的 Hugging Face 仓库标识，以及已知情况下需要审计的 revision：

```text
使用 $add-kv-cache-profile 为 <organization>/<model> 在
<branch、tag 或 commit> revision 上添加专用 Architecture Profile。
```

对于不能自动发现 Codex skills 的 agent，可以直接提供 skill 文件：

```text
按照 skills/add-kv-cache-profile/SKILL.md 调查、实现并验证
<organization>/<model> 的 KV Cache Profile。
```

提交 pull request 前，应检查生成的 evidence ledger，确认每个持久化 GPU-resident decode state 都已映射到一个 buffer 或有明确的排除理由。贡献必须包含不可变的上游证据、独立 golden vectors、config 与 commit drift 覆盖、模型特有的 workload 边界，以及成功的 `npm test` 和 `npm run build` 结果。如果现有证据无法建立完整的 KV Cache Layout，应提交包含调查缺口的 [issue](https://github.com/4mengy/hf_model_viewer/issues)，而不是把通用估算描述为已验证结果。

### 功能特性

- **零下载解析** — 通过 HTTP Range 请求仅读取 `safetensors` 文件头部的 JSON 元数据，无需下载权重数据，可在数秒内解析超大型 MoE 仓库。
- **动态显存估算** — 根据张量的磁盘 DType 与 Shape 逐张量计算权重显存，再叠加来自专用 Architecture Profile 或通用回退的 Effective KV Cache Payload。
- **按仓库路由的 Architecture Profile** — 规范的 Model Repository Identifier 选择唯一专用 Profile。当前 commit 与算法依赖的全部 config 输入都会和审计基线比较；发生漂移时产生 warning，同时继续使用有效的当前输入计算。
- **通用 KV Cache 回退** — 没有专用 Profile 的仓库会根据当前 config 近似计算统一的 full-context MHA/GQA key/value cache。结果始终标记为 `approximate` 并公开全部假设，绝不会表示成 Verified Layout。
- **首批专用布局** — GLM 5.2 IndexShare、DeepSeek V4 Pro HCA/CSA（含 indexer 和 compressor state）、Hunyuan 3 full-context GQA，以及 Qwen 3.6 hybrid Gated DeltaNet/GQA。每个 Profile 拥有一套完整 Layout。
- **可审计明细** — 展示仓库 provenance、Calculation Assurance、结构化 config 差异、Architecture Profile/KV Cache Layout 版本，以及每类 buffer 的层组、元素数、DType、字节数和公式。
- **细粒度组成** — 总览图按 Tensor Name Pattern 拆分显存组成并叠加 KV Cache，每个条目旁标注其磁盘 DType。
- **双语界面** — 一键切换中文 / 英语，语言偏好保存在本地。

### 设计原则

估算器仅统计模型语义要求常驻 GPU 的有效载荷，不包含 top-k 工作区、框架容量预留、内存分配器碎片、offload 及可选 speculative runtime。若任一必要组成未知，总显存也保持未知。

### 数学模型

```
Vtotal   = Vweights + Vkv_cache
Vweights = Σ params × B_dtype / 1024³
Vkv      = Σ dedicated-profile or generic-fallback buffer bytes / 1024³
Vgeneric = 2 × tokens × layers × kv_heads × head_dim × dtype_bytes / 1024³
```

GLM 5.2 的已验证语义载荷为 `B × S × 95,232 bytes`，Hunyuan 3 为 `B × S × 327,680 bytes`。DeepSeek V4 Pro 分别计算 HCA/CSA 的本地与压缩 KV、indexer KV 和 F32 compressor live state。Qwen 3.6 35B A3B 由十个 full-attention 层每 token `20,480 bytes` 的载荷，以及三十个 linear-attention 层每 active sequence `64,389,120 bytes` 的 BF16 convolution history 和 F32 recurrent state 组成。界面审计视图展示完整的逐 buffer 公式。

公开的 `computeKV(...)` 接口要求提供 `source: { repoId, commitId }`、当前 config 和 workload；`estimateVRAM(...)` 还需要相同 provenance 和 tensor metadata。两者都接受 `sequenceLengths: number[]`。对于 ragged batch，GLM 5.2 和 Hunyuan 3 使用 `Σ sequenceLengths`；DeepSeek V4 Pro 会对每条序列独立计算窗口与压缩边界；Qwen 3.6 只为 active sequence 增加 recurrent state。

Calculation Assurance 与估算完整性相互独立。审计 commit 与全部算法输入精确匹配时为 `verified`；有效的新 config 仍然可以生成 Complete VRAM Estimate，但会标记为 `warning` 并报告当前 commit、审计 commit 和结构化 config 差异。没有 Profile 时，通用计算使用 `num_hidden_layers`、attention/KV head 数、`head_dim` 或 `hidden_size / num_attention_heads`，以及配置的 cache/model dtype；缺失或未知 dtype 时默认使用 BF16。包含 sliding window、recurrent state、latent/compressed cache 或其他自定义语义的模型可能存在显著差异，因此该路径始终保持 `approximate`。

### 本地开发

```bash
npm install
npm run dev        # 本地开发服务器
npm run build      # 构建静态站点 -> dist-web/
npm test           # 运行测试套件
```
