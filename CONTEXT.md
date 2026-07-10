# LLM 显存估算

本上下文描述模型推理显存估算中的领域语言，重点区分模型身份、注意力机制与实际驻留显存的 KV Cache。

## Language

**模型类标识（Model Class Identifier）**:
`config.architectures[]` 中声明可加载该组预训练权重的模型类名称；它不是 KV Cache 布局的完备身份。
_Avoid_: architect、布局标识

**模型架构（Model Architecture）**:
决定一套完整 KV Cache 存储语义的模型级架构类别；它不是 MHA、GQA、MLA 等单一注意力机制的同义词。
_Avoid_: 模型类标识、注意力架构

**模型架构档案（Architecture Profile）**:
经专项识别与人工审核确定的模型架构变体；每个档案唯一对应一个专用的完整 KV Cache 布局，纯类名别名可归入同一档案。
_Avoid_: 通用架构猜测、布局模板

**KV Cache 布局（KV Cache Layout）**:
一个模型架构在推理期间需要驻留显存的全部缓存结构；混合机制模型也拥有一个专用的完整布局，而不是多个既有布局的组合。
_Avoid_: 混合布局、layout 拼装

**KV Cache Buffer 原语（KV Cache Buffer Primitive）**:
完整布局内部一种明确的驻留缓存结构；它可以复用表达与计算机制，但不能替代模型架构档案自己的完整布局定义和端到端验证。
_Avoid_: 通用布局、共享公式即正确

**已验证布局（Verified Layout）**:
拥有第一方实现依据、独立显存推导、档案级端到端 golden tests，并在复杂情况下经过官方 cache shape 或实测显存交叉核对的专用布局；未达到证据门槛的布局不得作为已支持能力。
_Avoid_: 推测支持、仅原语测试通过

**有效 KV Cache Payload（Effective KV Cache Payload）**:
在指定 batch 与 context 下，模型语义要求常驻 GPU 的全部 KV Cache 数据量，包含模型自身的压缩、窗口、索引器与 cache dtype，但不包含推理框架预留容量、内存分配器碎片或 offload 策略。
_Avoid_: 框架实际分配量、CUDA reserved memory

**完整显存估算（Complete VRAM Estimate）**:
只有权重、已验证 KV Cache payload 与固定开销均已知时才能给出的总显存结果；任一必要组成未知时，总显存也必须保持未知。
_Avoid_: 将未知 KV 按零计、部分总显存

**架构布局目录（Architecture Layout Catalog）**:
经人工审查或专项验证形成的显式清单，将模型类标识通过该模型的专用识别过程映射到一个模型架构档案及其唯一 KV Cache 布局；目录项不得由名称模式或通用启发式规则自动推断。
_Avoid_: 自动架构猜测、规则路由

**不支持的模型架构（Unsupported Model Architecture）**:
未被架构布局目录明确收录的模型架构，其 KV Cache 显存占用为未知；不得通过通用公式或启发式回退给出估算值。
_Avoid_: 猜测结果、默认 MHA
