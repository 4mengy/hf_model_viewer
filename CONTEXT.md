# LLM VRAM Estimation

This context defines the domain language for model-inference VRAM estimation, with particular emphasis on distinguishing model identity, attention mechanisms, and the KV Cache that remains GPU-resident.

## Language

**Model Class Identifier**:
A model class name declared in `config.architectures[]` as capable of loading a set of pretrained weights; it is not a complete identity for a KV Cache Layout.
_Avoid_: Architect, layout identifier

**Model Architecture**:
A model-level architecture category that determines complete KV Cache storage semantics; it is not synonymous with an individual attention mechanism such as MHA, GQA, or MLA.
_Avoid_: Model Class Identifier, attention architecture

**Architecture Profile**:
A specifically identified and manually reviewed model-architecture variant; each Profile maps uniquely to one dedicated complete KV Cache Layout, while aliases that differ only by class name may share a Profile.
_Avoid_: Generic architecture guess, layout template

**KV Cache Layout**:
The complete set of cache structures that a Model Architecture requires to remain GPU-resident during inference; a hybrid-mechanism model owns one dedicated complete Layout rather than a composition of existing Layouts.
_Avoid_: Hybrid layout, layout composition

**KV Cache Buffer Primitive**:
A precisely defined resident cache structure within a complete Layout; it may reuse representation and accounting mechanisms but cannot replace an Architecture Profile's complete Layout definition and end-to-end verification.
_Avoid_: Generic layout, shared formula as proof of correctness

**Verified Layout**:
A dedicated Layout backed by a first-party implementation, an independent VRAM derivation, Profile-level end-to-end golden tests, and, for complex cases, cross-checks against official cache Shapes or measured VRAM; a Layout that does not meet this evidence threshold is not supported.
_Avoid_: Speculative support, primitive-only test coverage

**Effective KV Cache Payload**:
All KV Cache data that model semantics require to remain GPU-resident for a specified batch and context, including model-defined compression, windows, indexers, and cache DType but excluding framework capacity reservations, allocator fragmentation, and offload policy.
_Avoid_: Framework allocation, CUDA reserved memory

**Complete VRAM Estimate**:
A total VRAM result available only when weights and a Verified KV Cache Payload are both known; if either required component is unknown, the total must remain unknown.
_Avoid_: Treating unknown KV as zero, partial total VRAM

**Architecture Layout Catalog**:
An explicit, manually reviewed or specifically verified catalog that maps a Model Class Identifier, through model-specific identification, to one Architecture Profile and its unique KV Cache Layout; entries must not be inferred from naming patterns or generic heuristics.
_Avoid_: Automatic architecture guessing, rule-based routing

**Unsupported Model Architecture**:
A Model Architecture that is not explicitly included in the Architecture Layout Catalog; its KV Cache VRAM usage is unknown and must not be estimated through a generic formula or heuristic fallback.
_Avoid_: Guessed result, default MHA

**Tensor Name Tree**:
A lossless presentation of parsed tensor metadata as a left-to-right prefix tree of dot-delimited tensor names; branch rows show cumulative prefixes and direct child counts, while leaves retain the original Shape, DType, parameters, and bytes.
_Avoid_: Layer View, architecture tree

**Numeric Path Branch**:
A Tensor Name Tree branch whose accumulated path contains a standalone non-negative integer segment; it is independently collapsible and carries no inferred Layer or Expert meaning.
_Avoid_: Layer ID, Expert ID, architecture inference

**Repeated Tensor Group**:
A bottom-up display group of numeric sibling subtrees whose normalized relative keys, terminal Shapes, and terminal DTypes are identical; one representative is displayed with its repeat count and IDs relative to the current parent, while parameters and bytes include every original tensor across ancestor repetitions. The public `groupRepeatedTensorSubtrees` operation creates these groups.
_Avoid_: Architecture-based grouping, representative-only VRAM, shape-only grouping

**Tensor Name Pattern**:
A presentation-only aggregation identity for tensors whose dot-delimited names differ only in standalone non-negative integer segments, represented by `*`; unlike a Repeated Tensor Group, it does not require matching terminal Shapes or DTypes.
_Avoid_: Repeated Tensor Group, Tensor category, architecture grouping
