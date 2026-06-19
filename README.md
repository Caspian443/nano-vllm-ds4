# nano-vllm-ds4

针对 `deepseek-v4-mini-1B-from-flash` 的教学型最小推理引擎。

项目以理解原理为目标，按问题驱动的方式逐步推进：核心机制先分析、设计，
再手动实现；脚手架、检查工具和重复性工作可以使用 AI 加速。

## 当前阶段

当前从模型权重加载开始：仓库内包含 checkpoint 对应的模型架构、最小
safetensors loader，以及独立的单元测试和真实 checkpoint 集成测试。

## 已实测确认的 checkpoint 结构(1B)

- 24 层,hidden 1024,16 专家选 2,head_dim 64,q/o_lora_rank 384。
- 总权重 1889 个(主干 1812 + mtp 77),分片齐全(index.json 一致),总约 2GB。
- 层类型(compress_ratios): sliding=[0,1,23], CSA(ratio4)=[2,4,...,22], HCA(ratio96)=[3,5,...,21]。
- hash_moe 层=[0,1](有 tid2eid),其余普通 router(有 bias)。
- indexer 仅在 CSA 层,与 compress_ratios 完全吻合。
