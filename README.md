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

## 真实文本验证与 Benchmark

真实文本回归覆盖中英文问答、代码、长上下文、等长 batched prefill、变长
packed prefill、chunked prefill、continuous batching 和 KV slot/page 复用。

```bash
DEEPSEEK_V4_CHECKPOINT=/models/deepseek-v4-mini-1B-from-flash \
python -m pytest tests/test_text_generation_integration.py -v -s
```

性能基线：

```bash
DEEPSEEK_V4_CHECKPOINT=/models/deepseek-v4-mini-1B-from-flash \
python -m benchmarks.benchmark_text_generation \
  --max-new-tokens 16 \
  --max-prefill-tokens 128 \
  --json-output benchmark.json
```

- TTFT: 请求提交到第一个输出 token 的时间。
- TPOT: 第一个 token 之后，每个输出 token 的平均时间。
- ITL: 相邻输出 token 的实际时间间隔。
- Output throughput: 整个场景每秒生成的 token 数。
- Prefill/decode throughput: 使用 CUDA Event 统计的模型调用吞吐。
- Peak GPU memory / KV pages: 峰值显存和最大同时占用的 KV page 数。

Benchmark 会先 warmup，再测试等长 batch 和混合 continuous/chunked 场景，最后
用完整 forward 校验前两个输出 token。当前结果代表 PyTorch reference backend，
后续 CPU/GPU offload、Shadow Radix 和 SM120 kernel 使用同一脚本对比。
