## Stress 场景 Benchmark：Baseline Eager vs Optimized CUDA Graph

### 1. 测试目的

验证多请求 Chunked Prefill 优化在高并发、有限 Token Budget 场景下的表现，重点观察：

- Decode 优先调度对 Decode 延迟的影响
- 多请求 Chunked Prefill 对 Prefill 调度的影响
- CUDA Graph 对 Decode 执行效率的影响
- 整体吞吐量、TTFT、TPOT 和请求延迟的变化

### 2. 测试环境与配置

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 3060 Laptop 6GB |
| 模型 | Qwen3-0.6B |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| Python | 3.11.16 |
| Requests | 16 |
| Prompt Tokens | 512 |
| Output Tokens | 8 |
| Max Batched Tokens | 512 |
| Seed | 42 |
| Benchmark Runs | 每组 5 轮 |

执行模式：

| 版本 | Prefill | Decode |
|---|---|---|
| Baseline | Eager | Eager |
| Optimized | Eager | CUDA Graph Replay |

说明：Optimized 并不是整个推理过程都使用 CUDA Graph。Prefill 仍然使用 Eager，只有符合条件的 Decode Batch 使用 CUDA Graph Replay。

### 3. 原始 Benchmark 数据

#### 3.1 Baseline（Eager）

| Run | Total Time (ms) | Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | Request Latency (ms) | Steps |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1688.45 | 75.81 | 616.26 | 153.16 | 1688.35 | 23 |
| 2 | 1749.10 | 73.18 | 658.11 | 155.85 | 1749.05 | 23 |
| 3 | 1770.66 | 72.29 | 674.41 | 156.60 | 1770.60 | 23 |
| 4 | 1670.98 | 76.60 | 604.76 | 152.31 | 1670.91 | 23 |
| 5 | 1741.31 | 73.51 | 673.81 | 152.49 | 1741.25 | 23 |

#### 3.2 Optimized（Eager Prefill + CUDA Graph Decode）

| Run | Total Time (ms) | Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | Request Latency (ms) | Steps |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1545.50 | 82.82 | 1338.26 | 22.77 | 1497.62 | 30 |
| 2 | 1502.55 | 85.19 | 1305.05 | 20.99 | 1451.96 | 30 |
| 3 | 1522.02 | 84.10 | 1317.59 | 22.46 | 1474.79 | 30 |
| 4 | 1552.23 | 82.46 | 1360.80 | 20.59 | 1504.90 | 30 |
| 5 | 1541.48 | 83.04 | 1347.25 | 21.02 | 1494.38 | 30 |

### 4. 统计结果

以下采用五轮实验的均值和中位数进行比较。相对变化按照中位数计算。

| 指标 | Baseline Mean | Optimized Mean | Baseline Median | Optimized Median | Median Change |
|---|---:|---:|---:|---:|---:|
| Total Time (ms) | 1724.10 | 1532.76 | 1741.31 | 1541.48 | -11.48% |
| Throughput (tok/s) | 74.28 | 83.52 | 73.51 | 83.04 | +12.96% |
| Mean TTFT (ms) | 645.47 | 1333.79 | 658.11 | 1338.26 | +103.35% |
| Mean TPOT (ms) | 154.08 | 21.57 | 153.16 | 21.02 | -86.28% |
| Request Latency (ms) | 1724.03 | 1484.73 | 1741.25 | 1494.38 | -14.18% |
| Engine Steps | 23 | 30 | 23 | 30 | +30.43% |

注：
- Total Time、TTFT、TPOT 和 Request Latency 越低越好。
- Throughput 越高越好。
- Median Change 为 Optimized 相对于 Baseline 的中位数变化。
- 耗时指标的负变化表示耗时下降；吞吐量的正变化表示吞吐量提高。

### 5. 显存对比

| 指标 | Baseline | Optimized |
|---|---:|---:|
| Peak Allocated | 4092.02 MiB | 4100.19 MiB |
| Peak Reserved | 4152.00 MiB | 4176.00 MiB |

Optimized 的 CUDA Graph 带来了少量额外显存开销。本轮测试没有观察到显存占用下降。

### 6. 结果分析

#### 6.1 Decode 性能明显改善

Mean TPOT 从 153.16 ms 降低到 21.02 ms，中位数下降约 86.28%。

这说明 Optimized 执行路径下的 Decode 性能有明显改善。CUDA Graph Replay 能够减少 CPU 侧逐步提交 GPU Kernel 的开销，对 Decode 这种小 Batch、重复执行的场景比较有利。

#### 6.2 整体吞吐量提高

Output Throughput 中位数从 73.51 tok/s 提升到 83.04 tok/s，提升约 12.96%。

总耗时中位数下降约 11.48%，说明在本次 Stress 负载下，Optimized 获得了更高的整体执行效率。

#### 6.3 TTFT 明显退化

Mean TTFT 中位数从 658.11 ms 增加到 1338.26 ms，增加约 103.35%。

主要需要关注 Decode 严格优先策略带来的调度影响：当系统中存在多个 Decode 请求时，新进入的 Prefill 请求可能需要等待，导致首 Token 延迟增加。

这表明当前方案更偏向优化 Decode 服务效率，而不是同时优化 Prefill 和 Decode 的延迟。

#### 6.4 Benchmark 稳定性

五轮实验中：

- Baseline Total Time 标准差约为 42.37 ms。
- Optimized Total Time 标准差约为 20.28 ms。

在本次实验中，Optimized 的总耗时波动更小，表现出较好的重复性。

### 7. 阶段性结论

本次 Stress Benchmark 表明：

1. 多请求 Chunked Prefill 与 Decode 优先调度能够在有限 Token Budget 下实现 Decode 和 Prefill 的混合调度。
2. Optimized 在本次负载下获得了约 12.96% 的吞吐量提升。
3. Mean TPOT 中位数下降约 86.28%，Decode 延迟改善明显。
4. Mean TTFT 中位数增加约 103.35%，Prefill 请求等待时间成为当前方案的主要问题。
5. CUDA Graph 带来少量额外显存开销。
6. 当前结果不能证明 CUDA Graph 单独贡献了全部性能提升，因为 Baseline 和 Optimized 同时存在调度策略与执行模式的差异。

**总体结论：** 当前优化在 Stress 场景下改善了 Decode 性能和整体吞吐量，但牺牲了 Prefill 首 Token 延迟。这是一种有明确性能取舍的调度优化，而不是所有指标都得到改善的通用优化。

### 8. 后续实验计划

| 阶段 | 实验内容 | 目的 | 状态 |
|---|---|---|---|
| 1 | Stress：16 Requests，Prompt 512，Budget 512 | 验证高并发负载 | 已完成 |
| 2 | Staggered：分批提交请求 | 观察在线 Decode 与新 Prefill 请求之间的干扰 | 待测试 |
| 3 | All-at-once：同时提交请求 | 观察普通批量请求场景 | 待测试 |
| 4 | CUDA Graph Ablation | 分离调度策略和 CUDA Graph 的贡献 | 待测试 |
| 5 | 最终性能总结 | 整理适用场景、性能收益和局限 | 待完成 |

CUDA Graph Ablation 计划：

| 组别 | Scheduler | Prefill | Decode |
|---|---|---|---|
| A | Baseline | Eager | Eager |
| B | Optimized | Eager | Eager |
| C | Optimized | Eager | CUDA Graph |

通过 A/B 对比调度策略的贡献，通过 B/C 对比 CUDA Graph 对 Decode 性能的贡献。
