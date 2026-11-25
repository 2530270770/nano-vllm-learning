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




## 多请求 Chunked Prefill 性能评估

### 1. 实验目的

对比原始 nano-vLLM Baseline 与多请求 Chunked Prefill Optimized 版本在不同请求负载下的推理性能，重点评估：

- Decode 优先调度对 Decode 延迟的影响。
- 多请求 Chunked Prefill 对 Prefill 调度的影响。
- CUDA Graph 对 Decode 执行效率的影响。
- 不同请求到达模式下的 TTFT、TPOT、吞吐量和请求延迟变化。

### 2. 测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 3060 Laptop 6GB |
| 模型 | Qwen3-0.6B |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| Python | 3.11.16 |
| 操作系统 | WSL Ubuntu |
| Benchmark Runs | 每组每个场景 5 轮 |

执行模式：

| 版本 | Prefill | Decode |
|---|---|---|
| Baseline | Eager | Eager |
| Optimized | Eager | CUDA Graph Replay |

说明：

Optimized 并不是整个推理过程都使用 CUDA Graph。Prefill 仍然使用 Eager，符合条件的 Decode Batch 使用 CUDA Graph Replay。

### 3. Benchmark 场景

本次设计了三种请求负载场景：

| 场景 | Requests | Prompt Tokens | Output Tokens | Token Budget | 请求提交方式 |
|---|---:|---:|---:|---:|---|
| Stress | 16 | 512 | 8 | 512 | 所有请求同时提交 |
| Staggered | 4 | 512 | 8 | 512 | 首个 Engine Step 后提交剩余请求 |
| All-at-once | 4 | 128 | 8 | 16384 | 所有请求同时提交 |

所有场景使用 Seed 42，每组执行 5 轮。

### 4. Stress 场景

#### 4.1 测试结果

Stress 用于模拟高并发、有限 Token Budget 下的请求竞争，重点观察 Decode 优先调度和多请求 Chunked Prefill 的表现。

Baseline（Eager）：

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Request Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 1688.45 | 75.81 | 616.26 | 153.16 | 1688.35 |
| 2 | 1749.10 | 73.18 | 658.11 | 155.85 | 1749.05 |
| 3 | 1770.66 | 72.29 | 674.41 | 156.60 | 1770.60 |
| 4 | 1670.98 | 76.60 | 604.76 | 152.31 | 1670.91 |
| 5 | 1741.31 | 73.51 | 673.81 | 152.49 | 1741.25 |

Optimized（Eager Prefill + CUDA Graph Decode）：

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Request Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 1545.50 | 82.82 | 1338.26 | 22.77 | 1497.62 |
| 2 | 1502.55 | 85.19 | 1305.05 | 20.99 | 1451.96 |
| 3 | 1522.02 | 84.10 | 1317.59 | 22.46 | 1474.79 |
| 4 | 1552.23 | 82.46 | 1360.80 | 20.59 | 1504.90 |
| 5 | 1541.48 | 83.04 | 1347.25 | 21.02 | 1494.38 |

#### 4.2 统计对比

| 指标 | Baseline Mean | Optimized Mean | Baseline Median | Optimized Median | Median Change |
|---|---:|---:|---:|---:|---:|
| Total Time (ms) | 1724.10 | 1532.76 | 1741.31 | 1541.48 | -11.48% |
| Throughput (tok/s) | 74.28 | 83.52 | 73.51 | 83.04 | +12.96% |
| Mean TTFT (ms) | 645.47 | 1333.79 | 658.11 | 1338.26 | +103.35% |
| Mean TPOT (ms) | 154.08 | 21.57 | 153.16 | 21.02 | -86.28% |
| Request Latency (ms) | 1724.03 | 1484.73 | 1741.25 | 1494.38 | -14.18% |

#### 4.3 结果分析

- 吞吐量中位数提高约 12.96%。
- TPOT 中位数下降约 86.28%，Decode 性能明显改善。
- 总耗时中位数下降约 11.48%。
- TTFT 中位数增加约 103.35%，说明 Decode 优先策略在高并发场景下增加了 Prefill 等待时间。
- Baseline 和 Optimized 分别执行 23 和 30 个 Engine Steps。

显存：

| 指标 | Baseline | Optimized |
|---|---:|---:|
| Peak Allocated | 4092.02 MiB | 4100.19 MiB |
| Peak Reserved | 4152 MiB | 4176 MiB |

结论：Stress 场景下，Optimized 提升了整体吞吐量并显著改善 Decode 延迟，但 Prefill 首 Token 延迟明显退化。

### 5. Staggered 场景

#### 5.1 测试结果

Staggered 用于模拟在线请求到达场景：系统先处理一部分请求，在第一个 Engine Step 后再提交剩余请求，观察新请求进入时对已有 Decode 任务的影响。

Baseline（Eager）：

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Request Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 1504.33 | 21.27 | 164.88 | 159.22 | 1279.42 |
| 2 | 955.10 | 33.50 | 120.23 | 102.47 | 837.55 |
| 3 | 940.02 | 34.04 | 124.92 | 97.73 | 809.05 |
| 4 | 1031.01 | 31.04 | 142.61 | 99.56 | 839.56 |
| 5 | 1127.45 | 28.38 | 141.38 | 114.53 | 943.06 |

Optimized（Eager Prefill + CUDA Graph Decode）：

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Request Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 879.90 | 36.37 | 466.67 | 31.72 | 688.73 |
| 2 | 752.52 | 42.52 | 435.21 | 31.29 | 654.25 |
| 3 | 927.51 | 34.50 | 494.74 | 34.96 | 739.47 |
| 4 | 824.44 | 38.81 | 446.96 | 31.82 | 669.73 |
| 5 | 890.18 | 35.95 | 498.69 | 33.24 | 731.40 |

#### 5.2 统计对比

| 指标 | Baseline Mean | Optimized Mean | Baseline Median | Optimized Median | Median Change |
|---|---:|---:|---:|---:|---:|
| Total Time (ms) | 1111.58 | 854.91 | 1031.01 | 879.90 | -14.66% |
| Throughput (tok/s) | 29.65 | 37.63 | 31.04 | 36.37 | +17.17% |
| Mean TTFT (ms) | 138.80 | 468.45 | 141.38 | 466.67 | +230.08% |
| Mean TPOT (ms) | 114.70 | 32.61 | 102.47 | 31.82 | -68.95% |
| Request Latency (ms) | 941.73 | 696.72 | 839.56 | 688.73 | -17.97% |

#### 5.3 结果分析

- 吞吐量中位数提高约 17.17%。
- TPOT 中位数下降约 68.95%。
- 总耗时中位数下降约 14.66%。
- 请求延迟中位数下降约 17.97%。
- TTFT 中位数增加约 230.08%，是三个场景中最明显的退化。

显存：

| 指标 | Baseline | Optimized |
|---|---:|---:|
| Peak Allocated | 4092.02 MiB | 4100.16 MiB |
| Peak Reserved | 4152 MiB | 4176 MiB |

结论：Staggered 场景下，Decode 性能和整体吞吐量得到改善，但新请求 Prefill 受到 Decode 优先策略影响，TTFT 明显增加。这是当前调度策略需要重点关注的局限。

### 6. All-at-once 场景

#### 6.1 测试结果

All-at-once 用于模拟多个请求同时到达、Prompt 较短且 Token Budget 充足的场景。

Baseline（Eager）：

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Request Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 649.83 | 49.24 | 381.79 | 38.29 | 649.80 |
| 2 | 682.72 | 46.87 | 371.00 | 44.53 | 682.69 |
| 3 | 667.72 | 47.92 | 372.27 | 42.20 | 667.68 |
| 4 | 677.87 | 47.21 | 372.31 | 43.65 | 677.84 |
| 5 | 675.89 | 47.34 | 373.78 | 43.15 | 675.86 |

Optimized（Eager Prefill + CUDA Graph Decode）：

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Request Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 442.71 | 72.28 | 355.01 | 12.52 | 442.67 |
| 2 | 542.82 | 58.95 | 468.10 | 10.67 | 542.79 |
| 3 | 423.38 | 75.58 | 349.51 | 10.55 | 423.35 |
| 4 | 509.01 | 62.87 | 349.42 | 22.79 | 508.97 |
| 5 | 442.51 | 72.32 | 367.91 | 10.65 | 442.47 |

#### 6.2 统计对比

| 指标 | Baseline Mean | Optimized Mean | Baseline Median | Optimized Median | Median Change |
|---|---:|---:|---:|---:|---:|
| Total Time (ms) | 670.81 | 472.09 | 675.89 | 442.71 | -34.50% |
| Throughput (tok/s) | 47.72 | 68.40 | 47.34 | 72.28 | +52.68% |
| Mean TTFT (ms) | 374.23 | 377.99 | 372.31 | 355.01 | -4.65% |
| Mean TPOT (ms) | 42.36 | 13.44 | 43.15 | 10.67 | -75.27% |
| Request Latency (ms) | 670.77 | 472.05 | 675.86 | 442.67 | -34.50% |

#### 6.3 结果分析

- 吞吐量中位数提高约 52.68%。
- TPOT 中位数下降约 75.27%。
- 总耗时中位数下降约 34.50%。
- TTFT 中位数下降约 4.65%，没有出现明显退化。
- 两个版本均执行 8 个 Engine Steps。

显存：

| 指标 | Baseline | Optimized |
|---|---:|---:|
| Peak Allocated | 3979.02 MiB | 3987.16 MiB |
| Peak Reserved | 4040 MiB | 4064 MiB |

结论：All-at-once 场景下，所有请求同时到达，Prompt 较短且 Token Budget 充足，Prefill 可以一次完成。因此该场景主要体现 Decode 执行效率的改善，不能充分验证多请求 Chunked Prefill 的调度能力。

### 7. 三种场景综合对比

以下变化均按照五轮实验的中位数计算：

| 指标 | Stress | Staggered | All-at-once |
|---|---:|---:|---:|
| Total Time | -11.48% | -14.66% | -34.50% |
| Throughput | +12.96% | +17.17% | +52.68% |
| Mean TTFT | +103.35% | +230.08% | -4.65% |
| Mean TPOT | -86.28% | -68.95% | -75.27% |
| Request Latency | -14.18% | -17.97% | -34.50% |

#### 综合观察

1. **Decode 性能：** 三种场景的 TPOT 均明显下降，说明 Optimized 执行路径在 Decode 阶段具有较明显的性能优势。
2. **吞吐量：** 三种场景的吞吐量均有提升，All-at-once 场景的提升最明显。
3. **Prefill 延迟：** Stress 和 Staggered 场景的 TTFT 明显增加，All-at-once 场景则没有明显退化。
4. **请求到达模式：** 新请求在 Decode 已经运行时进入，会受到 Decode 优先策略的影响，Staggered 场景尤其明显。
5. **显存：** CUDA Graph 带来少量额外显存开销，没有观察到显存占用下降。

### 8. 阶段性结论

本次三种场景的 5×5 Benchmark 表明：

- 多请求 Chunked Prefill 优化版本能够在不同请求负载下完成推理。
- Decode 优先调度与 CUDA Graph Decode 相结合，可以明显降低 TPOT。
- 三种场景的吞吐量均有提升。
- 在 Prefill 与 Decode 存在持续竞争的场景中，严格 Decode 优先会导致新请求 TTFT 明显退化。
- 当 Prompt 较短、Token Budget 充足时，Prefill 可以一次完成，TTFT 退化不明显。
- 当前方案更偏向优化 Decode 服务效率，而不是同时优化 Prefill 与 Decode 的响应延迟。

**总体结论：** 当前实现取得了明确的 Decode 性能和吞吐量收益，但存在 Prefill 首 Token 延迟退化的问题。该优化具有负载场景依赖性，不能将其描述为所有指标全面提升的通用优化。

注意：Baseline 与 Optimized 同时改变了 Scheduler 策略和 Decode 执行模式，因此上述结果只能证明组合优化的整体效果，不能将全部收益单独归因于 CUDA Graph。

### 9. 后续工作：CUDA Graph Ablation

为了进一步区分 Scheduler 优化与 CUDA Graph 的独立贡献，计划进行以下消融实验：

| 组别 | Scheduler | Prefill | Decode |
|---|---|---|---|
| A | Baseline | Eager | Eager |
| B | Optimized | Eager | Eager |
| C | Optimized | Eager | CUDA Graph |

对比方式：

- A vs B：评估调度策略本身的贡献。
- B vs C：评估 CUDA Graph 对 Decode 执行效率的额外贡献。
- A vs C：评估最终组合优化效果。

后续还需要关注 Decode 优先策略下的 Prefill 等待时间，探索 Decode 并发限制、Prefill 公平性和 Token Budget 分配之间的平衡。

