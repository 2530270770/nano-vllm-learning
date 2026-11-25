# Multi-Request Chunked Prefill — Benchmark Report

## 1. 项目背景

本项目基于 nano-vLLM，实现多请求 Chunked Prefill 调度优化。

原始 Scheduler 存在的限制是：Chunked Prefill 主要围绕第一个 Prefill 请求进行处理，无法充分利用多个请求之间的 Token Budget。

本次优化的主要目标：

1. 支持多个请求同时参与 Chunked Prefill。
2. Decode 优先调度，减少已有请求的生成等待。
3. 引入独立的 Decode 请求数量上限。
4. 使用 Water-Filling 算法分配 Prefill Token Budget。
5. KV Cache 分配失败时跳过当前请求，继续尝试其他请求。
6. 允许不同请求在不同调度周期参与 Prefill。

### 当前调度设计

| 配置 | 策略 |
|---|---|
| Decode 优先级 | Strict Priority |
| Decode 最大请求数 | 8 |
| Prefill Budget 分配 | Integer Water-Filling |
| KV Cache 分配失败 | Skip |
| Prefill 执行模式 | Eager |
| Decode 执行模式 | Eager |

---

## 2. Benchmark 环境与配置

### 基础环境

| 项目 | 配置 |
|---|---|
| 推理框架 | nano-vLLM |
| 模型 | Qwen3-0.6B |
| GPU | RTX 3060 Laptop 6GB |
| PyTorch | 2.6.0 |
| CUDA | 12.4 |
| 执行模式 | Eager |
| 随机种子 | 42 |
| 每种配置重复次数 | 5 |

Baseline 和 Optimized 分别使用独立代码目录：

- Baseline：`~/nano-vllm-baseline`
- Optimized：`~/nano-vllm`

运行时通过 `PYTHONPATH` 指定对应代码目录，并验证 `nanovllm.__file__`，避免错误导入。

### 测试场景

目前已经完成两个场景的五轮重复测试。

| 参数 | Staggered | All-at-Once |
|---|---:|---:|
| Requests | 4 | 4 |
| Prompt tokens/request | 512 | 128 |
| Total prompt tokens | 2048 | 512 |
| Output tokens/request | 8 | 8 |
| Max batched tokens | 512 | 16384 |
| Arrival | 分批到达 | 同时到达 |

**Staggered：**

先提交第一个请求，执行一个 Engine Step 后，再提交剩余三个请求。

主要考察：
- Decode 优先调度。
- 新请求到达时的 Prefill 等待。
- Decode 和 Prefill 之间的调度权衡。

**All-at-Once：**

四个请求同时进入 Waiting Queue。

主要考察：
- 多请求同时到达。
- Prefill 和 Decode 的整体执行表现。
- 多请求调度下的基本性能表现。

注意：当前 All-at-Once 使用 128 tokens 的 Prompt 和 16384 的 Token Budget，四个请求可以一次完成 Prefill，因此尚未真正触发多请求 Chunked Prefill 的预算分配压力。

---

## 3. Staggered Benchmark

### 3.1 Baseline 五轮结果

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 918.83 | 34.83 | 123.28 | 95.50 | 791.80 |
| 2 | 985.88 | 32.46 | 135.77 | 97.51 | 818.33 |
| 3 | 958.60 | 33.38 | 161.25 | 93.13 | 813.17 |
| 4 | 884.32 | 36.19 | 115.80 | 94.43 | 776.83 |
| 5 | 926.02 | 34.56 | 119.07 | 98.43 | 808.06 |

### 3.2 Optimized 五轮结果

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 1081.58 | 29.59 | 504.95 | 57.80 | 909.56 |
| 2 | 984.77 | 32.49 | 478.76 | 59.33 | 894.06 |
| 3 | 1114.78 | 28.71 | 520.08 | 57.80 | 924.67 |
| 4 | 1109.23 | 28.85 | 509.98 | 60.33 | 932.29 |
| 5 | 981.73 | 32.60 | 485.84 | 58.16 | 892.97 |

### 3.3 统计对比

以下以五轮测试的中位数作为主要比较依据。

| 指标 | Baseline 均值 | Optimized 均值 | Baseline 中位数 | Optimized 中位数 |
|---|---:|---:|---:|---:|
| Total Time | 934.73 | 1054.42 | 926.02 | 1081.58 |
| Throughput | 34.28 | 30.45 | 34.56 | 29.59 |
| TTFT | 131.03 | 499.92 | 123.28 | 504.95 |
| TPOT | 95.80 | 58.68 | 95.50 | 58.16 |
| Request Latency | 801.64 | 910.71 | 808.06 | 909.56 |

### 3.4 性能变化

以中位数计算：

| 指标 | 相对变化 | 结果 |
|---|---:|---|
| Total Time | +16.80% | 退化 |
| Throughput | -14.38% | 退化 |
| TTFT | +309.60% | 明显退化 |
| TPOT | -39.10% | 改善 |
| Request Latency | +12.56% | 退化 |

### 3.5 结果分析

Staggered 场景呈现出明显的 Decode 优先调度特征。

**正向结果：**

TPOT 从 95.50 ms 降低到 58.16 ms，改善约 39.10%。

说明 Decode 优先策略能够减少已有请求在生成阶段的等待时间。

**负向结果：**

TTFT 从 123.28 ms 上升到 504.95 ms，增加约 309.60%。

新请求需要等待 Decode 优先调度完成，导致首 Token 延迟明显增加。

同时：
- 整体耗时增加约 16.80%。
- 吞吐量下降约 14.38%。
- 平均请求延迟增加约 12.56%。

**阶段性结论：**

当前实现改善了 Decode 延迟，但在 Staggered 场景中牺牲了新请求的 TTFT 和整体吞吐量。

Decode 优先策略与 Prefill 延迟之间存在明显的调度权衡，需要进一步优化。

---

## 4. All-at-Once Benchmark

### 4.1 Baseline 五轮结果

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 637.69 | 50.18 | 364.67 | 39.00 | 637.66 |
| 2 | 647.41 | 49.43 | 347.26 | 42.87 | 647.38 |
| 3 | 659.72 | 48.51 | 364.01 | 42.24 | 659.69 |
| 4 | 696.25 | 45.96 | 349.96 | 49.47 | 696.22 |
| 5 | 637.75 | 50.18 | 359.88 | 39.69 | 637.72 |

### 4.2 Optimized 五轮结果

| Run | Total Time (ms) | Throughput (tok/s) | TTFT (ms) | TPOT (ms) | Latency (ms) |
|---|---:|---:|---:|---:|---:|
| 1 | 609.38 | 52.51 | 356.89 | 36.06 | 609.33 |
| 2 | 627.00 | 51.04 | 346.82 | 40.02 | 626.98 |
| 3 | 704.05 | 45.45 | 349.68 | 50.62 | 704.02 |
| 4 | 714.84 | 44.77 | 367.01 | 49.69 | 714.81 |
| 5 | 612.32 | 52.26 | 350.03 | 37.47 | 612.29 |

### 4.3 统计对比

| 指标 | Baseline 均值 | Optimized 均值 | Baseline 中位数 | Optimized 中位数 |
|---|---:|---:|---:|---:|
| Total Time | 655.76 | 653.52 | 647.41 | 627.00 |
| Throughput | 48.85 | 49.21 | 49.43 | 51.04 |
| TTFT | 357.16 | 354.09 | 359.88 | 350.03 |
| TPOT | 42.65 | 42.77 | 42.24 | 40.02 |
| Request Latency | 655.73 | 653.49 | 647.38 | 626.98 |

### 4.4 性能变化

以中位数计算：

| 指标 | 相对变化 | 结果 |
|---|---:|---|
| Total Time | -3.15% | 略有改善 |
| Throughput | +3.26% | 略有改善 |
| TTFT | -2.74% | 略有改善 |
| TPOT | -5.26% | 略有改善 |
| Request Latency | -3.15% | 略有改善 |

### 4.5 结果分析

All-at-Once 场景中，Optimized 的中位数指标均略有改善。

但是需要注意：

- Baseline Total Time 样本标准差约为 24.37 ms。
- Optimized Total Time 样本标准差约为 51.63 ms。

Optimized 的测试波动明显更大。

从均值来看：
- Total Time 仅改善约 0.34%。
- Throughput 仅提升约 0.72%。
- TPOT 均值反而略微增加。

**阶段性结论：**

当前测试没有发现 All-at-Once 场景下明显的整体性能退化，但也没有足够证据证明优化带来了稳定的性能收益。

更重要的是，本次测试的总 Prefill Token 数只有 512，而 Token Budget 为 16384。

因此四个请求可以在一个 Prefill Step 中全部完成，未真正考察多请求 Chunked Prefill 的核心调度能力。

---

## 5. 两种场景的综合结论

| 对比维度 | Staggered | All-at-Once |
|---|---|---|
| Decode TPOT | 明显改善 | 中位数略有改善 |
| Prefill TTFT | 明显退化 | 中位数略有改善 |
| 整体吞吐量 | 下降 | 基本持平 |
| 总耗时 | 增加 | 基本持平 |
| 性能稳定性 | 有一定波动 | Optimized 波动较大 |
| 是否真正测试多请求 Chunked Prefill | 未充分验证 | 未触发 |

目前可以得到以下认识：

1. Decode 优先调度确实改善了 Decode 阶段的延迟。
2. 严格 Decode 优先策略可能导致新请求 Prefill 等待时间过长。
3. All-at-Once 场景目前没有证明多请求 Chunked Prefill 的实际收益。
4. 当前优化不能简单归纳为整体推理性能提升。
5. 需要进一步验证 Water-Filling 和多请求 Chunked Prefill 的实际调度效果。

---

## 6. 下一阶段：真正的 Multi-Request Chunked Prefill 压力测试

接下来需要增加一个专门的压力测试场景。

### 测试配置

| 参数 | 数值 |
|---|---:|
| Requests | 4 |
| Prompt tokens/request | 512 |
| Total prompt tokens | 2048 |
| Output tokens/request | 8 |
| Max batched tokens | 512 |
| Arrival | All-at-Once |

### 预期验证内容

1. 四个请求同时进入 Waiting Queue。
2. Prefill Token Budget 限制为 512。
3. 多个请求参与同一轮 Chunked Prefill。
4. Water-Filling 算法分配 Prefill Token Budget。
5. 请求可以在不同 Step 中继续完成剩余 Prefill。
6. Prefill 完成的请求能够进入 Decode。
7. Decode 优先策略不会造成不可接受的 Prefill 饥饿。
8. KV Cache 分配失败时能够跳过当前请求并继续调度其他请求。

### 重点指标

- Mean TTFT
- Mean TPOT
- Request Latency
- Output Throughput
- Total Time
- Prefill Tokens Per Step
- Decode Tokens Per Step
- 每轮参与 Prefill 的请求数量
- 每个请求完成 Prefill 所需的 Step 数

### 测试原则

Baseline 和 Optimized 必须使用完全一致的：
- 模型与硬件环境
- 请求数量
- Prompt 长度
- Output 长度
- Token Budget
- 随机种子
- Eager 配置

每种配置至少重复五轮。

同时保留每轮原始 JSON，不能只记录汇总数据。

---

## 7. Benchmark 文件组织

当前测试结果保存在两个独立代码目录中。

```text
nano-vllm/
└── outputs/
    └── benchmark/
        └── chunkprefill/
            ├── optimized_staggered_run1.json
            ├── optimized_staggered_run2.json
            ├── optimized_staggered_run3.json
            ├── optimized_staggered_run4.json
            ├── optimized_staggered_run5.json
            ├── optimized_all_at_once_run1.json
            ├── optimized_all_at_once_run2.json
            ├── optimized_all_at_once_run3.json
            ├── optimized_all_at_once_run4.json
            └── optimized_all_at_once_run5.json

nano-vllm-baseline/
└── outputs/
    └── benchmark/
        └── chunkprefill/
            ├── baseline_staggered_run1.json
            ├── baseline_staggered_run2.json
            ├── baseline_staggered_run3.json
            ├── baseline_staggered_run4.json
            ├── baseline_staggered_run5.json
            ├── baseline_all_at_once_run1.json
            ├── baseline_all_at_once_run2.json
            ├── baseline_all_at_once_run3.json
            ├── baseline_all_at_once_run4.json
            └── baseline_all_at_once_run5.json
```

此外，目录中还保留了早期 Smoke Test 和 Rerun 结果。正式五轮统计以 `*_run1.json` 至 `*_run5.json` 为准，不混入早期单次测试。

---

## 8. 当前项目进度

| 工作内容 | 状态 |
|---|---|
| 多请求 Chunked Prefill 调度实现 | 已完成初步实现 |
| Decode 优先调度 | 已实现 |
| Decode 独立数量限制 | 已实现 |
| Integer Water-Filling | 已实现 |
| KV Cache 分配失败跳过策略 | 已实现 |
| Scheduler 单元测试 | 已通过 |
| ModelRunner 相关测试 | 已通过此前测试 |
| Staggered 五轮 Baseline | 已完成 |
| Staggered 五轮 Optimized | 已完成 |
| All-at-Once 五轮 Baseline | 已完成 |
| All-at-Once 五轮 Optimized | 已完成 |
| 真正的多请求 Chunked Prefill 压力测试 | 待完成 |
| 压力测试下的性能分析 | 待完成 |

## 9. 最终阶段性评价

截至目前，项目已经完成了多请求 Chunked Prefill 的主要调度逻辑实现和基础测试，并完成了两种请求到达模式下的 Baseline/Optimized 五轮 Benchmark。

现有结果证明了 Decode 优先策略对 TPOT 的改善，但也暴露了 Staggered 场景下 TTFT 明显恶化的问题。

对于多请求 Chunked Prefill 本身，目前还缺少真正具有 Token Budget 竞争的压力测试，因此暂时不能宣称 Water-Filling 已经带来了性能提升。

**下一阶段的重点不再是单纯增加普通 Benchmark 次数，而是构造真正触发多请求 Chunked Prefill 的压力场景，验证调度正确性、公平性和实际性能收益。**