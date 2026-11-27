# Multi-Request Chunked Prefill：实现进展与性能评估

> 分支：`chunkprefill`\
> 仓库：[`2530270770/nano-vllm-learning`](https://github.com/2530270770/nano-vllm-learning/tree/chunkprefill)\
> 当前评估日期：2026-11-26\
> 模型：Qwen3-0.6B\
> 测试环境：WSL Ubuntu、PyTorch 2.6.0+cu124、CUDA 12.4、RTX 3060 Laptop
> 6GB\
> 说明：本文记录当前已实现的功能、已经完成的 Benchmark
> 结果，以及后续优化计划。性能目标尚未全部达成，未验证的数据不会写成已取得的成果。

------------------------------------------------------------------------

## 1. 项目背景与目标

原始 nano-vLLM 调度器在 Step 级别采用 Prefill / Decode
互斥的执行方式：一个 Engine Step 主要执行 Prefill 或
Decode。这种方式在有新请求进入时，可能让正在 Decode 的在线请求等待
Prefill；而长 Prompt 的 Prefill 也可能占用较多计算预算。

本项目希望实现多请求 Chunked Prefill 与 Decode 混合调度：

1.  Decode 优先调度，尽量降低在线请求的逐 Token 延迟。
2.  使用 Token Budget 控制每个 Engine Step 的计算量。
3.  将 Prefill 拆分为多个 Chunk，允许多个请求在同一轮获得 Prefill 预算。
4.  同一 Engine Step 内先执行 Decode，再使用剩余预算执行 Prefill。
5.  跨 Step 保存请求的增量计算进度与 KV Cache 状态。
6.  在保证正确性的前提下，评估吞吐量、TTFT、TPOT、Step Latency
    和显存变化。

目标性能指标（**待验证目标，不是当前实测结果**）：

  指标                        目标

------------------- ------------

  模型                  Qwen3-0.6B
  并发请求数                   256
  Chunk Size                  1024
  端到端吞吐量           提升 4.6%
  P999 Step Latency     降低 36.1%
  峰值显存               降低 3.0%

------------------------------------------------------------------------

## 2. 已实现的功能

### 2.1 Scheduler

当前优化版调度器包含以下逻辑：

-   Decode 优先调度。
-   Decode 独立并发上限：`max_num_decode_seqs = 8`。
-   根据 Decode 已使用的 Token 数量计算剩余 Token Budget。
-   使用剩余预算调度 Prefill。
-   支持多个等待中的请求同时参与 Chunked Prefill。
-   使用 Integer Water-Filling 分配 Prefill Token Budget。
-   KV Cache 分配失败时跳过当前请求，继续检查后续候选请求。
-   维护请求在 waiting / running 队列之间的状态，以及跨 Step 的 Prefill
    进度。

### 2.2 ModelRunner

执行路径拆分为准备阶段和运行阶段：

-   `prepare_decode` / `run_decode`
-   `prepare_prefill` / `run_prefill`

Decode 与 Prefill
使用各自的输入准备和执行路径，并复用公共执行流程。目标是让 Decode
继续使用适合的执行路径，并让分块 Prefill 使用 Eager 执行。

> CUDA Graph 是否在目标 Batch Size 下实际命中、Prefill
> 是否在所有相关路径中确实走 Eager，仍需通过代码检查和运行时 Trace
> 进一步确认。不能仅凭代码中存在 CUDA Graph 相关逻辑，就认定所有 Decode
> Step 都命中了 Graph。

### 2.3 LLMEngine 主循环

优化版 `LLMEngine.step()`：

1.  获取 Scheduler 生成的 `ScheduleOutput`。
2.  先执行 Decode forward。
3.  对 Decode 结果进行 postprocess。
4.  再执行 Prefill forward。
5.  对 Prefill 结果进行 postprocess。
6.  返回 Decode / Prefill 的 Token 统计。

优化版 `ScheduleOutput` 包含：

-   `decode_seqs`
-   `prefill_seqs`
-   `decode_tokens`
-   `prefill_tokens`

### 2.4 已有正确性验证

此前已完成 Scheduler、ModelRunner、GPU ModelRunner 和 LLMEngine
相关测试，测试结果曾报告通过（21 passed）。

另外，16 请求压力测试中 Baseline 与 Optimized 均能完成所有请求，没有出现
OOM。

> 后续如需对外发布，应重新运行当前分支的完整测试命令，并把最新测试日志、Commit
> SHA 和环境信息归档；历史测试通过记录不等于当前代码版本的永久保证。

------------------------------------------------------------------------

## 3. Benchmark 设计与口径

Benchmark 脚本：`benchmarks/chunkprefill/benchmark.py`

当前场景配置：

  场景            请求到达方式                                  Prompt 长度   Token Budget

--------------- ------------------------------------------- ------------- --------------

  `all_at_once`   一次性提交全部请求                                    128          16384
  `staggered`     先提交 1 个请求，首个 Step 后提交其余请求             512            512
  `stress`        一次性提交全部请求                                    512            512

已经完成的对比测试均使用：

-   模型：Qwen3-0.6B
-   Eager 模式：开启
-   `max_model_len=1024`
-   `gpu_memory_utilization=0.85`
-   `temperature=1.0`
-   `ignore_eos=True`
-   正式测试重新设置相同随机种子
-   同一场景下 Baseline / Optimized 使用相同请求数、Prompt 长度、Output
    长度和 Token Budget

### 指标定义

-   **Total Time**：本轮所有请求从 Benchmark 开始到全部完成的时间。
-   **Output Throughput**：总生成 Token 数 / Total Time。
-   **TTFT**：单个请求从到达到生成首 Token 的时间。
-   **TPOT**：单个请求首 Token 之后的平均逐 Token 时间。
-   **Request Latency**：单个请求从到达到完成的时间。
-   **Step Latency**：单个 Engine Step 的执行时间。
-   **Peak Allocated / Reserved**：PyTorch 统计的峰值已分配 /
    已保留显存。

注意：当前报告中的 P999 Step Latency 尚未计算。已有 Benchmark 的 Step
时间记录可以作为后续统计输入，但必须明确样本范围、分位数算法和统计单位。

------------------------------------------------------------------------

## 4. 已完成的 Benchmark 结果

下列数据来自目前已提供的运行日志。5
轮测试以中位数作为主要对比依据，同时列出均值。吞吐量单位为
tokens/s，其余延迟单位为 ms。

### 4.1 `all_at_once`：4 请求、短 Prompt

配置：

-   请求数：4
-   Prompt：128 tokens
-   Output：8 tokens
-   Token Budget：16384
-   到达方式：一次性提交全部请求
-   Baseline / Optimized：各 5 轮

------------------------------------------------------------------------

  指标          Baseline 均值 Optimized 均值       Baseline      Optimized
                                                     中位数         中位数

------------ -------------- -------------- -------------- --------------

  Total Time           655.76         653.52         647.41         627.00

  Throughput            48.85          49.21          49.43          51.04

  Mean TTFT            357.16         354.09         359.88         350.03

  Mean TPOT             42.65          42.77          42.24          40.02

  Mean Request         655.73         653.49         647.38         626.98

  Latency                                                   
  ------------------------------------------------------------------------

中位数变化（Optimized 相对 Baseline）：

-   Total Time：约降低 3.15%。
-   Throughput：约提高 3.26%。
-   TTFT：约降低 2.74%。
-   TPOT：约降低 5.26%。

**解释与限制：** 这是短 Prompt、较大 Token Budget 的普通批量 Prefill
场景，`4 × 128 = 512`，而 Budget 为 16384。它没有形成目标中的长
Prompt、多请求分块竞争场景，因此不能用它证明多请求 Chunked Prefill
的核心收益。两版均值差异很小，且 Optimized
的波动较大，暂时不能据此认定有稳定性能提升。

### 4.2 `staggered`：4 请求、长 Prompt、错峰到达

配置：

-   请求数：4
-   Prompt：512 tokens
-   Output：8 tokens
-   Token Budget：512
-   到达方式：先提交 1 个请求，首个 Step 后提交其他 3 个请求
-   Baseline / Optimized：各 5 轮

------------------------------------------------------------------------

  指标          Baseline 均值 Optimized 均值       Baseline      Optimized
                                                     中位数         中位数

------------ -------------- -------------- -------------- --------------

  Total Time           934.73        1054.42         926.02        1081.58

  Throughput            34.28          30.45          34.56          29.59

  Mean TTFT            131.03         499.92         123.28         504.95

  Mean TPOT             95.80          58.68          95.50          58.16

  Mean Request         801.64         910.71         808.06         909.56

  Latency                                                   
  ------------------------------------------------------------------------

中位数变化（Optimized 相对 Baseline）：

-   Total Time：约增加 16.80%（变慢）。
-   Throughput：约下降 14.38%。
-   TTFT：约增加 309.60%（变差）。
-   TPOT：约降低 39.10%（改善）。
-   Request Latency：约增加 12.56%（变差）。

**解释：** Decode TPOT 有明显改善，但新到请求的 TTFT
和整体完成时间显著退化。这表明当前 Decode 优先策略可能对 Prefill
造成较大的等待代价。该场景需要进一步分析请求级 TTFT 分布和每 Step
预算分配。

### 4.3 `stress`：16 请求、长 Prompt、同时到达

配置：

-   请求数：16
-   Prompt：512 tokens
-   Output：8 tokens
-   Token Budget：512
-   到达方式：一次性提交全部请求
-   Baseline / Optimized：各 5 轮

#### Baseline 每轮结果

------------------------------------------------------------------------------

         Run Total Time   Throughput  Mean TTFT  Mean TPOT    Request      Steps
                                                              Latency 

---------- ---------- ------------ ---------- ---------- ---------- ----------

           1    1478.66        86.56     516.80     137.40    1478.61         23
    
           2    1547.46        82.72     589.16     136.89    1547.39         23
    
           3    1713.94        74.68     662.45     150.20    1713.89         23
    
           4    1562.65        81.91     583.06     139.93    1562.59         23
    
           5    1625.55        78.74     648.50     139.57    1625.49         23

------------------------------------------------------------------------------

#### Optimized 每轮结果

------------------------------------------------------------------------------

         Run Total Time   Throughput  Mean TTFT  Mean TPOT    Request      Steps
                                                              Latency 

---------- ---------- ------------ ---------- ---------- ---------- ----------

           1    1774.65        72.13    1237.34      56.96    1636.07         30
    
           2    1764.05        72.56    1238.58      56.52    1634.19         30
    
           3    1799.97        71.11    1254.08      59.56    1671.01         30
    
           4    1753.65        72.99    1224.29      57.13    1624.20         30
    
           5    1767.86        72.40    1230.30      58.02    1636.41         30

------------------------------------------------------------------------------

#### 五轮统计

------------------------------------------------------------------------

  指标          Baseline 均值 Optimized 均值       Baseline      Optimized
                                                     中位数         中位数

------------ -------------- -------------- -------------- --------------

  Total Time          1585.65        1772.04        1562.65        1767.86

  Throughput            80.92          72.24          81.91          72.40

  Mean TTFT            599.99        1236.92         589.16        1237.34

  Mean TPOT            140.80          57.64         139.57          57.13

  Mean Request        1585.59        1640.38        1562.59        1636.07

  Latency                                                   
  ------------------------------------------------------------------------

中位数变化（Optimized 相对 Baseline）：

  指标                         变化 结论

---------------------- ---------- -----------------

  Total Time                +13.13% 退化
  Throughput                -11.61% 退化
  Mean TTFT                +110.02% 退化
  Mean TPOT                 -59.07% 改善
  Mean Request Latency       +4.70% 退化
  Total Steps               23 → 30 优化版多 7 Step

#### 显存

  指标                  Baseline     Optimized

---------------- ------------- -------------

  Peak Allocated     4092.02 MiB   4092.03 MiB
  Peak Reserved      4152.00 MiB   4152.00 MiB

**解释：**

-   TPOT 中位数从 139.57 ms 降到 57.13 ms，改善约 59.07%。
-   TTFT 中位数从 589.16 ms 增至 1237.34 ms，恶化约 110.02%。
-   Throughput 中位数从 81.91 降至 72.40 tokens/s，下降约 11.61%。
-   Request Latency 中位数增加约 4.70%。
-   峰值显存基本持平，没有观察到 3% 的显存下降。
-   Optimized 每轮 30 Steps，Baseline 每轮 23 Steps。

当前结果说明：优化版在这组 16 并发压力负载下显著改善了 Decode 逐 Token
延迟，但付出了 Prefill 首 Token
延迟和整体吞吐量退化的代价。**当前不能声称端到端吞吐提升，也不能声称达到
P999 或显存目标。**

### 4.4 Stress Smoke

正式五轮之前，两版分别进行过一次 16 请求 Smoke：

  指标                     Baseline Smoke   Optimized Smoke

---------------------- ---------------- -----------------

  Total Time                   1727.62 ms        1769.25 ms
  Throughput                        74.09             72.35
  Mean TTFT                     652.57 ms        1257.36 ms
  Mean TPOT                     153.57 ms          54.73 ms
  Mean Request Latency         1727.56 ms        1640.48 ms
  Steps                                23                30
  Peak Allocated              4092.02 MiB       4092.03 MiB
  Peak Reserved               4152.00 MiB       4152.00 MiB

Smoke 仅用于确认测试配置能运行、请求能完成，不纳入正式五轮统计。

------------------------------------------------------------------------

## 5. 当前结论

截至 2026-11-25：

### 已经证明的事情

-   多请求 Chunked Prefill 的调度链路已经实现。
-   Decode 优先、剩余 Token Budget 分配和 Integer Water-Filling 已落地。
-   16 个同时到达的请求可以完成压力测试。
-   在当前 16 请求负载下，Optimized 的 TPOT 有显著改善。
-   两版测试峰值显存基本相同。

### 尚未证明的事情

-   256 并发下的稳定性和性能。
-   Chunk Size=1024 下的性能表现。
-   CUDA Graph 在目标 Decode Batch Size 下的实际命中情况。
-   Prefill Eager 路径的实际执行情况。
-   P999 Step Latency 降低 36.1%。
-   端到端吞吐量提升 4.6%。
-   峰值显存降低 3.0%。

因此当前最准确的项目定位是：

> 已完成多请求 Chunked Prefill
> 混合调度的核心功能实现和初步正确性验证；初步压力测试显示 Decode TPOT
> 明显改善，但 TTFT
> 和整体吞吐量出现退化。项目进入调度策略分析、执行路径验证和高并发性能优化阶段。

------------------------------------------------------------------------

## 6. 后续阶段计划

### 阶段 1：Step Trace 与请求级延迟分析（当前优先级最高）

**目标：定位 TTFT 退化和 Step 数增加的原因，不急于改调度代码。**

工作项：

-   [ ] 汇总 Baseline / Optimized 每轮 Step Latency。
-   [ ] 计算 Step Latency 的 P50 / P90 / P99 / P999。
-   [ ] 统计 Decode-only、Prefill-only、Decode+Prefill 混合 Step
    数量和占比。
-   [ ] 分析每个 Step 的 Decode Token 与 Prefill Token 分配。
-   [ ] 输出 16 个请求的 TTFT 排序、分布、最大值和最小值。
-   [ ] 区分 Prefill 排队等待时间与 Prefill 实际执行时间。
-   [ ] 检查是否存在长期得不到 Prefill 预算的请求。

**阶段产物：** Baseline / Optimized Step Trace 分析报告，明确 TTFT
退化的主要来源。

### 阶段 2：Scheduler 策略优化

**目标：在保留 Decode 优先设计的前提下，降低 Prefill 等待成本。**

工作项：

-   [ ] 分析 Decode 独立上限 `max_num_decode_seqs=8` 对剩余 Prefill
    预算的影响。
-   [ ] 检查 Decode 优先后每轮剩余 Token Budget 的分布。
-   [ ] 分析 Integer Water-Filling 对不同剩余 Prompt
    长度请求的分配效果。
-   [ ] 评估请求等待时间、剩余 Prefill Token 数和 KV Cache 容量等因素。
-   [ ] 检查 KV Cache 分配失败跳过策略是否造成请求饥饿或不公平。
-   [ ] 每次只调整一个策略变量，并用相同 Benchmark 做回归对比。

**阶段产物：** Scheduler 策略说明、参数对比实验和回归测试结果。

### 阶段 3：ModelRunner 与 CUDA Graph 执行路径验证

**目标：确认设计中的 Decode Graph / Prefill Eager 路径真实生效。**

工作项：

-   [ ] 检查 CUDA Graph 捕获的 Batch Size 范围。
-   [ ] 记录实际 Decode Batch Size 分布。
-   [ ] 统计 Decode Graph 命中次数、未命中次数和回退路径。
-   [ ] 确认混合调度中的 Prefill 使用 Eager 执行。
-   [ ] 分析 `prepare_decode` / `run_decode` 和 `prepare_prefill` /
    `run_prefill` 的 CPU 开销。
-   [ ] 检查两条路径是否存在重复的 Tensor 准备、内存分配或同步。
-   [ ] 使用 profiler / Trace 对关键路径进行验证。

**阶段产物：** ModelRunner 执行路径图、Graph
命中统计和关键路径性能分析。

### 阶段 4：Chunk Size 与调度参数 Sweep

**目标：找到适合当前模型和负载的 Chunk Size / Budget。**

建议测试：

  参数                          候选值

----------------------------- ----------------------------------

  Chunk Size / Prefill Budget   128、256、512、1024、2048
  Decode 并发上限               4、8、16（需结合配置和硬件验证）
  并发请求数                    16、32、64（逐级增加）

工作项：

-   [ ] 明确区分全局 Token Budget 与单请求 Chunk Size。
-   [ ] 对每组配置执行相同的 Baseline / Optimized 对比。
-   [ ] 记录吞吐量、TTFT、TPOT、Step Latency 和显存。
-   [ ] 观察 Chunk Size 增大后 Prefill 执行效率与 Decode
    等待之间的变化。
-   [ ] 避免只根据单次结果选择参数，至少进行多轮重复。

**阶段产物：** 参数 Sweep CSV / JSON、性能曲线和推荐配置。

### 阶段 5：高并发与目标规模验证

**目标：逐步接近 256 并发目标，并在适合的硬件上完成最终实验。**

建议递进：

1.  16 并发：已有数据，先完成 Trace 分析。
2.  32 并发：验证调度稳定性和显存增长。
3.  64 并发：观察 KV Cache 和 Prefill 等待。
4.  128 并发：评估调度 CPU 开销与 Decode Batch Size。
5.  256 并发：在显存和算力足够的硬件上验证目标配置。

工作项：

-   [ ] 每个并发档位先跑 Smoke，再跑正式重复实验。
-   [ ] 记录模型、GPU、驱动、CUDA、PyTorch、Commit SHA。
-   [ ] 固定请求到达模式、Prompt / Output 长度分布和随机种子。
-   [ ] 记录 KV Cache 使用率、抢占次数、请求完成率和 OOM 情况。
-   [ ] 在 256 并发、Chunk Size=1024 条件下进行正式对比。

**硬件限制：** 当前 RTX 3060 Laptop 6GB 已经在 16 请求测试中显示约 4 GiB
的 PyTorch 峰值分配。不能假定它能够承载 256
并发目标。可以先做低并发和缩短上下文的开发验证，但不能把缩小配置的结果当作
256 并发目标结果。

**阶段产物：** 高并发压力测试报告和 256 并发正式 Benchmark 结果。

### 阶段 6：最终性能验收与项目总结

**目标：验证最初设定的三个性能指标。**

验收项：

-   [ ] 端到端吞吐量相对 Baseline 提升至少 4.6%。
-   [ ] P999 Step Latency 相对 Baseline 降低至少 36.1%。
-   [ ] 峰值显存相对 Baseline 降低至少 3.0%。
-   [ ] 256 并发、Chunk Size=1024 下正确性与稳定性通过。
-   [ ] 保存原始 JSON、汇总 CSV、Trace、运行环境和 Commit SHA。
-   [ ] README 中只填写已经实测并可复现的结果。

如果最终实验未达到某个指标，应如实报告实际结果和适用负载，不应把目标值写成已达成结果。

------------------------------------------------------------------------

## 7. 推荐的下一步执行顺序

当前建议严格按下面顺序推进：

1.  **先做阶段 1：** 分析已经保存的 `optimized_stress_run1.json` 至
    `optimized_stress_run5.json`，并与 Baseline 对照。
2.  **再做阶段 2：** 根据 Trace 证据调整 Scheduler，不要同时改多个策略。
3.  **然后做阶段 3：** 确认 CUDA Graph 和 Eager 执行路径。
4.  **再做阶段 4：** 做 Chunk Size / Decode 上限参数 Sweep。
5.  **最后做阶段 5、6：** 逐级提高并发，完成目标规模验收。

当前不建议直接把并发数从 16 跳到 256，也不建议在没有 Trace
证据的情况下继续盲目修改 Water-Filling。

------------------------------------------------------------------------

## 8. Benchmark 文件与结果归档

当前已使用的结果文件：

``` text
outputs/benchmark/chunkprefill/
├── baseline_stress_smoke.json
├── optimized_stress_smoke.json
├── baseline_stress_run1.json
├── baseline_stress_run2.json
├── baseline_stress_run3.json
├── baseline_stress_run4.json
├── baseline_stress_run5.json
├── optimized_stress_run1.json
├── optimized_stress_run2.json
├── optimized_stress_run3.json
├── optimized_stress_run4.json
└── optimized_stress_run5.json
```

此前 `all_at_once` 和 `staggered` 的测试结果也已完成五轮统计。请确保对应
JSON 文件在本地归档；如果文件名与上述目录不同，以实际生成文件为准。

建议后续统一保留：

-   原始 Benchmark JSON。
-   汇总统计 CSV。
-   Step Trace。
-   运行命令和配置。
-   Git Commit SHA。
-   GPU / CUDA / PyTorch 环境信息。

这样才能保证项目结果可复现、可解释，也方便在面试中说明优化前后的差异。

------------------------------------------------------------------------

## 9. 简历表述建议（当前阶段）

在 256 并发目标数据尚未取得前，建议使用下面这种保守表述：

> **多请求 Chunked Prefill 混合调度优化：** 针对原框架 Step 级 Prefill /
> Decode 调度互斥问题，重构 Scheduler、ModelRunner 与推理主循环，实现
> Decode 优先、Token Budget 驱动的多请求 Chunked Prefill 调度，并通过
> Integer Water-Filling 分配 Prefill 预算，支持跨 Step 增量 Prefill 与
> KV Cache 状态维护。基于 Qwen3-0.6B 完成 16
> 请求压力测试；当前实验观察到 Decode TPOT 中位数降低约 59.07%，但 TTFT
> 中位数增加约 110.02%，整体吞吐量下降约
> 11.61%，正在继续优化调度策略并验证高并发性能。

等 256 并发、Chunk Size=1024
的正式实验确实达到目标后，再将目标指标替换为实测成果。

------------------------------------------------------------------------

## 10. Decode 并发上限 Sweep（2026-11-26）

本节记录在多请求错峰到达负载下，对 `max_num_decode_seqs` 进行的 5 轮重复实验。该实验用于评估 Decode 并发上限对整体吞吐、请求延迟以及不同 Prompt 长度请求的影响。

### 10.1 实验配置

| 参数 | 配置 |
|---|---:|
| 模型 | Qwen3-0.6B |
| 请求数 | 32 |
| Prompt 长度分布 | 128 × 11、512 × 11、1024 × 10 |
| Output Tokens | 8 |
| `max_num_batched_tokens` | 512 |
| `max_num_prefill_seqs` | 8 |
| `execution_mode` | `cudagraph` |
| 请求到达方式 | staggered |
| 首批请求数 | 4 |
| 后续每批请求数 | 4 |
| 最小到达间隔 | 2 Steps |
| 随机种子 | 42 |
| Decode 并发上限 | 4、8、16 |
| 重复次数 | 每档 5 轮 |

三档配置的请求总数和 Prompt 长度分布完全一致，每档共统计 160 个请求。所有配置均完成请求，没有因 KV Cache 分配失败产生的记录，也没有 Decode Preemption。

### 10.2 整体性能结果

下表为每档 5 轮的均值。吞吐量单位为 tokens/s，延迟单位为 ms。

| 指标 | Decode cap=4 | Decode cap=8 | Decode cap=16 |
|---|---:|---:|---:|
| Total Time | 3103.73 | 3049.40 | 3008.92 |
| Output Throughput | 82.54 | 84.14 | 85.17 |
| Mean TTFT | 964.65 | 987.78 | 994.64 |
| Mean TPOT | 86.07 | 62.17 | 62.16 |
| Mean Request Latency | 1567.12 | 1422.98 | 1429.74 |
| Engine Steps | 60 | 44 | 42 |
| Total Time 标准差 | 91.46 | 161.03 | 108.40 |
| Throughput 标准差 | 2.38 | 4.37 | 3.04 |

主要观察：

- cap4 的 Decode 并发偏低：平均 TPOT 为 86.07 ms，且需要 60 个 Engine Steps。
- cap8 相比 cap4，平均 TPOT 从 86.07 ms 降至 62.17 ms，平均请求延迟从 1567.12 ms 降至 1422.98 ms，Engine Steps 从 60 降至 44。
- cap16 相比 cap8，平均吞吐从 84.14 提升至 85.17 tokens/s，约提升 1.22%；Total Time 约降低 1.33%。但平均 TPOT 几乎不变，Mean TTFT 和 Mean Request Latency 略有上升。
- cap16 的平均吞吐最高，但相对 cap8 的提升幅度较小，且当前 5 轮实验不足以证明这 1.22% 的差异具有统计显著性。

### 10.3 按 Prompt 长度拆分的请求指标

以下为合并 5 轮后的请求级统计均值，单位为 ms。Pooled P50/P95 是将该配置 5 轮的请求样本合并后计算；它们不是“5 个轮均值”的分位数。

| Prompt | Decode cap | Mean TTFT | Mean Queue Wait | Mean Request Latency | Mean TPOT |
|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 302.87 | 57.86 | 988.83 | 97.99 |
| 128 | 8 | 314.10 | 59.99 | 813.84 | 71.39 |
| 128 | 16 | 312.89 | 60.41 | 811.52 | 71.23 |
| 512 | 4 | 964.32 | 57.18 | 1638.67 | 96.34 |
| 512 | 8 | 979.58 | 58.95 | 1478.32 | 71.25 |
| 512 | 16 | 987.15 | 60.17 | 1491.34 | 72.03 |
| 1024 | 4 | 1692.98 | 56.42 | 2124.54 | 61.65 |
| 1024 | 8 | 1737.84 | 58.55 | 2032.16 | 42.05 |
| 1024 | 16 | 1752.81 | 58.94 | 2042.01 | 41.32 |

观察与解释：

- cap4 提升到 cap8 后，三种 Prompt 长度的 TPOT 均明显下降；请求总延迟也均下降，其中 Prompt=128 的下降幅度最大。
- cap8 提升到 cap16 后，各 Prompt 长度的请求总延迟变化很小：Prompt=128 略有下降，Prompt=512 和 1024 略有上升。
- 三种 Prompt 长度的 TTFT 均随 Decode cap 从 4 增至 8、16 而略有上升。
- 不同 Prompt 长度的平均首次 Prefill 队列等待约为 56–60 ms，三档 Decode cap 之间差异较小。该指标只表示首次被调度 Prefill 之前的队列等待，不包含后续 Chunk Prefill 的全部执行时间。
- Prompt=128、512、1024 的 TTFT 差异明显，但不能把这种差异简单归因于排队；Prompt 计算量本身也不同。需要结合每个 Step 的实际执行时间和请求到达时间进一步分析。

请求级 Pooled 分位数示例（单位 ms）：

| Prompt | cap | TTFT P50 / P95 | Queue Wait P50 / P95 | Request Latency P50 / P95 |
|---:|---:|---:|---:|---:|
| 128 | 4 | 343.63 / 414.67 | 70.91 / 139.88 | 1045.34 / 1296.87 |
| 128 | 8 | 343.28 / 444.99 | 71.19 / 143.76 | 827.47 / 1005.56 |
| 128 | 16 | 349.92 / 442.66 | 70.90 / 145.68 | 839.71 / 942.94 |
| 512 | 4 | 1115.68 / 1217.18 | 67.39 / 139.01 | 1717.41 / 2027.75 |
| 512 | 8 | 1100.36 / 1249.11 | 69.59 / 143.99 | 1592.98 / 1752.95 |
| 512 | 16 | 1119.53 / 1271.57 | 68.38 / 152.50 | 1640.31 / 1763.03 |
| 1024 | 4 | 1728.75 / 1934.33 | 69.53 / 139.89 | 2106.34 / 2365.61 |
| 1024 | 8 | 1769.66 / 1990.67 | 70.09 / 143.85 | 2024.40 / 2367.64 |
| 1024 | 16 | 1805.46 / 2006.07 | 70.02 / 146.03 | 2014.95 / 2370.13 |

### 10.4 ModelRunner 与 Scheduler 运行时诊断

5 轮汇总的平均诊断指标：

| 指标 | cap4 | cap8 | cap16 |
|---|---:|---:|---:|
| Decode Graph Hit Rate | 100% | 100% | 100% |
| Decode Graph Calls（平均每轮） | 59 | 43 | 41 |
| Decode Eager Calls（平均每轮） | 0 | 0 | 0 |
| Aggregate Token Budget Utilization | 56.98% | 77.70% | 81.40% |
| 平均 Scheduler Decode Count | 3.73 | 5.09 | 5.33 |
| 平均 Prefill Candidates | 4.17 | 5.70 | 5.98 |
| 平均 Prefill Wait Steps | 0.81 | 0.81 | 0.81 |
| Prefill Wait Steps P95 | 2 | 2 | 2 |
| 平均 Prefill Queue Wait | 57.17 ms | 59.18 ms | 59.87 ms |
| KV Allocation Failures | 0 | 0 | 0 |
| Decode Preemptions | 0 | 0 | 0 |

口径说明：

- Graph Hit Rate 是 ModelRunner 调用级统计；本组测试中 Graph Hit Rate 为 100%、Decode Eager Calls 为 0，表示记录到的 Decode 调用均命中 Graph 路径，不代表 GPU kernel 已达到最优效率。
- Aggregate Token Budget Utilization 按所有 Step 的 `used_tokens / token_budget` 汇总计算，包含 Decode 和 Prefill Token，不是 Prefill 专属预算利用率。
- Scheduler Decode Count 是每个 Step 调度的 Decode 请求数的平均值。
- Prefill Candidates 是参与候选分配的等待请求数，不等于实际获得 Prefill Token 的请求数。
- KV Allocation Failures 和 Decode Preemptions 在本组实验中均为 0。

### 10.5 当前结论与配置选择

当前建议将 `max_num_decode_seqs=8` 保留为默认值：

- cap4 的 Decode 并发不足，TPOT 和请求完成延迟明显较高。
- cap8 在 TPOT、请求完成延迟、TTFT 和整体吞吐之间取得相对均衡的结果。
- cap16 可以作为吞吐优先的候选配置，但相对 cap8 的平均吞吐提升约 1.22%，目前没有足够证据证明该提升稳定且具有统计显著性。

这组实验没有证明 cap16 导致 Prefill 饥饿：三档配置的 Prefill Wait Steps 均值和 P95 相同，KV 分配失败也均为 0。但首次队列等待和候选数不能完整代表所有 Chunk 的公平性，因此仍需单独做请求级 Prefill 公平性实验。

### 10.6 后续实验

请求级 Prefill Trace 已加入 Benchmark（见第 11 节），尚未在真实 GPU 环境执行验证。下一步优先使用新版本重新跑 cap8 / cap16 配对实验，检查：

- [ ] 每个请求每个 Step 实际获得的 Prefill Token 数。
- [ ] 每个请求首次 Prefill、每次 Chunk Prefill 和 Prefill 完成的 Step。
- [ ] 每个请求的累计等待 Step、最大连续等待 Step 和 Prefill 完成时间。
- [ ] 不同 Prompt 长度请求的进度与完成时间，是否存在饥饿。
- [ ] Scheduler CPU 时间与 Decode / Prefill 模型 Forward GPU 时间的关系。

本节已有性能结果来自旧版 Trace；新增字段必须重新运行后才会出现在新 JSON 中。已有结果不应被描述为已经完成请求级公平性验证。

本节结果只适用于上述 Qwen3-0.6B、32 请求、指定 Prompt 分布和当前 GPU 环境；不能外推为 256 并发或其他模型上的性能结论。

------------------------------------------------------------------------

## 11. 请求级 Prefill 公平性 Trace（已完成 GPU 验证）

为定位多请求 Chunked Prefill 中的等待与进度差异，`benchmarks/chunkprefill/benchmark.py` 新增请求级 Trace。该改动只观测 Scheduler 输出和 ModelRunner Forward，不改变调度决策。

### 11.1 每个请求新增字段

| 字段 | 含义 |
|---|---|
| `prefill_chunk_count` | 请求实际获得 Prefill 分配的 Chunk 数 |
| `prefill_chunks` | 每个 Chunk 的 Step、起始 Cached Token、分配 Token、执行后 Cached Token、是否完成 Prompt |
| `prefill_completion_step` | `num_cached_tokens` 首次达到 `num_prompt_tokens` 的 Engine Step |
| `prefill_completion_time_ms` | Prefill 完成所在 Step 结束时的相对时间 |
| `prefill_completion_latency_ms` | 从请求到达到 Prefill 完成的时间 |
| `prefill_total_wait_steps` | Prompt 尚未完成时，没有获得 Prefill 分配的 Step 总数 |
| `prefill_max_consecutive_wait_steps` | 连续没有获得 Prefill 分配的最大 Step 数 |
| `prefill_total_allocated_tokens` | 所有 Prefill Chunk 分配 Token 数之和 |

### 11.2 每个 Engine Step 新增字段

- `prefill_chunks`：本 Step 实际调度的 Chunk 明细，并在 Forward 完成后补充执行结果。
- `scheduler_schedule_time_ms`：Scheduler.schedule() 的 CPU wall-clock 时间。
- `decode_forward_gpu_time_ms`：Decode 模型 Forward / Logits 计算的 CUDA Event 时间。
- `prefill_forward_gpu_time_ms`：Prefill 模型 Forward / Logits 计算的 CUDA Event 时间。

GPU 时间是整个批次的阶段级时间，不是单请求耗时；它不包含输入 Tensor 准备与采样，也不能简单与 Step 总耗时相加。CUDA Event 不在每个阶段后强制同步，以避免人为增加同步开销。

### 11.3 汇总字段与统计口径

`summary.prefill_fairness` 和 `summary.prefill_fairness_by_prompt_length` 汇总累计等待、最大连续等待、Chunk 数和 Prefill 完成延迟。

累计等待定义：每个 Engine Step 结束后，如果请求尚未完成 Prefill，且本 Step 没有获得任何 Prefill 分配，则累计等待加一；获得任意 Prefill 分配的 Step 不计等待。Prefill 完成后立即停止累计，后续 Decode Step 不计入等待。完成状态由首次达到 `num_cached_tokens >= num_prompt_tokens` 的 Step 显式记录，避免后续 Decode 阶段 Cached Token 状态变化污染等待统计。它同时覆盖首次排队等待和 Chunk 之间的等待，不等同于原有的 `prefill_wait_steps`（首次 Prefill 选择前等待的 Step 数）。

Prefill 完成定义：执行完某个 Engine Step 后，`num_cached_tokens >= num_prompt_tokens`。完成时间记录在该 Step 的 CUDA 同步之后，因此是 Step 结束时刻，不是单独的 Prefill Kernel 完成时刻。

### 11.4 验证状态

- 代码已推送到 `chunkprefill` 分支。
- 已在 RTX 3060 Laptop 上通过 32 请求 Smoke 和 cap8 / cap16 各 5 轮正式实验；对应数据见第 12 节。
- 新 JSON 生成后，先检查每个请求的 `prefill_chunks[0].cached_tokens_before + prefill_total_allocated_tokens == prompt_tokens`（考虑可能命中的 Prefix Cache），且各 Chunk 的 Cached Token 区间连续、不重叠；再分析 cap8 / cap16 的公平性。
- 新增的 GPU 阶段计时也需要在真实运行中确认事件记录完整且非负。

建议先执行 1 次 32 请求 Smoke，再执行 cap8 / cap16 各 5 轮正式实验。不要将旧 JSON 与新增字段混合统计。


------------------------------------------------------------------------

## 12. Decode cap 8 vs 16：请求级公平性正式复测（2026-11-27）

本节是在请求级 Prefill Trace 修复并完成 GPU 验证后，对 Decode 并发上限 8 和 16 进行的 5 组配对实验。它补充验证第 10 节中的 cap8 / cap16 结论，并首次提供请求级累计等待与最大连续等待数据。第 10 节 cap4 / cap8 / cap16 的历史结果仍保留；两次实验的运行批次与记录口径不同，不应把数值混在一起计算。

### 12.1 实验配置与完整性

| 参数 | 配置 |
|---|---:|
| 模型 | Qwen3-0.6B |
| 请求数 | 32 |
| Prompt 长度分布 | 128 × 11、512 × 11、1024 × 10 |
| Output Tokens | 8 |
| `max_num_batched_tokens` | 512 |
| `max_num_prefill_seqs` | 8 |
| Decode cap | 8、16 |
| Execution Mode | CUDA Graph |
| 请求到达方式 | staggered：首批 4 个，后续每批 4 个，每隔 2 Steps |
| 随机种子 | 42、43、44、45、46 |
| 重复次数 | 每档 5 轮，共 10 轮 |

两档配置使用相同的 5 个 Seed，且每个 Seed 都有 cap8 / cap16 对应结果。运行顺序交替，以降低固定运行顺序对结果的影响。

完整性检查：
- 10 个 JSON 均成功读取，配置中的 Decode cap、Seed、Scenario、Token Budget 与预期一致。
- 每轮均包含 32 个请求，32 个请求全部完成 Prefill。
- 每轮均生成 256 个 Output Tokens；总 Prompt Prefill Tokens 为 17,280。
- `steps` 与 `scheduler_steps` 数组长度均与 `summary.total_steps` 一致。
- 10 轮均无 KV Allocation Failure、无 Decode Preemption。
- 所有轮次 Decode Graph Hit Rate 均为 100%，Decode Eager Calls 均为 0。

原始结果位于：

```text
outputs/benchmark/chunkprefill/decode_cap_sweep/
├── cap8_run1_seed42.json
├── cap8_run2_seed43.json
├── cap8_run3_seed44.json
├── cap8_run4_seed45.json
├── cap8_run5_seed46.json
├── cap16_run1_seed42.json
├── cap16_run2_seed43.json
├── cap16_run3_seed44.json
├── cap16_run4_seed45.json
└── cap16_run5_seed46.json
```

### 12.2 整体性能结果

下表为每档 5 轮的均值；时间单位 ms，吞吐量单位 tokens/s。

| 指标 | cap8 | cap16 | cap16 相对 cap8 |
|---|---:|---:|---:|
| Total Time | 2903.02 | 2909.44 | +0.22% |
| Output Throughput | 88.19 | 88.00 | -0.22% |
| Mean TTFT | 962.10 | 966.70 | +0.48% |
| Mean TPOT | 60.48 | 60.78 | +0.49% |
| Mean Request Latency | 1385.48 | 1392.16 | +0.48% |
| Engine Steps | 44 | 42 | -4.55% |

5 轮样本标准差：

| 指标 | cap8 SD | cap16 SD |
|---|---:|---:|
| Total Time | 20.79 ms | 25.69 ms |
| Output Throughput | 0.63 tok/s | 0.77 tok/s |
| Mean TTFT | 5.04 ms | 4.21 ms |
| Mean TPOT | 0.33 ms | 0.78 ms |
| Mean Request Latency | 6.74 ms | 9.63 ms |

配对 Seed 的结果并非每轮都一致：cap16 在 5 组中有 3 组总耗时更短、2 组更长；吞吐量也是 3 组更高、2 组更低。5 组配对均值显示 cap16 吞吐略低约 0.22%，差异很小，不应据此宣称 cap8 在统计意义上显著更快。

主要观察：
- cap16 每轮固定减少 2 个 Engine Steps（44 → 42），但没有减少端到端总耗时。
- cap16 的 Aggregate Token Budget Utilization 从 77.70% 增至 81.40%，提高约 3.70 个百分点；更高的预算利用率没有带来吞吐提升。
- cap8 在本次实验的吞吐、TTFT、TPOT 和请求延迟均值上略优；优势幅度小，属于当前负载和设备下的实测观察。
- 这组结果不支持把 Decode cap 从 8 提高到 16 作为性能优化。

### 12.3 Prefill 公平性结果

以下为每轮汇总指标的均值。等待单位为 Engine Steps。

| 指标 | cap8 | cap16 |
|---|---:|---:|
| Mean Prefill First-Selection Wait | 0.8125 | 0.8125 |
| P95 Prefill First-Selection Wait | 2 | 2 |
| Max Prefill First-Selection Wait | 2 | 2 |
| Mean Total Prefill Wait | 5.21875 | 5.21875 |
| P95 Total Prefill Wait | 11 | 11 |
| Max Total Prefill Wait | 11 | 11 |
| Mean Max Consecutive Wait | 1.375 | 1.375 |
| Max Consecutive Wait | 2 | 2 |
| Mean Chunk Count / Request | 7.84375 | 7.84375 |
| Mean Prefill Completion Latency | 962.10 ms | 966.70 ms |
| P95 Prefill Completion Latency | 1918.45 ms | 1926.10 ms |

按 Prompt 长度分组的累计等待与 Chunk 数在两档 Decode cap 下也一致：
- Prompt=128：平均累计等待 1.455 Steps，平均 Chunk 数 2.091。
- Prompt=512：平均累计等待 5.636 Steps，平均 Chunk 数 7.636。
- Prompt=1024：平均累计等待 8.900 Steps，平均 Chunk 数 14.400。

解释：
- cap16 没有导致本次测试中 Prefill 等待增加；但它也没有改善 Prefill 公平性。
- 长 Prompt 的累计等待和 Chunk 数更高，主要反映其需要更多次增量 Prefill；不能只凭累计等待绝对值判断调度不公平。
- 所有请求最大连续等待均不超过 2 个 Engine Steps，说明本负载下没有观察到长时间连续饥饿。
- 这些指标是当前 32 请求、给定 Prompt 分布和错峰到达模式下的结果，不代表更高并发或其他负载下也不会饥饿。

### 12.4 ModelRunner / Scheduler 诊断

| 指标 | cap8 | cap16 |
|---|---:|---:|
| Decode Calls / Run | 43 | 41 |
| Decode Graph Hit Rate | 100% | 100% |
| Decode Eager Calls | 0 | 0 |
| Prefill Calls / Run | 35 | 35 |
| Aggregate Token Budget Utilization | 77.70% | 81.40% |
| KV Allocation Failures | 0 | 0 |
| Decode Preemptions | 0 | 0 |
| Peak Allocated Delta | 15.01 MiB | 15.01 MiB |
| Peak Reserved Delta | 6 MiB | 6 MiB |

Graph Hit Rate 表示 Decode ModelRunner 调用路径命中 CUDA Graph 的比例，不代表 GPU Kernel 已达到最优效率。Token Budget Utilization 是 Decode 与 Prefill Token 合并统计的整体预算利用率，不是 Prefill 专属利用率。

### 12.5 配置结论

**当前保留 `max_num_decode_seqs=8` 作为默认配置。**

理由：
1. cap16 虽然将 Engine Steps 从 44 降到 42，并提高预算利用率，但没有带来吞吐提升。
2. cap8 的吞吐、TTFT、TPOT 和请求延迟均值略优于 cap16。
3. 两档配置的 Prefill 累计等待、连续等待、Chunk 数完全一致；cap16 没有表现出公平性优势。
4. 两者差异总体较小，当前 5 组配对实验不支持“cap8 显著优于 cap16”的统计结论；选择 cap8 是基于当前设备和负载下更均衡的实测结果，而不是普适最优值。

因此不建议为了提高 Token Budget Utilization 或减少两个 Engine Steps 而把默认 Decode cap 提高到 16。后续如果负载特征发生变化，应重新进行针对性 Sweep。

### 12.6 本阶段状态与下一步

- [x] 修复并验证请求级 Prefill 等待统计。
- [x] 完成 cap8 / cap16 各 5 轮 GPU 配对实验。
- [x] 验证请求 Prefill Chunk 连续性、完成状态与输出 Token 数。
- [x] 验证 CUDA Graph 调用路径、KV Allocation Failure 与 Decode Preemption 统计。
- [x] 根据实测结果保留 Decode cap=8 默认值。
- [x] 将本轮 10 个 JSON 的汇总统计脚本化，并生成 CSV / Markdown 报告（见第 12.7 节）。
- [ ] 在更高请求数和不同 Prompt 分布下验证调度策略，不把当前 32 请求结果外推到 256 并发。

本节结论仅适用于上述模型、设备、32 请求工作负载和参数配置。


### 12.7 自动化结果汇总（CPU-only）

新增 `benchmarks/chunkprefill/summarize_results.py`，用于读取本目录的原始 JSON，执行完整性校验并自动生成 CSV / Markdown 报告。脚本只使用 Python 标准库，不加载模型、不初始化 CUDA，也不会修改原始 Benchmark JSON。

执行命令：

```bash
# 在仓库根目录执行；无需 GPU
python -m py_compile benchmarks/chunkprefill/summarize_results.py
python -m unittest discover -s tests -p 'test_chunkprefill_report.py' -v

python benchmarks/chunkprefill/summarize_results.py
```

默认读取：

```text
outputs/benchmark/chunkprefill/decode_cap_sweep/
```

默认要求 cap8 / cap16 各有 5 轮结果，并检查两档 Seed 配对一致。校验包括：
- 文件名中的 Decode cap / Seed 与 JSON 配置一致。
- 请求数、Prefill 完成数、Output Token 总数和 Step 数组一致。
- 每个请求的 Prefill Chunk Cached Token 游标连续，且最终达到 Prompt 长度。
- ModelRunner Decode Graph 调用数不超过 Decode 总调用数。

生成文件位于 `outputs/benchmark/chunkprefill/decode_cap_sweep/analysis/`：

| 文件 | 内容 |
|---|---|
| `run_summary.csv` | 每轮实验配置、性能和运行时诊断 |
| `aggregate_summary.csv` | 每档配置跨轮均值、样本标准差、中位数、最值 |
| `paired_comparison.csv` | 按 Seed 配对的 cap8 / cap16 指标差值与百分比 |
| `prompt_fairness.csv` | 合并请求样本后，按 Prompt 长度统计等待、Chunk 和完成延迟 |
| `report.md` | 可直接放入项目文档的汇总报告 |

统计口径：
- Run-level 均值和标准差以每轮 JSON 的 summary 指标为样本，标准差使用样本标准差。
- Prompt fairness 是将该 cap 的所有轮次请求合并后统计，P50 / P95 使用 nearest-rank。
- Aggregate Token Budget Utilization 使用所有 Step 的 `sum(used_tokens) / sum(token_budget)`，不是各 Step 利用率的简单平均。
- 配对差值定义为高 cap 减低 cap；延迟指标正差值通常代表变差，吞吐指标正差值代表变好。
- 该脚本只负责离线结果分析，不能替代 GPU 正确性或性能测试。

当前 10 轮实验的 CSV 和 Markdown 报告已随仓库提交，位于上述 analysis 目录。后续新增实验完成后，重新运行脚本即可覆盖生成新的分析文件；请先归档旧报告或使用不同的 `--output-dir` 保留历史版本。

------------------------------------------------------------------------

## 13. 更高并发容量验证（新增测试入口，待 GPU 实测）

当前 32 请求结果不能外推到更高并发。新增
`benchmarks/chunkprefill/run_capacity_sweep.py`，用于按请求数递增运行
Optimized GPU Benchmark。默认档位为 32、40、48，每档先执行 1 次 Smoke；
每次都使用 staggered 到达、Prompt 长度循环 128/512/1024、Output=8、
Decode cap=8、Prefill cap=8、CUDA Graph。

### 13.1 先做 CPU-only 检查

```bash
python -m py_compile benchmarks/chunkprefill/run_capacity_sweep.py
python -m unittest discover -s tests -p 'test_chunkprefill_capacity_sweep.py' -v
python benchmarks/chunkprefill/run_capacity_sweep.py --dry-run
```

Dry run 只打印命令并写入测试清单，不会加载模型或调用 CUDA。

### 13.2 RTX 3060 Laptop 递进 Smoke

```bash
python benchmarks/chunkprefill/run_capacity_sweep.py
```

脚本默认依次运行 32、40、48 请求，每档 1 次。每轮原始 JSON 写入：

```text
outputs/benchmark/chunkprefill/capacity_sweep/
```

`capacity_sweep_manifest.json` 记录每轮命令、Seed、输出路径和退出状态。
若某轮 Benchmark 返回非零退出码（例如 OOM），脚本会立即停止，不会自动继续更高请求数。先检查 GPU 状态和失败日志，再决定是否降低请求数、Prompt 长度或 Output 长度。

### 13.3 正式重复实验

只有当 32/40/48 三档 Smoke 全部完成、所有请求均正常完成且无 KV Allocation Failure 后，才建议进行重复实验。例如：

```bash
python benchmarks/chunkprefill/run_capacity_sweep.py \
  --request-counts 32,40,48 \
  --repeats 3 \
  --seed-base 42 \
  --output-dir outputs/benchmark/chunkprefill/capacity_sweep_repeats
```

该脚本是串行执行器，不会并行启动多个模型进程，也不会覆盖已有同名 JSON。默认档位只是保守的实验起点，不代表已证明当前显卡能够承载 48 请求。

**当前状态：** Runner 与 CPU 测试已提交；更高并发 GPU Smoke / 正式实验尚未执行，因此 README 不记录任何高并发性能结论。当前 RTX 3060 Laptop 6GB 不应直接尝试 128/256 请求。