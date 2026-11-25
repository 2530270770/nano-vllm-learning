# Mixed-Batch Chunked Prefill：nano-vLLM 调度优化

## 1. 项目背景

本项目基于 [nano-vLLM](https://github.com/2530270770/nano-vllm-learning)，实现 Mixed-Batch Chunked Prefill 调度优化。

在 LLM 推理过程中，主要存在两种计算阶段：

- **Prefill**：处理输入 Prompt，计算并写入 KV Cache。
- **Decode**：基于已有 KV Cache，自回归地逐 Token 生成输出。

原有调度方式在处理多个请求时，Prefill 和 Decode 的调度存在一定局限：

1. 长 Prompt 的 Prefill 可能占用较大的计算预算，影响 Decode 请求的执行。
2. 多个 Prefill 请求同时等待时，单个长请求可能占用大量 Token Budget。
3. Prefill 和 Decode 的计算特征不同，需要采用不同的执行策略。
4. 当多个请求混合到达时，需要合理分配 GPU 计算预算和并发执行名额。

因此，本项目希望实现一种 **Decode 优先、Prefill 分块、混合执行** 的调度机制，在保证请求正确执行的前提下，提高不同请求之间的资源利用效率。

## 2. 项目目标

本项目主要实现以下目标：

- **Decode 优先调度**：优先保证正在生成 Token 的请求获得执行机会，降低 Prefill 对 Decode 的影响。
- **Mixed-Batch 执行**：在同一个 Engine Step 中，同时安排 Decode 和 Prefill 请求。
- **Chunked Prefill**：将较长的 Prompt 拆分成多个 Chunk，分多个 Step 完成 Prefill。
- **公平分配 Token Budget**：将 Decode 执行后的剩余预算分配给多个 Prefill 请求，避免单个长请求占用全部预算。
- **KV Cache 状态管理**：确保跨 Step 的 Prefill 能够正确维护 KV Cache 和请求进度。
- **保持原有推理流程兼容**：尽量复用 nano-vLLM 原有的 Scheduler、ModelRunner 和 LLMEngine 架构。

## 3. 核心设计

### 3.1 Mixed-Batch 调度

每个 Engine Step 分为两个执行阶段：

1. Decode Forward
2. Prefill Forward

执行顺序如下：

```text
                 LLMEngine.step()
                        |
                        v
                 Scheduler.schedule()
                        |
              +---------+---------+
              |                   |
              v                   v
         Decode Batch        Prefill Batch
              |                   |
              v                   |
        run_decode()              |
              |                   |
              v                   |
       Decode Postprocess         |
              |                   |
              +---------+---------+
                        |
                        v
                 run_prefill()
                        |
                        v
               Prefill Postprocess
                        |
                        v
                   Next Step
```

这里的 Mixed-Batch 并不是将 Decode 和 Prefill 合并成一个统一的 GPU Tensor Batch，而是在同一个 Engine Step 中分别执行两个 Forward。

这样可以针对两种计算阶段采用不同的执行方式：

- Decode：使用独立的 Decode 执行路径，在支持的配置下使用 CUDA Graph。
- Prefill：使用独立的 Prefill 执行路径，采用 Eager 模式处理分块输入。

### 3.2 Decode 优先调度

Scheduler 首先从 Running 队列中选择 Decode 请求。

Decode 调度同时受到两个条件约束：

- Decode 并发数量上限。
- 全局 Token Budget 上限。

设：

- \(M_s\)：最大并发请求数。
- \(M_d\)：最大 Decode 请求数。
- \(M_b\)：最大 Batch Token 数。

Decode 最大调度数量为：

\[
D_{\max}=\min(M_d,M_s,M_b)
\]

实际 Decode 数量为 \(D\)，每个 Decode 请求在当前 Step 消耗一个 Token Budget。

因此，Prefill 可以使用的剩余资源为：

\[
S_p=M_s-D
\]

\[
B_p=M_b-D
\]

其中：

- \(S_p\)：Prefill 剩余并发名额。
- \(B_p\)：Prefill 剩余 Token Budget。

### 3.3 Chunked Prefill

对于一个尚未完成 Prefill 的请求，不再要求一次性处理全部剩余 Prompt Token，而是根据当前可用预算分配一部分 Token。

假设一个请求还剩 \(R_i\) 个 Prompt Token，则当前 Step 分配的 Chunk 大小为：

\[
0\le C_i\le R_i
\]

所有 Prefill 请求需要满足：

\[
\sum_i C_i\le B_p
\]

每个 Step 完成后：

- 更新请求已经处理的 Prompt Token 数量。
- 保留已经计算的 KV Cache。
- 尚未完成 Prefill 的请求重新进入 Waiting 队列。
- 已经完成 Prefill 的请求进入 Running 队列，后续参与 Decode。

这样，长 Prompt 可以跨多个 Engine Step 完成计算，不必一次性占用大量 Token Budget。

### 3.4 Integer Water-Filling 公平分配

当多个 Prefill 请求同时等待时，采用 Integer Water-Filling 策略分配剩余 Token Budget。

分配时考虑：

- 每个请求剩余的 Prompt Token 数量。
- 当前剩余 Token Budget。
- 当前可用 Prefill 并发名额。

目标是在满足总预算约束的情况下，尽可能均衡地分配 Chunk。

对于每个请求：

\[
C_i=\min(R_i,L)
\]

其中 \(L\) 为当前可分配的均分水位。

由于 Token 数量必须为整数，分配过程中还需要处理剩余的 Token，将剩余预算依次分配给尚未满足需求的请求。

该策略可以避免简单轮询或按请求长度分配造成的预算使用不均衡。

### 3.5 KV Cache 管理

当前实现采用以下方式管理 Prefill 的 KV Cache：

- 请求首次进入 Prefill 时，为整条 Sequence 分配所需 KV Cache。
- 后续 Chunk 计算复用已经分配的 KV Cache。
- 每次 Prefill Forward 只处理当前 Chunk 对应的 Token。
- 通过 `num_cached_tokens` 等状态记录 Prefill 进度。
- Prefill 尚未完成时，不提前将其当作 Decode 请求处理。
- Prefill 完成后，才进入正常 Decode 生命周期。

需要注意：**当前实现的 Chunked Prefill 是分块计算，不是逐 Chunk 增量分配物理 KV Cache。**

如果首次分配整条 Sequence 的 KV Cache 失败，Scheduler 会跳过当前请求，继续尝试后续 Waiting 请求，避免单个请求的分配失败阻塞其他请求。

## 4. 主要修改模块

| 文件 | 主要修改内容 |
|---|---|
| `nanovllm/engine/scheduler.py` | 实现 Decode 优先调度、剩余预算计算、Prefill 候选筛选、Integer Water-Filling、Chunked Prefill 和 KV 分配失败处理 |
| `nanovllm/engine/model_runner.py` | 将 Decode 和 Prefill 拆分为独立执行路径，分别处理两种 Forward |
| `nanovllm/engine/llm_engine.py` | 调整 Engine Step 执行流程，先执行 Decode，再执行 Prefill，并分别完成 Postprocess |
| `tests/test_scheduler.py` | 增加调度策略、预算分配、KV 分配失败、跨 Step Chunked Prefill 和 Decode 名额管理等测试 |
| `tests/test_model_runner.py` | 覆盖 ModelRunner 相关执行逻辑 |
| `tests/test_model_runner_gpu.py` | 增加 GPU 测试资源清理，降低测试之间的显存干扰 |
| `tests/test_llm_engine.py` | 覆盖 Engine 调度与执行流程 |
| `tests/test_llm_engine_gpu.py` | 增加真实 GPU Mixed-Batch 集成测试 |

## 5. 测试验证

### 5.1 Scheduler 单元测试

主要验证：

- 只有 Prefill 请求时的调度。
- 只有 Decode 请求时的调度。
- Decode 和 Prefill 混合调度。
- Decode 独立并发上限。
- Decode 与 Prefill 共享 Token Budget。
- KV Cache 分配失败后跳过当前请求。
- Integer Water-Filling 分配结果。
- Chunked Prefill 跨多个 Step 执行。
- Decode 请求发生 KV Cache 压力时的抢占处理。
- Decode 请求结束后，Prefill 请求能够接管 Decode 名额。
- Prefill 并发名额不足时，剩余请求能够继续保留在 Waiting 队列。

### 5.2 GPU 集成测试

使用 Qwen3-0.6B 在真实 GPU 上验证 Mixed-Batch 执行流程。

测试覆盖：

- 真实 Scheduler、ModelRunner 和 LLMEngine。
- 同一个 Engine Step 中先执行 Decode，再执行 Prefill。
- 多个 Prefill 请求分多个 Step 完成。
- Prefill 完成后进入 Decode。
- 不同请求的 Prefill 进度能够独立维护。

一次测试中观察到的执行顺序：

```text
['run_decode', 'run_prefill']
```

多轮 Chunked Prefill 的进度验证：

```text
(64, 62, 63)
(96, 93, 63)
(97, 96, 3)
```

最终两个请求分别缓存了 97 和 96 个 Prompt Token，并正常进入后续生成阶段。

### 5.3 完整测试结果

执行命令：

```bash
RUN_MODEL_RUNNER_GPU_TESTS=1 \
NANOVLLM_MODEL_PATH=/home/chen17/huggingface/Qwen3-0.6B \
python -m pytest -v -s \
tests/test_scheduler.py \
tests/test_model_runner.py \
tests/test_model_runner_gpu.py \
tests/test_llm_engine.py \
tests/test_llm_engine_gpu.py
```

当前测试结果：

```text
21 passed in 18.95s
```

测试覆盖 Scheduler、ModelRunner、LLMEngine 的单元测试和 GPU 集成测试。

## 6. 当前实现的边界

当前版本主要完成了 Mixed-Batch Chunked Prefill 的功能实现和正确性验证。

仍有一些值得进一步研究的方向：

1. **性能 Benchmark**：对比优化前后的 TTFT、TPOT、吞吐量和显存使用情况。
2. **不同负载下的性能分析**：测试不同 Prompt 长度、并发请求数和 Decode/Prefill 混合比例。
3. **增量 KV Cache 分配**：研究是否可以在 Prefill 过程中按需分配 KV Cache Block，降低长 Prompt 请求首次分配时的显存压力。
4. **CUDA Graph 兼容性**：进一步验证不同混合调度配置下的 CUDA Graph 执行表现。

需要强调的是：目前测试通过说明调度功能和执行流程通过了相应的正确性验证，**并不代表已经证明端到端推理性能得到提升**。实际性能收益需要通过统一配置下的 Benchmark 得出。

## 7. 项目总结

本项目围绕 nano-vLLM 的推理调度流程，实现了 Mixed-Batch Chunked Prefill：

- 将 Decode 和 Prefill 拆分为独立执行路径。
- 在同一个 Engine Step 中优先执行 Decode，再利用剩余预算执行 Prefill。
- 通过 Chunked Prefill 支持长 Prompt 跨多个 Step 处理。
- 通过 Integer Water-Filling 实现多个 Prefill 请求之间的预算分配。
- 处理 KV Cache 分配失败、请求状态流转和 Decode 抢占等边界情况。
- 通过 Scheduler 单元测试和真实 GPU 集成测试验证实现正确性。

项目的核心目标是探索 LLM 推理过程中 Decode 与 Prefill 的协同调度方式，为后续推理吞吐优化、KV Cache 管理优化和更复杂的批处理策略打下基础。