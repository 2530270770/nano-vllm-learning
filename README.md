# nano-vLLM 推理引擎优化：基于 Triton 的算子融合

本项目基于 [nano-vLLM](https://github.com/2530270770/nano-vllm-learning)，使用 Triton 对大语言模型推理过程中的关键算子进行优化，重点研究算子融合、GPU Kernel 性能、显存访问以及端到端推理表现。

项目使用 Qwen3-0.6B 作为测试模型，目标是在理解推理引擎执行流程的基础上，完成算子实现、正确性验证、性能 Benchmark 和引擎集成。

## 1. 项目目标

围绕大模型推理中的关键计算环节，逐步开展以下优化：

- **算子融合**：减少中间张量、显存读写和 Kernel Launch。
- **Kernel 优化**：通过 Triton 实现适合不同输入规模的 GPU Kernel。
- **引擎集成**：将自定义算子接入 nano-vLLM 的实际推理流程。
- **性能评估**：对比 PyTorch Eager、PyTorch Compile 和 Triton 实现。
- **端到端验证**：观察算子优化能否转化为真实推理性能收益。

## 2. 技术栈与实验环境

| 项目 | 配置 |
|---|---|
| 推理框架 | nano-vLLM |
| GPU Kernel | Triton |
| 深度学习框架 | PyTorch 2.6.0 |
| CUDA | 12.4（PyTorch CUDA Runtime） |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| 显存 | 6GB |
| 测试模型 | Qwen3-0.6B |
| 数据类型 | BF16 |
| 开发环境 | WSL2 Ubuntu |

## 3. 已完成工作：SwiGLU 算子融合

### 3.1 算子原理

SwiGLU 是 Transformer FFN 中常见的门控激活结构，其计算过程为：

\[
\operatorname{SwiGLU}(X)
=
\operatorname{SiLU}(X_{\text{gate}})
\odot X_{\text{up}}
\]

其中：

\[
\operatorname{SiLU}(x)=x\cdot \operatorname{sigmoid}(x)
\]

输入张量最后一个维度为 \(2N\)，分别包含 Gate 和 Up 两部分，输出维度为 \(N\)。

传统 PyTorch 写法：

```python
gate, up = x.chunk(2, dim=-1)
output = F.silu(gate) * up
```

计算过程中涉及张量拆分、SiLU 激活和逐元素乘法。

本项目使用 Triton 将 Gate 加载、Up 加载、SiLU 计算和逐元素乘法融合到一个 Kernel 中，直接写入最终输出，避免单独物化 SiLU 中间结果。

### 3.2 实现与集成

核心实现：

- `nanovllm/layers/triton_kernels.py`
  - 实现 SwiGLU Triton Kernel。
  - 使用 FP32 中间计算，最终写回输入对应的数据类型。
  - 支持二维展平后的多行输入。
- `nanovllm/layers/activation.py`
  - 将 Triton 实现接入 `SiluAndMul`。
  - 支持在 Eager、Compile 和 Triton 三种实现之间切换。

通过环境变量选择实现：

```bash
# PyTorch Eager
NANOVLLM_SILU_IMPL=eager python benchmarks/benchmark_e2e.py

# PyTorch Compile
NANOVLLM_SILU_IMPL=compile python benchmarks/benchmark_e2e.py

# Triton
NANOVLLM_SILU_IMPL=triton python benchmarks/benchmark_e2e.py
```

可通过以下环境变量观察推理时遇到的输入形状：

```bash
NANOVLLM_TRACE_SILU=1 \
NANOVLLM_SILU_IMPL=triton \
python benchmarks/benchmark_e2e.py
```

### 3.3 正确性测试

测试文件：

```text
tests/test_silu_and_mul.py
```

测试重点：

- Triton 输出与 PyTorch 参考实现的数值一致性。
- 不同输入形状下的正确性。
- BF16 等数据类型下的误差表现。
- 输入维度和连续性约束。

运行测试：

```bash
pytest tests/test_silu_and_mul.py -v
```

### 3.4 单算子性能 Benchmark

测试文件：

```text
benchmarks/benchmark_swiglu_three_way.py
```

对比三种实现：

1. PyTorch Eager
2. PyTorch Compile
3. Triton

重点测试两种推理场景：

| 场景 | M | K | N |
|---|---:|---:|---:|
| Prefill | 512 | 6144 | 3072 |
| Decode | 4 | 6144 | 3072 |

其中 M 表示输入行数，K 表示输入最后一维长度，N 表示输出最后一维长度。

**实验结果**

以下为稳定性复测中的代表性结果，延迟单位为 μs：

| 场景 | PyTorch Eager | PyTorch Compile | Triton（最佳中位数配置） |
|---|---:|---:|---:|
| Prefill | 76.571 | 36.828 | 35.787 |
| Decode | 6.886 | 2.042 | 2.008 |

对应 Triton 配置：

```text
BLOCK_SIZE = 512
num_warps = 8
```

相对于 PyTorch Compile：

- Prefill 中位延迟降低约 2.91%。
- Decode 中位延迟降低约 1.67%。

需要注意：不同 Triton 配置之间的差异较小，不能仅凭最佳中位数就认定该配置在所有场景下都具有稳定优势。

### 3.5 PyTorch Compile 实现分析

通过 PyTorch Inductor 的代码生成日志，观察到 PyTorch Compile 已经将 SiLU 和 Mul 融合为一个 Triton Kernel。

因此，本项目的 Triton 实现并不是简单地将多个独立 PyTorch Kernel 融合起来，而是在与 PyTorch Inductor 自动生成的融合 Kernel 进行比较。

这也是 Triton 相对 Compile 的性能提升有限的重要原因。

### 3.6 端到端推理 Benchmark

测试模型：Qwen3-0.6B。

测试配置：

| 参数 | 数值 |
|---|---:|
| Requests | 4 |
| Prompt Length | 128 |
| Output Length | 32 |
| max_num_seqs | 4 |
| max_num_batched_tokens | 16384 |
| 正式测试轮数 | 15 |
| 数据类型 | BF16 |

采用同进程交错测试：每轮随机安排 Eager、Compile 和 Triton 的执行顺序，减少固定执行顺序和运行状态对结果的影响。

**实验结果**

以下为 15 轮测试的中位数：

| 指标 | Eager | Compile | Triton |
|---|---:|---:|---:|
| TTFT (ms) | 51.594 | 50.444 | 49.996 |
| TPOT (ms) | 38.125 | 38.847 | 38.099 |
| Throughput (tokens/s) | 103.682 | 102.083 | 103.780 |
| Completion (ms) | 1234.506 | 1253.848 | 1233.351 |

**实验结论**

- Triton 相对于 Eager 的端到端表现基本相当，没有观察到稳定、明显的性能提升。
- Triton 相对于 Compile 的各项中位数略有改善，但幅度较小。
- 单算子延迟的改善没有明显转化为端到端推理性能提升。
- 当前实验不能证明手写 Triton SwiGLU 在端到端场景下显著优于 PyTorch Compile。

本项目保留单算子和端到端两类 Benchmark，避免仅凭局部 Kernel 延迟判断优化是否有效。

## 4. 项目目录

```text
nano-vllm-learning/
├── nanovllm/
│   └── layers/
│       ├── activation.py
│       └── triton_kernels.py
├── benchmarks/
│   ├── benchmark_swiglu_three_way.py
│   ├── benchmark_e2e.py
│   └── benchmark_e2e_interleaved.py
├── tests/
│   └── test_silu_and_mul.py
└── outputs/
    └── benchmark/
```

## 5. 后续计划

### 第一阶段：算子融合

- [x] SwiGLU / SiluAndMul Triton Kernel
- [x] 接入 nano-vLLM
- [x] 正确性测试
- [x] Eager、Compile、Triton 三路单算子 Benchmark
- [x] 端到端交错 Benchmark
- [ ] Add + RMSNorm 融合 Kernel
- [ ] Add + RMSNorm 正确性与性能测试
- [ ] 对比不同实现的端到端效果

### 第二阶段：Attention 优化

- [ ] 深入理解 Attention 计算流程
- [ ] 学习并实现 FlashAttention
- [ ] 研究 GQA 在推理过程中的实现与优化

### 第三阶段：推理引擎优化

- [ ] Chunked Prefill
- [ ] KV Cache 管理优化
- [ ] PagedAttention
- [ ] FP8 量化

## 6. 项目原则

本项目关注的不只是实现一个更快的 Kernel，还包括：

1. 理解原始 PyTorch 算子的计算过程。
2. 分析显存访问、数据布局和 Kernel Launch 开销。
3. 验证 Triton 实现的数值正确性。
4. 使用合理的 Benchmark 方法比较不同实现。
5. 结合 nano-vLLM 实际执行流程评估端到端收益。
6. 如实记录实验结果，不将微小波动描述为显著性能提升。

---

**项目方向：AI Infra / LLM Inference Engine Optimization**

**核心技术：Python、PyTorch、CUDA、Triton、nano-vLLM**