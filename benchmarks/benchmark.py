import sys
sys.path.insert(0, "/home/chen17/nano-vllm")

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


GIB = 1024 ** 3


def percentile(values, p):
    """Nearest-rank percentile."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = max(0, math.ceil(p * len(values)) - 1)
    return values[index]


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "p50": None, "p95": None}

    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def make_prompt_tokens(tokenizer, target_tokens, salt):
    """
    构造指定 Token 数量的测试 Prompt。
    不同 salt 尽量让不同请求具有不同的前缀，
    减少重复 Prompt 对 Prefix Cache 实验的干扰。
    """
    if target_tokens < 1:
        raise ValueError("prompt_tokens 必须 >= 1")

    text = (
        f"Benchmark sample {salt}. "
        "Explain transformer inference, attention, "
        "KV cache, prefill and decode performance. "
    )
    base_ids = tokenizer.encode(text, add_special_tokens=False)

    if not base_ids:
        raise ValueError("Tokenizer 没有生成任何 Token")

    repeats = math.ceil(target_tokens / len(base_ids))
    return (base_ids * repeats)[:target_tokens]


def gpu_memory_snapshot():
    """读取 PyTorch 分配器和 GPU 的当前显存状态。"""
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    return {
        "allocated_gib": torch.cuda.memory_allocated() / GIB,
        "reserved_gib": torch.cuda.memory_reserved() / GIB,
        "gpu_used_gib": (total_bytes - free_bytes) / GIB,
        "gpu_free_gib": free_bytes / GIB,
        "gpu_total_gib": total_bytes / GIB,
    }


def run_batch(
    llm,
    prompts,
    sampling_params,
    run_id,
    measure_memory=True,
):
    """
    直接驱动 LLMEngine 的 add_request()/step()，
    记录每个请求的首 Token、输出进度和完成时间。

    时间戳在 step() 返回后采集，因此是 step 级时间戳，
    并非 CUDA Kernel 级时间戳。
    """
    engine = llm
    request_starts = []

    # 本轮开始前应没有遗留请求。
    if not engine.is_finished():
        raise RuntimeError("引擎仍有未完成请求，不能开始下一轮")

    wall_start = time.perf_counter()

    # 记录每个请求进入引擎的时间。
    for prompt in prompts:
        request_start = time.perf_counter()
        engine.add_request(prompt, sampling_params)
        request_starts.append(request_start)

    # 当前实现的 add_request() 会将新请求放入 waiting。
    # 保留 Sequence 引用，即使请求完成并从队列移除也能读取结果。
    sequences = list(engine.scheduler.waiting)

    if len(sequences) != len(prompts):
        raise RuntimeError(
            f"请求数量不一致：期望 {len(prompts)}，"
            f"实际 {len(sequences)}"
        )

    states = {}

    for index, seq in enumerate(sequences):
        states[seq.seq_id] = {
            "request_id": index,
            "seq_id": seq.seq_id,
            "request_start": request_starts[index],
            "prompt_tokens": seq.num_prompt_tokens,
            "token_times": [],
            "finish_time": None,
        }

    # 不在每个 Token 后额外调用 torch.cuda.synchronize()，
    # 避免人为增加同步开销。记录 step() 返回时的时间。
    while not engine.is_finished():
        engine.step()
        step_end = time.perf_counter()

        for seq in sequences:
            state = states[seq.seq_id]
            token_times = state["token_times"]
            completion_count = seq.num_completion_tokens

            # 一个请求通常每次 step 最多新增一个输出 Token。
            # 如果有多个 Token 未记录，使用同一 step 时间戳；
            # 这也能避免对异常情况静默丢失 Token。
            while len(token_times) < completion_count:
                token_times.append(step_end)

            if seq.is_finished and state["finish_time"] is None:
                state["finish_time"] = step_end

    wall_end = time.perf_counter()

    request_rows = []

    for seq in sequences:
        state = states[seq.seq_id]
        token_times = state["token_times"]

        output_tokens = len(seq.completion_token_ids)
        first_token_time = token_times[0] if token_times else None
        last_token_time = token_times[-1] if token_times else None
        finish_time = state["finish_time"]

        start = state["request_start"]

        ttft = (
            first_token_time - start
            if first_token_time is not None
            else None
        )

        e2e = (
            finish_time - start
            if finish_time is not None
            else None
        )

        # TPOT 只统计首 Token 之后的生成间隔。
        if output_tokens > 1 and last_token_time is not None:
            decode_span = last_token_time - first_token_time
            tpot = decode_span / (output_tokens - 1)
            decode_tps = (
                (output_tokens - 1) / decode_span
                if decode_span > 0
                else None
            )
        else:
            tpot = None
            decode_tps = None

        # 单请求端到端输出速率，包含 TTFT。
        output_tps = (
            output_tokens / e2e
            if e2e is not None and e2e > 0
            else None
        )

        request_rows.append({
            "run_id": run_id,
            "request_id": state["request_id"],
            "seq_id": state["seq_id"],
            "prompt_tokens": state["prompt_tokens"],
            "output_tokens": output_tokens,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "e2e_s": e2e,
            "output_tokens_per_s": output_tps,
            "decode_tokens_per_s": decode_tps,
            "finished": seq.is_finished,
            "generated_text": engine.tokenizer.decode(
                seq.completion_token_ids
            ),
        })

    # 所有请求同时提交时，批次吞吐量用最早提交到最后完成的时间。
    earliest_start = min(request_starts)
    latest_finish = max(
        state["finish_time"]
        for state in states.values()
        if state["finish_time"] is not None
    )

    batch_elapsed = latest_finish - earliest_start
    total_output_tokens = sum(
        row["output_tokens"] for row in request_rows
    )

    batch_row = {
        "run_id": run_id,
        "num_requests": len(request_rows),
        "total_output_tokens": total_output_tokens,
        "wall_time_s": batch_elapsed,
        "aggregate_output_tokens_per_s": (
            total_output_tokens / batch_elapsed
            if batch_elapsed > 0
            else None
        ),
        "step_loop_wall_s": wall_end - wall_start,
    }

    if measure_memory:
        batch_row["memory"] = gpu_memory_snapshot()

    return request_rows, batch_row


def main():
    parser = argparse.ArgumentParser(
        description="nano-vLLM inference benchmark"
    )
    parser.add_argument(
        "--model",
        default="/home/chen17/huggingface/Qwen3-0.6B",
    )
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=16384
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--output-dir", default="outputs/benchmark"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 环境没有可用的 CUDA GPU")

    if args.num_requests < 1 or args.repeats < 1:
        raise ValueError("num-requests 和 repeats 必须 >= 1")

    if args.max_tokens < 1 or args.prompt_tokens < 1:
        raise ValueError("Prompt 和输出 Token 数必须 >= 1")

    if args.max_num_seqs < args.num_requests:
        print(
            "注意：max_num_seqs 小于请求数，"
            "请求可能需要多个 step 才能完成调度。"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("nano-vLLM Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print(f"Requests: {args.num_requests}")
    print(f"Prompt tokens: {args.prompt_tokens}")
    print(f"Max output tokens: {args.max_tokens}")
    print(f"max_num_seqs: {args.max_num_seqs}")
    print(
        "max_num_batched_tokens:",
        args.max_num_batched_tokens,
    )
    print("=" * 64)

    # 初始化模型不计入请求 TTFT，但单独观察模型加载后的显存。
    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # 使用固定长度 Token ID，保证每轮输入长度一致。
    # 每个请求使用不同 salt，尽量避免完全相同的 Prefix。
    def build_prompts(run_id):
        return [
            make_prompt_tokens(
                llm.tokenizer,
                args.prompt_tokens,
                salt=run_id * 1000 + i,
            )
            for i in range(args.num_requests)
        ]

    # 检查模型最大上下文长度。
    if args.prompt_tokens + args.max_tokens > 4096:
        print(
            "注意：请确认 Prompt + 输出长度不超过模型和"
            "当前 Config 的 max_model_len。"
        )

    # 预热：不计入正式统计。
    print("\n[1/3] Warmup")
    for i in range(args.warmup_runs):
        warmup_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=min(args.max_tokens, 8),
            ignore_eos=True,
        )
        run_batch(
            llm,
            build_prompts(-(i + 1)),
            warmup_params,
            run_id=f"warmup-{i + 1}",
            measure_memory=False,
        )
        print(f"  Warmup {i + 1}/{args.warmup_runs} finished")

    # 等待预热工作完成，再重置峰值统计。
    torch.cuda.synchronize()
    baseline_memory = gpu_memory_snapshot()
    torch.cuda.reset_peak_memory_stats()

    all_request_rows = []
    all_batch_rows = []

    print("\n[2/3] Benchmark")
    for run_id in range(args.repeats):
        rows, batch = run_batch(
            llm,
            build_prompts(run_id + 1),
            params,
            run_id=run_id + 1,
            measure_memory=True,
        )

        all_request_rows.extend(rows)
        all_batch_rows.append(batch)

        print(
            f"  Run {run_id + 1}/{args.repeats}: "
            f"{batch['aggregate_output_tokens_per_s']:.2f} "
            "aggregate output tokens/s"
        )

    # 采集本次测量窗口内 PyTorch 分配器的峰值。
    peak_allocated = torch.cuda.max_memory_allocated() / GIB
    peak_reserved = torch.cuda.max_memory_reserved() / GIB
    after_memory = gpu_memory_snapshot()

    memory_summary = {
        "baseline_after_warmup": baseline_memory,
        "peak_torch_allocated_gib": peak_allocated,
        "peak_torch_reserved_gib": peak_reserved,
        "incremental_peak_allocated_gib": max(
            0.0,
            peak_allocated - baseline_memory["allocated_gib"],
        ),
        "after_benchmark": after_memory,
    }

    # 汇总每个请求的指标。
    metric_names = [
        "ttft_s",
        "tpot_s",
        "e2e_s",
        "output_tokens_per_s",
        "decode_tokens_per_s",
    ]

    request_summary = {
        name: summarize([row[name] for row in all_request_rows])
        for name in metric_names
    }

    batch_summary = {
        "aggregate_output_tokens_per_s": summarize([
            row["aggregate_output_tokens_per_s"]
            for row in all_batch_rows
        ]),
        "batch_wall_time_s": summarize([
            row["wall_time_s"] for row in all_batch_rows
        ]),
    }

    summary = {
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(0),
        "request_metrics": request_summary,
        "batch_metrics": batch_summary,
        "memory": memory_summary,
        "runs": all_batch_rows,
    }

    # CSV：逐请求结果。
    csv_path = output_dir / "request_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_request_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_request_rows)

    # JSON：汇总结果和每轮批次吞吐量。
    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[3/3] Results")
    print("\nRequest metrics:")
    for name, stats in request_summary.items():
        print(
            f"  {name:28s} "
            f"mean={stats['mean']} "
            f"p50={stats['p50']} "
            f"p95={stats['p95']}"
        )

    print("\nBatch aggregate output tokens/s:")
    print(batch_summary["aggregate_output_tokens_per_s"])

    print("\nMemory:")
    print(json.dumps(memory_summary, indent=2, ensure_ascii=False))

    print(f"\nCSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    print("\nBenchmark finished.")


if __name__ == "__main__":
    main()
