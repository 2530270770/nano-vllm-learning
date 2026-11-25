import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["baseline", "optimized"],
        required=True,
    )
    parser.add_argument(
        "--scenario",
        choices=["all_at_once", "staggered"],
        required=True,
    )
    parser.add_argument(
        "--model",
        default="/home/chen17/huggingface/Qwen3-0.6B",
    )
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_prompt_ids(tokenizer, request_id, prompt_len):
    """生成长度固定、请求之间内容不同的 Token ID。"""
    text = (
        f"Benchmark request {request_id}. "
        f"This is a performance evaluation prompt. "
        f"Please read the following information carefully. "
    )

    ids = tokenizer.encode(text, add_special_tokens=False)

    # 重复文本，确保得到足够的 Token
    while len(ids) < prompt_len:
        ids.extend(ids)

    return ids[:prompt_len]


def step_engine(engine):
    """兼容 Baseline 和 Optimized 两种 step() 返回值。"""
    result = engine.step()

    if len(result) == 2:
        # Baseline: outputs, signed_num_tokens
        outputs, signed_tokens = result

        if signed_tokens >= 0:
            decode_tokens = 0
            prefill_tokens = signed_tokens
        else:
            decode_tokens = -signed_tokens
            prefill_tokens = 0

    else:
        # Optimized: outputs, decode_tokens, prefill_tokens
        outputs, decode_tokens, prefill_tokens = result

    return outputs, decode_tokens, prefill_tokens


def add_requests(engine, sequences, arrival_times, prompt_ids, indices,
                 sampling_params, benchmark_start):
    """提交请求，并保存对应的 Sequence 对象。"""
    for index in indices:
        arrival_time = time.perf_counter()

        engine.add_request(prompt_ids[index], sampling_params)

        # add_request 将新请求加入 waiting 队列
        seq = engine.scheduler.waiting[-1]

        sequences[index] = seq
        arrival_times[index] = arrival_time - benchmark_start


def warmup(engine, tokenizer):
    """正式测试前执行短请求，预热模型和采样器。"""
    warmup_ids = tokenizer.encode(
        "This is a short benchmark warmup request.",
        add_special_tokens=False,
    )

    engine.add_request(
        warmup_ids,
        SamplingParams(max_tokens=2, ignore_eos=True),
    )

    while not engine.scheduler.is_finished():
        step_engine(engine)

    torch.cuda.synchronize()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if args.num_requests != 4:
        raise ValueError("第一版 Benchmark 暂时固定测试 4 个请求")

    if args.scenario == "all_at_once":
        prompt_len = 128
        max_num_batched_tokens = 16384
    else:
        prompt_len = 512
        max_num_batched_tokens = 512

    print("=" * 60)
    print("Benchmark Configuration")
    print("=" * 60)
    print("Variant:", args.variant)
    print("Scenario:", args.scenario)
    print("Model:", args.model)
    print("Requests:", args.num_requests)
    print("Prompt tokens:", prompt_len)
    print("Output tokens:", args.output_tokens)
    print("Max batched tokens:", max_num_batched_tokens)
    print("Eager mode: True")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    prompt_ids = [
        make_prompt_ids(tokenizer, i, prompt_len)
        for i in range(args.num_requests)
    ]

    # 两个版本使用相同的模型、调度公共参数和采样参数
    engine = LLM(
        args.model,
        max_num_seqs=args.num_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    print("\nWarming up...")
    set_seed(args.seed)
    warmup(engine, tokenizer)

    if not engine.scheduler.is_finished():
        raise RuntimeError("Warmup did not finish cleanly")

    # 清除预热阶段对峰值显存统计的影响
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    # 正式测试重新设置随机种子
    set_seed(args.seed)

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )

    sequences = [None] * args.num_requests
    arrival_times = [None] * args.num_requests
    first_token_times = [None] * args.num_requests
    finish_times = [None] * args.num_requests

    step_records = []

    torch.cuda.synchronize()
    benchmark_start = time.perf_counter()

    if args.scenario == "all_at_once":
        add_requests(
            engine,
            sequences,
            arrival_times,
            prompt_ids,
            range(args.num_requests),
            sampling_params,
            benchmark_start,
        )
    else:
        # 先提交第一个请求
        add_requests(
            engine,
            sequences,
            arrival_times,
            prompt_ids,
            [0],
            sampling_params,
            benchmark_start,
        )

    step_id = 0

    while not engine.scheduler.is_finished():
        torch.cuda.synchronize()
        step_start = time.perf_counter()

        _, decode_tokens, prefill_tokens = step_engine(engine)

        torch.cuda.synchronize()
        step_end = time.perf_counter()

        step_records.append({
            "step": step_id,
            "step_time_ms": (step_end - step_start) * 1000,
            "decode_tokens": decode_tokens,
            "prefill_tokens": prefill_tokens,
        })

        # 记录首 Token 时间和请求完成时间
        for i, seq in enumerate(sequences):
            if seq is None:
                continue

            if (
                first_token_times[i] is None
                and seq.num_completion_tokens > 0
            ):
                first_token_times[i] = step_end - benchmark_start

            if seq.is_finished and finish_times[i] is None:
                finish_times[i] = step_end - benchmark_start

        # 第一次 Engine Step 结束后，再提交剩余请求
        if args.scenario == "staggered" and step_id == 0:
            add_requests(
                engine,
                sequences,
                arrival_times,
                prompt_ids,
                range(1, args.num_requests),
                sampling_params,
                benchmark_start,
            )

        step_id += 1

        if step_id > 10000:
            raise RuntimeError("Too many steps; benchmark may be stuck")

    torch.cuda.synchronize()
    benchmark_end = time.perf_counter()

    total_time = benchmark_end - benchmark_start

    request_results = []

    for i, seq in enumerate(sequences):
        if seq is None:
            raise RuntimeError(f"Request {i} was never submitted")

        if first_token_times[i] is None or finish_times[i] is None:
            raise RuntimeError(f"Request {i} did not finish correctly")

        completion_tokens = seq.num_completion_tokens

        if completion_tokens != args.output_tokens:
            raise RuntimeError(
                f"Request {i}: expected {args.output_tokens} output tokens, "
                f"got {completion_tokens}"
            )

        ttft = first_token_times[i] - arrival_times[i]
        latency = finish_times[i] - arrival_times[i]

        if completion_tokens > 1:
            tpot = (
                finish_times[i] - first_token_times[i]
            ) / (completion_tokens - 1)
        else:
            tpot = None

        request_results.append({
            "request_id": i,
            "prompt_tokens": len(prompt_ids[i]),
            "output_tokens": completion_tokens,
            "arrival_time_ms": arrival_times[i] * 1000,
            "ttft_ms": ttft * 1000,
            "tpot_ms": None if tpot is None else tpot * 1000,
            "request_latency_ms": latency * 1000,
        })

    total_output_tokens = sum(
        item["output_tokens"] for item in request_results
    )

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    total_decode_tokens = sum(
        item["decode_tokens"] for item in step_records
    )
    total_prefill_tokens = sum(
        item["prefill_tokens"] for item in step_records
    )

    result = {
        "variant": args.variant,
        "scenario": args.scenario,
        "model": args.model,
        "config": {
            "num_requests": args.num_requests,
            "prompt_tokens": prompt_len,
            "output_tokens": args.output_tokens,
            "max_num_seqs": args.num_requests,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_model_len": 1024,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "temperature": 1.0,
            "seed": args.seed,
        },
        "summary": {
            "total_time_ms": total_time * 1000,
            "total_output_tokens": total_output_tokens,
            "output_throughput": total_output_tokens / total_time,
            "mean_ttft_ms": sum(
                x["ttft_ms"] for x in request_results
            ) / len(request_results),
            "mean_tpot_ms": sum(
                x["tpot_ms"] for x in request_results
            ) / len(request_results),
            "mean_request_latency_ms": sum(
                x["request_latency_ms"] for x in request_results
            ) / len(request_results),
            "total_steps": len(step_records),
            "total_prefill_tokens": total_prefill_tokens,
            "total_decode_tokens": total_decode_tokens,
            "initial_allocated_mib": initial_allocated / 1024**2,
            "initial_reserved_mib": initial_reserved / 1024**2,
            "peak_allocated_mib": peak_allocated / 1024**2,
            "peak_reserved_mib": peak_reserved / 1024**2,
            "peak_allocated_delta_mib": (
                peak_allocated - initial_allocated
            ) / 1024**2,
            "peak_reserved_delta_mib": (
                peak_reserved - initial_reserved
            ) / 1024**2,
        },
        "requests": request_results,
        "steps": step_records,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path("outputs/benchmark/chunkprefill")
            / f"{args.scenario}.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Total time:          {total_time * 1000:.2f} ms")
    print(f"Output throughput:   {total_output_tokens / total_time:.2f} tokens/s")
    print(f"Mean TTFT:           {result['summary']['mean_ttft_ms']:.2f} ms")
    print(f"Mean TPOT:           {result['summary']['mean_tpot_ms']:.2f} ms")
    print(f"Mean request latency:{result['summary']['mean_request_latency_ms']:.2f} ms")
    print(f"Total steps:         {len(step_records)}")
    print(f"Peak allocated:      {peak_allocated / 1024**2:.2f} MiB")
    print(f"Peak reserved:       {peak_reserved / 1024**2:.2f} MiB")
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
