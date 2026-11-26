import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def parse_prompt_lengths(value):
    try:
        lengths = [
            int(item.strip())
            for item in value.split(",")
        ]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Prompt lengths must be comma-separated integers."
        )

    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError(
            "Prompt lengths must all be positive integers."
        )

    return lengths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["baseline", "optimized"],
        required=True,
    )
    parser.add_argument(
        "--scenario",
        choices=["all_at_once", "staggered", "stress"],
        required=True,
    )
    parser.add_argument(
        "--model",
        default="/home/chen17/huggingface/Qwen3-0.6B",
    )
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument(
        "--staggered-initial-requests",
        type=int,
        default=4,
        help="Number of requests in the first staggered-arrival batch.",
    )
    parser.add_argument(
        "--staggered-batch-size",
        type=int,
        default=4,
        help="Number of requests in each subsequent staggered-arrival batch.",
    )
    parser.add_argument(
        "--staggered-interval-steps",
        type=int,
        default=2,
        help="Number of Engine steps between staggered-arrival batches.",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=parse_prompt_lengths,
        default=[128, 512, 1024],
        help="Comma-separated prompt lengths, cycled across requests.",
    )
    parser.add_argument(
        "--max-num-decode-seqs",
        type=int,
        default=8,
        help="Maximum Decode sequences scheduled per Engine step.",
    )
    parser.add_argument(
        "--max-num-prefill-seqs",
        type=int,
        default=None,
        help="Maximum number of Prefill requests scheduled per step. None means unlimited.",
    )
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--enable-prefill-aging",
        action="store_true",
        help="Enable Prefill Aging scheduling.",
    )
    parser.add_argument(
    "--execution-mode",
    choices=["auto", "eager", "cudagraph"],
    default="auto",
    help="Execution mode: auto follows variant, eager disables CUDA Graph, cudagraph enables it.",
)
    return parser.parse_args()


def percentile(values, percentile_value):
    """Nearest-rank percentile; returns None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int((percentile_value * len(ordered) + 0.999999)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def summarize_wait_metrics(requests):
    """Summarize queueing delay until a request is first selected for Prefill."""
    wait_steps = [item["prefill_wait_steps"] for item in requests]
    wait_ms = [item["prefill_queue_wait_ms"] for item in requests]
    return {
        "request_count": len(requests),
        "mean_wait_steps": (
            sum(wait_steps) / len(wait_steps) if wait_steps else None
        ),
        "p50_wait_steps": percentile(wait_steps, 0.50),
        "p95_wait_steps": percentile(wait_steps, 0.95),
        "max_wait_steps": max(wait_steps) if wait_steps else None,
        "mean_wait_ms": sum(wait_ms) / len(wait_ms) if wait_ms else None,
        "p50_wait_ms": percentile(wait_ms, 0.50),
        "p95_wait_ms": percentile(wait_ms, 0.95),
        "max_wait_ms": max(wait_ms) if wait_ms else None,
    }


def summarize_prefill_fairness(requests):
    """Summarize per-request progress and waiting across all Prefill chunks."""
    total_wait = [item["prefill_total_wait_steps"] for item in requests]
    max_consecutive_wait = [
        item["prefill_max_consecutive_wait_steps"] for item in requests
    ]
    chunk_counts = [item["prefill_chunk_count"] for item in requests]
    completion_latency = [
        item["prefill_completion_latency_ms"] for item in requests
    ]
    return {
        "request_count": len(requests),
        "mean_total_wait_steps": (
            sum(total_wait) / len(total_wait) if total_wait else None
        ),
        "p50_total_wait_steps": percentile(total_wait, 0.50),
        "p95_total_wait_steps": percentile(total_wait, 0.95),
        "max_total_wait_steps": max(total_wait) if total_wait else None,
        "mean_max_consecutive_wait_steps": (
            sum(max_consecutive_wait) / len(max_consecutive_wait)
            if max_consecutive_wait else None
        ),
        "max_consecutive_wait_steps": (
            max(max_consecutive_wait) if max_consecutive_wait else None
        ),
        "mean_chunk_count": (
            sum(chunk_counts) / len(chunk_counts) if chunk_counts else None
        ),
        "mean_prefill_completion_latency_ms": (
            sum(completion_latency) / len(completion_latency)
            if completion_latency else None
        ),
        "p50_prefill_completion_latency_ms": percentile(
            completion_latency, 0.50
        ),
        "p95_prefill_completion_latency_ms": percentile(
            completion_latency, 0.95
        ),
        "max_prefill_completion_latency_ms": (
            max(completion_latency) if completion_latency else None
        ),
    }


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

    if args.execution_mode == "auto":
        enforce_eager = (args.variant == "baseline")
    elif args.execution_mode == "eager":
        enforce_eager = True
    else:
        enforce_eager = False

    execution_mode = "eager" if enforce_eager else "cudagraph"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.max_num_decode_seqs <= 0:
        raise ValueError("--max-num-decode-seqs must be positive")

    if args.scenario == "staggered":
        if not 1 <= args.staggered_initial_requests <= args.num_requests:
            raise ValueError(
                "--staggered-initial-requests must be between 1 and "
                "--num-requests"
            )
        if args.staggered_batch_size <= 0:
            raise ValueError("--staggered-batch-size must be positive")
        if args.staggered_interval_steps <= 0:
            raise ValueError("--staggered-interval-steps must be positive")

    if args.scenario == "all_at_once":
        max_num_batched_tokens = 16384

    elif args.scenario in ("staggered", "stress"):
        max_num_batched_tokens = 512

    # 按请求 ID 循环分配 Prompt 长度
    prompt_lengths = [
        args.prompt_lengths[i % len(args.prompt_lengths)]
        for i in range(args.num_requests)
    ]

    print("=" * 60)
    print("Benchmark Configuration")
    print("=" * 60)
    print("Variant:", args.variant)
    print("Scenario:", args.scenario)
    print("Model:", args.model)
    print("Requests:", args.num_requests)
    print("Prompt lengths:", args.prompt_lengths)
    print(
        "Prompt length distribution:",
        {
            length: prompt_lengths.count(length)
            for length in sorted(set(prompt_lengths))
        },
    )
    print("Output tokens:", args.output_tokens)
    print("Max batched tokens:", max_num_batched_tokens)
    print("Execution mode:", execution_mode)
    print("Eager mode:", enforce_eager)
    print("Max Decode sequences:", args.max_num_decode_seqs)
    print("Max Prefill sequences:", args.max_num_prefill_seqs)
    if args.scenario == "staggered":
        print(
            "Staggered arrival:",
            f"initial={args.staggered_initial_requests},",
            f"batch_size={args.staggered_batch_size},",
            f"interval_steps={args.staggered_interval_steps}",
        )
    

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    prompt_ids = [
        make_prompt_ids(tokenizer, i, prompt_lengths[i])
        for i in range(args.num_requests)
    ]

    # 两个版本使用相同的模型、调度公共参数和采样参数
    engine = LLM(
        args.model,
        max_num_seqs=args.num_requests,
        max_num_decode_seqs=args.max_num_decode_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
        max_num_prefill_seqs=args.max_num_prefill_seqs,
        enable_prefill_aging=args.enable_prefill_aging,
    )

    print("\nWarming up...")
    set_seed(args.seed)
    warmup(engine, tokenizer)

    if not engine.scheduler.is_finished():
        raise RuntimeError("Warmup did not finish cleanly")

    # 清除预热阶段对峰值显存统计的影响
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Exclude warmup activity from ModelRunner diagnostics.
    for key, value in engine.model_runner.execution_stats.items():
        if hasattr(value, "clear"):
            value.clear()
        else:
            engine.model_runner.execution_stats[key] = 0

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
    arrival_steps = [None] * args.num_requests
    arrival_batch_ids = [None] * args.num_requests
    first_token_times = [None] * args.num_requests
    finish_times = [None] * args.num_requests

    step_records = []
    scheduler_step_records = []
    prefill_chunk_records = []
    first_prefill_steps = {}
    first_prefill_times = {}
    prefill_completion_steps = {}
    prefill_completion_times = {}
    prefill_total_wait_steps = [0] * args.num_requests
    prefill_current_wait_steps = [0] * args.num_requests
    prefill_max_consecutive_wait_steps = [0] * args.num_requests
    schedule_observer = {
        "step": 0,
        "decode_request_ids": [],
        "prefill_request_ids": [],
        "prefill_chunks": [],
        "schedule_time_ms": 0.0,
        "gpu_events": [],
    }

    torch.cuda.synchronize()
    benchmark_start = time.perf_counter()

    # Observe the scheduler's actual output without changing scheduling policy.
    # This measures queue wait until first Prefill selection, not GPU completion.
    original_schedule = engine.scheduler.schedule

    def observed_schedule():
        schedule_start = time.perf_counter()
        schedule_output = original_schedule()
        schedule_end = time.perf_counter()
        schedule_observer["schedule_time_ms"] = (
            schedule_end - schedule_start
        ) * 1000
        schedule_time = schedule_end - benchmark_start
        seq_to_request = {
            seq.seq_id: index
            for index, seq in enumerate(sequences)
            if seq is not None
        }
        decode_ids = [
            seq_to_request[seq.seq_id]
            for seq in schedule_output.decode_seqs
            if seq.seq_id in seq_to_request
        ]
        prefill_ids = [
            seq_to_request[seq.seq_id]
            for seq in schedule_output.prefill_seqs
            if seq.seq_id in seq_to_request
        ]
        schedule_observer["decode_request_ids"] = decode_ids
        schedule_observer["prefill_request_ids"] = prefill_ids
        schedule_observer["prefill_chunks"] = []

        for seq in schedule_output.prefill_seqs:
            request_id = seq_to_request.get(seq.seq_id)
            if request_id is None:
                continue
            chunk = {
                "request_id": request_id,
                "seq_id": seq.seq_id,
                "step": schedule_observer["step"],
                "prompt_tokens": seq.num_prompt_tokens,
                "cached_tokens_before": seq.num_cached_tokens,
                "allocated_tokens": seq.num_scheduled_tokens,
                "cached_tokens_after_schedule": (
                    seq.num_cached_tokens + seq.num_scheduled_tokens
                ),
                "remaining_tokens_before": max(
                    0, seq.num_prompt_tokens - seq.num_cached_tokens
                ),
                "prefill_completed": False,
            }
            schedule_observer["prefill_chunks"].append(chunk)
            prefill_chunk_records.append(chunk)

        for request_id in prefill_ids:
            if request_id not in first_prefill_steps:
                first_prefill_steps[request_id] = schedule_observer["step"]
                first_prefill_times[request_id] = schedule_time

        return schedule_output

    engine.scheduler.schedule = observed_schedule

    # CUDA events measure GPU stream time without synchronizing each stage.
    # This avoids adding a per-call synchronization point to the benchmark.
    original_model_runner_call = engine.model_runner.call

    def observed_model_runner_call(method_name, *call_args):
        if method_name not in ("run_decode", "run_prefill"):
            return original_model_runner_call(method_name, *call_args)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = original_model_runner_call(method_name, *call_args)
        end_event.record()
        schedule_observer["gpu_events"].append(
            (method_name, start_event, end_event)
        )
        return result

    engine.model_runner.call = observed_model_runner_call

    submitted_count = 0
    next_arrival_step = 0
    arrival_batch_id = 0

    if args.scenario in ("all_at_once", "stress"):
        # all_at_once 和 stress：一次性提交所有请求
        indices = range(args.num_requests)
        add_requests(
            engine,
            sequences,
            arrival_times,
            prompt_ids,
            indices,
            sampling_params,
            benchmark_start,
        )
        for i in indices:
            arrival_steps[i] = 0
            arrival_batch_ids[i] = 0
        submitted_count = args.num_requests
    else:
        print(
            "Staggered arrival plan:",
            f"first batch={args.staggered_initial_requests},",
            f"later batch size={args.staggered_batch_size},",
            f"every {args.staggered_interval_steps} Engine steps",
        )

    step_id = 0

    # 对 staggered 而言，即使当前队列暂时为空，只要还有未提交请求，
    # benchmark 就继续，并在下一次循环提交下一批请求。
    while (
        submitted_count < args.num_requests
        or not engine.scheduler.is_finished()
    ):
        newly_arrived = []

        if args.scenario == "staggered":
            should_arrive = (
                submitted_count < args.num_requests
                and (
                    step_id >= next_arrival_step
                    or engine.scheduler.is_finished()
                )
            )

            if should_arrive:
                if arrival_batch_id == 0:
                    batch_size = args.staggered_initial_requests
                else:
                    batch_size = args.staggered_batch_size

                batch_end = min(
                    submitted_count + batch_size,
                    args.num_requests,
                )
                newly_arrived = list(range(submitted_count, batch_end))

                add_requests(
                    engine,
                    sequences,
                    arrival_times,
                    prompt_ids,
                    newly_arrived,
                    sampling_params,
                    benchmark_start,
                )

                for i in newly_arrived:
                    arrival_steps[i] = step_id
                    arrival_batch_ids[i] = arrival_batch_id

                submitted_count = batch_end
                arrival_batch_id += 1
                next_arrival_step = (
                    step_id + args.staggered_interval_steps
                )

        schedule_observer["step"] = step_id
        schedule_observer["decode_request_ids"] = []
        schedule_observer["prefill_request_ids"] = []
        schedule_observer["prefill_chunks"] = []
        schedule_observer["schedule_time_ms"] = 0.0
        schedule_observer["gpu_events"] = []

        torch.cuda.synchronize()
        step_start = time.perf_counter()

        _, decode_tokens, prefill_tokens = step_engine(engine)
        scheduler_step_records.append(dict(engine.scheduler.last_schedule_stats))

        torch.cuda.synchronize()
        step_end = time.perf_counter()

        # Count steps in which an incomplete Prompt received no Prefill service.
        selected_prefill_ids = set(schedule_observer["prefill_request_ids"])
        for request_id, seq in enumerate(sequences):
            if seq is None or seq.num_cached_tokens >= seq.num_prompt_tokens:
                prefill_current_wait_steps[request_id] = 0
                continue
            if request_id in selected_prefill_ids:
                prefill_current_wait_steps[request_id] = 0
            else:
                prefill_total_wait_steps[request_id] += 1
                prefill_current_wait_steps[request_id] += 1
                prefill_max_consecutive_wait_steps[request_id] = max(
                    prefill_max_consecutive_wait_steps[request_id],
                    prefill_current_wait_steps[request_id],
                )

        for chunk in schedule_observer["prefill_chunks"]:
            request_id = chunk["request_id"]
            seq = sequences[request_id]
            chunk["cached_tokens_after"] = seq.num_cached_tokens
            chunk["prefill_completed"] = (
                seq.num_cached_tokens >= seq.num_prompt_tokens
            )
            if (
                chunk["prefill_completed"]
                and request_id not in prefill_completion_steps
            ):
                prefill_completion_steps[request_id] = step_id
                prefill_completion_times[request_id] = (
                    step_end - benchmark_start
                )

        stage_gpu_times = {"run_decode": 0.0, "run_prefill": 0.0}
        for method_name, start_event, end_event in schedule_observer["gpu_events"]:
            stage_gpu_times[method_name] += start_event.elapsed_time(end_event)

        step_records.append({
            "step": step_id,
            "step_time_ms": (step_end - step_start) * 1000,
            "scheduler_schedule_time_ms": schedule_observer["schedule_time_ms"],
            "decode_gpu_time_ms": stage_gpu_times["run_decode"],
            "prefill_gpu_time_ms": stage_gpu_times["run_prefill"],
            "decode_tokens": decode_tokens,
            "prefill_tokens": prefill_tokens,
            "newly_arrived_request_ids": newly_arrived,
            "decode_request_ids": schedule_observer["decode_request_ids"],
            "prefill_request_ids": schedule_observer["prefill_request_ids"],
            "prefill_chunks": [
                dict(chunk) for chunk in schedule_observer["prefill_chunks"]
            ],
            "waiting_queue_size": len(engine.scheduler.waiting),
            "running_queue_size": len(engine.scheduler.running),
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
        if i not in first_prefill_steps or i not in first_prefill_times:
            raise RuntimeError(
                f"Request {i} completed without being observed in a Prefill batch"
            )

        completion_tokens = seq.num_completion_tokens

        if completion_tokens != args.output_tokens:
            raise RuntimeError(
                f"Request {i}: expected {args.output_tokens} output tokens, "
                f"got {completion_tokens}"
            )

        ttft = first_token_times[i] - arrival_times[i]
        latency = finish_times[i] - arrival_times[i]
        prefill_wait_steps = first_prefill_steps[i] - arrival_steps[i]
        prefill_queue_wait_ms = (
            first_prefill_times[i] - arrival_times[i]
        ) * 1000

        if prefill_wait_steps < 0 or prefill_queue_wait_ms < -0.1:
            raise RuntimeError(
                f"Request {i} has invalid Prefill wait metrics: "
                f"steps={prefill_wait_steps}, ms={prefill_queue_wait_ms:.3f}"
            )

        if completion_tokens > 1:
            tpot = (
                finish_times[i] - first_token_times[i]
            ) / (completion_tokens - 1)
        else:
            tpot = None

        if i not in prefill_completion_steps or i not in prefill_completion_times:
            raise RuntimeError(
                f"Request {i} did not complete Prompt Prefill"
            )

        request_chunks = [
            dict(chunk)
            for chunk in prefill_chunk_records
            if chunk["request_id"] == i
        ]
        prefill_completion_latency_ms = (
            prefill_completion_times[i] - arrival_times[i]
        ) * 1000

        request_results.append({
            "request_id": i,
            "prompt_tokens": len(prompt_ids[i]),
            "output_tokens": completion_tokens,
            "arrival_time_ms": arrival_times[i] * 1000,
            "arrival_step": arrival_steps[i],
            "arrival_batch_id": arrival_batch_ids[i],
            "first_prefill_step": first_prefill_steps[i],
            "first_prefill_schedule_time_ms": first_prefill_times[i] * 1000,
            "prefill_wait_steps": prefill_wait_steps,
            "prefill_queue_wait_ms": max(0.0, prefill_queue_wait_ms),
            "prefill_chunk_count": len(request_chunks),
            "prefill_chunks": request_chunks,
            "prefill_completion_step": prefill_completion_steps[i],
            "prefill_completion_time_ms": prefill_completion_times[i] * 1000,
            "prefill_completion_latency_ms": prefill_completion_latency_ms,
            "prefill_total_wait_steps": prefill_total_wait_steps[i],
            "prefill_max_consecutive_wait_steps": (
                prefill_max_consecutive_wait_steps[i]
            ),
            "prefill_total_allocated_tokens": sum(
                chunk["allocated_tokens"] for chunk in request_chunks
            ),
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

    wait_by_prompt_length = {}
    for prompt_length in sorted(set(prompt_lengths)):
        group = [
            item for item in request_results
            if item["prompt_tokens"] == prompt_length
        ]
        wait_by_prompt_length[str(prompt_length)] = (
            summarize_wait_metrics(group)
        )

    result = {
        "variant": args.variant,
        "scenario": args.scenario,
        "model": args.model,
        "config": {
            "num_requests": args.num_requests,
            "prompt_lengths": args.prompt_lengths,
            "request_prompt_lengths": prompt_lengths,
            "output_tokens": args.output_tokens,
            "max_num_seqs": args.num_requests,
            "max_num_decode_seqs": args.max_num_decode_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": enforce_eager,
            "execution_mode": execution_mode,
            "temperature": 1.0,
            "seed": args.seed,
            "max_num_prefill_seqs": args.max_num_prefill_seqs,
            "enable_prefill_aging": args.enable_prefill_aging,
            "prefill_wait_metric_definition": (
                "Engine steps from request arrival to first scheduler "
                "selection in a Prefill batch; queue wait time is measured "
                "at schedule() return, before model execution."
            ),
            "prefill_total_wait_definition": (
                "For every Engine step while a request still has uncached "
                "Prompt tokens, count one wait step if it receives no Prefill "
                "allocation in that step. This includes initial queueing and "
                "between-chunk waits; it excludes a step in which it receives "
                "any Prefill allocation."
            ),
            "prefill_completion_definition": (
                "The first Engine step after which num_cached_tokens reaches "
                "num_prompt_tokens. Completion time is measured at the end "
                "of that Engine step, after CUDA synchronization."
            ),
            "stage_timing_definition": (
                "scheduler_schedule_time_ms is CPU wall time spent in "
                "Scheduler.schedule(). decode_gpu_time_ms and "
                "prefill_gpu_time_ms are CUDA-event elapsed times around "
                "ModelRunner run_decode/run_prefill calls; they are not "
                "per-request timings and exclude CPU-side preparation."
            ),
            "staggered_initial_requests": (
                args.staggered_initial_requests
                if args.scenario == "staggered"
                else None
            ),
            "staggered_batch_size": (
                args.staggered_batch_size
                if args.scenario == "staggered"
                else None
            ),
            "staggered_interval_steps": (
                args.staggered_interval_steps
                if args.scenario == "staggered"
                else None
            ),
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
            "prefill_wait": summarize_wait_metrics(request_results),
            "prefill_wait_by_prompt_length": wait_by_prompt_length,
            "prefill_fairness": summarize_prefill_fairness(request_results),
            "prefill_fairness_by_prompt_length": {
                str(prompt_length): summarize_prefill_fairness([
                    item for item in request_results
                    if item["prompt_tokens"] == prompt_length
                ])
                for prompt_length in sorted(set(prompt_lengths))
            },
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
        "scheduler_steps": scheduler_step_records,
        "model_runner_stats": {
            **{
                key: (dict(value) if hasattr(value, "items") else value)
                for key, value in engine.model_runner.execution_stats.items()
            },
            "decode_graph_hit_rate": (
                engine.model_runner.execution_stats["decode_graph_calls"]
                / engine.model_runner.execution_stats["decode_calls"]
                if engine.model_runner.execution_stats["decode_calls"] else None
            ),
        },
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
