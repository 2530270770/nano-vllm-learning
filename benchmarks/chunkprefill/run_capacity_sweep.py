#!/usr/bin/env python3
"""Run a conservative, sequential Chunked-Prefill concurrency sweep.

This is an orchestration helper only. Each child process runs the existing GPU
benchmark; the sweep stops at the first failed run so a capacity/OOM boundary
does not trigger a larger workload automatically.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive integers.")
    if values != sorted(set(values)):
        raise argparse.ArgumentTypeError("Values must be strictly increasing and unique.")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-counts", type=parse_int_list, default=[32, 40, 48])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--model", default="/home/chen17/huggingface/Qwen3-0.6B")
    parser.add_argument("--variant", choices=["baseline", "optimized"], default="optimized")
    parser.add_argument("--decode-cap", type=int, default=8)
    parser.add_argument("--prefill-cap", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--prompt-lengths", default="128,512,1024")
    parser.add_argument("--initial-requests", type=int, default=4)
    parser.add_argument("--arrival-batch-size", type=int, default=4)
    parser.add_argument("--arrival-interval-steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/benchmark/chunkprefill/capacity_sweep"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands and write no GPU results.")
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.decode_cap <= 0 or args.prefill_cap <= 0:
        parser.error("--repeats and both caps must be positive")
    if args.output_tokens <= 0:
        parser.error("--output-tokens must be positive")
    if not 1 <= args.initial_requests <= max(args.request_counts):
        parser.error("--initial-requests must be between 1 and the largest request count")
    if args.arrival_batch_size <= 0 or args.arrival_interval_steps <= 0:
        parser.error("arrival batch size and interval must be positive")
    return args


def build_command(args, request_count: int, run_index: int, seed: int, output: Path):
    benchmark = Path(__file__).with_name("benchmark.py")
    return [
        sys.executable, str(benchmark),
        "--variant", args.variant,
        "--scenario", "staggered",
        "--model", args.model,
        "--num-requests", str(request_count),
        "--staggered-initial-requests", str(min(args.initial_requests, request_count)),
        "--staggered-batch-size", str(args.arrival_batch_size),
        "--staggered-interval-steps", str(args.arrival_interval_steps),
        "--prompt-lengths", args.prompt_lengths,
        "--max-num-decode-seqs", str(args.decode_cap),
        "--max-num-prefill-seqs", str(args.prefill_cap),
        "--output-tokens", str(args.output_tokens),
        "--seed", str(seed),
        "--execution-mode", "cudagraph",
        "--output", str(output),
    ]


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "capacity_sweep_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "variant": args.variant,
        "scenario": "staggered",
        "prompt_lengths": args.prompt_lengths,
        "output_tokens": args.output_tokens,
        "decode_cap": args.decode_cap,
        "prefill_cap": args.prefill_cap,
        "request_counts": args.request_counts,
        "repeats": args.repeats,
        "runs": [],
        "stopped_early": False,
    }

    for request_count in args.request_counts:
        for run_index in range(1, args.repeats + 1):
            seed = args.seed_base + run_index - 1
            output = args.output_dir / (
                f"requests{request_count}_run{run_index}_seed{seed}.json"
            )
            if output.exists() and not args.dry_run:
                print(f"Refusing to overwrite existing result: {output}", file=sys.stderr)
                manifest["stopped_early"] = True
                manifest["stop_reason"] = "output_exists"
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                return 2

            command = build_command(args, request_count, run_index, seed, output)
            print("\n" + "=" * 72)
            print(f"Capacity run: requests={request_count}, run={run_index}, seed={seed}")
            print("Command:", " ".join(command), flush=True)
            record = {
                "request_count": request_count,
                "run": run_index,
                "seed": seed,
                "output": str(output),
                "command": command,
            }
            if args.dry_run:
                record["status"] = "dry_run"
                manifest["runs"].append(record)
                continue

            completed = subprocess.run(command, check=False)
            record["return_code"] = completed.returncode
            record["status"] = "passed" if completed.returncode == 0 else "failed"
            manifest["runs"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if completed.returncode != 0:
                manifest["stopped_early"] = True
                manifest["stop_reason"] = (
                    f"benchmark_failed_requests_{request_count}_run_{run_index}"
                )
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(
                    f"Stopping sweep after failed run (exit={completed.returncode}). "
                    f"Inspect GPU state and logs before attempting a larger workload.",
                    file=sys.stderr,
                )
                return completed.returncode or 1

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSweep orchestration complete. Manifest: {manifest_path}")
    if args.dry_run:
        print("Dry run only: no model was loaded and no GPU benchmark was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
