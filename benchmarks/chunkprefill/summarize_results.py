#!/usr/bin/env python3
"""CPU-only aggregator for Chunked Prefill benchmark JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
import re

NAME_RE = re.compile(r"^cap(?P<cap>\d+)_run(?P<run>\d+)_seed(?P<seed>\d+)\.json$")
METRICS = {
    "total_time_ms": ("summary.total_time_ms", "ms"),
    "output_throughput": ("summary.output_throughput", "tokens/s"),
    "mean_ttft_ms": ("summary.mean_ttft_ms", "ms"),
    "mean_tpot_ms": ("summary.mean_tpot_ms", "ms"),
    "mean_request_latency_ms": ("summary.mean_request_latency_ms", "ms"),
    "total_steps": ("summary.total_steps", "steps"),
    "prefill_first_wait_mean_steps": ("summary.prefill_wait.mean_wait_steps", "steps"),
    "prefill_first_wait_p95_steps": ("summary.prefill_wait.p95_wait_steps", "steps"),
    "prefill_total_wait_mean_steps": ("summary.prefill_fairness.mean_total_wait_steps", "steps"),
    "prefill_total_wait_p95_steps": ("summary.prefill_fairness.p95_total_wait_steps", "steps"),
    "prefill_total_wait_max_steps": ("summary.prefill_fairness.max_total_wait_steps", "steps"),
    "prefill_max_consecutive_wait_mean_steps": ("summary.prefill_fairness.mean_max_consecutive_wait_steps", "steps"),
    "prefill_max_consecutive_wait_max_steps": ("summary.prefill_fairness.max_consecutive_wait_steps", "steps"),
    "prefill_chunk_count_mean": ("summary.prefill_fairness.mean_chunk_count", "chunks/request"),
    "prefill_completion_latency_mean_ms": ("summary.prefill_fairness.mean_prefill_completion_latency_ms", "ms"),
    "prefill_completion_latency_p95_ms": ("summary.prefill_fairness.p95_prefill_completion_latency_ms", "ms"),
    "peak_allocated_delta_mib": ("summary.peak_allocated_delta_mib", "MiB"),
    "peak_reserved_delta_mib": ("summary.peak_reserved_delta_mib", "MiB"),
}
PAIRED = ("total_time_ms", "output_throughput", "mean_ttft_ms",
          "mean_tpot_ms", "mean_request_latency_ms", "total_steps")
REQUEST_FIELDS = {
    "total_wait_steps": "prefill_total_wait_steps",
    "max_consecutive_wait_steps": "prefill_max_consecutive_wait_steps",
    "chunk_count": "prefill_chunk_count",
    "prefill_completion_latency_ms": "prefill_completion_latency_ms",
    "ttft_ms": "ttft_ms",
    "tpot_ms": "tpot_ms",
    "request_latency_ms": "request_latency_ms",
}


def get_path(data, dotted):
    for part in dotted.split("."):
        data = data[part]
    return data


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def stats(values):
    values = [float(x) for x in values if x is not None]
    if not values:
        return {k: None for k in ("mean", "sd", "median", "min", "max", "p50", "p95")} | {"count": 0}
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values), "min": min(values), "max": max(values),
        "p50": percentile(values, 0.50), "p95": percentile(values, 0.95),
    }


def as_int(value):
    return int(value or 0)


def validate_result(path, data, expected):
    cap, run, seed = expected
    config, summary = data.get("config", {}), data.get("summary", {})
    requests = data.get("requests", [])
    steps, sched = data.get("steps", []), data.get("scheduler_steps", [])
    errors = []
    if as_int(config.get("max_num_decode_seqs", -1)) != cap:
        errors.append(f"{path.name}: Decode cap differs from filename")
    if as_int(config.get("seed", -1)) != seed:
        errors.append(f"{path.name}: seed differs from filename")
    if data.get("scenario") != "staggered":
        errors.append(f"{path.name}: expected staggered scenario")
    if len(requests) != as_int(config.get("num_requests", -1)):
        errors.append(f"{path.name}: request count differs from config")
    if len(requests) != as_int(summary.get("prefill_fairness", {}).get("request_count", -1)):
        errors.append(f"{path.name}: fairness request count mismatch")
    if len(steps) != as_int(summary.get("total_steps", -1)) or len(sched) != len(steps):
        errors.append(f"{path.name}: step arrays do not match summary")
    incomplete = [r.get("request_id") for r in requests if r.get("prefill_completion_step") is None]
    if incomplete:
        errors.append(f"{path.name}: incomplete Prefill requests {incomplete}")
    if sum(as_int(r.get("output_tokens")) for r in requests) != as_int(summary.get("total_output_tokens", -1)):
        errors.append(f"{path.name}: output token count mismatch")

    for request in requests:
        chunks = request.get("prefill_chunks", [])
        if not chunks:
            errors.append(f"{path.name}: request {request.get('request_id')} has no chunks")
            continue
        cursor = None
        for chunk in chunks:
            before = as_int(chunk.get("cached_tokens_before"))
            allocated = as_int(chunk.get("allocated_tokens"))
            after = as_int(chunk.get("cached_tokens_after"))
            if cursor is not None and before != cursor:
                errors.append(f"{path.name}: request {request.get('request_id')} chunk discontinuity")
                break
            if after != before + allocated:
                errors.append(f"{path.name}: request {request.get('request_id')} chunk cursor mismatch")
                break
            cursor = after
        if cursor != as_int(request.get("prompt_tokens")):
            errors.append(f"{path.name}: request {request.get('request_id')} final cursor mismatch")
        if sum(as_int(c.get("allocated_tokens")) for c in chunks) != as_int(request.get("prefill_total_allocated_tokens")):
            errors.append(f"{path.name}: request {request.get('request_id')} allocated token sum mismatch")
    runner = data.get("model_runner_stats", {})
    if as_int(runner.get("decode_graph_calls")) > as_int(runner.get("decode_calls")):
        errors.append(f"{path.name}: graph calls exceed Decode calls")
    return errors


def load_runs(input_dir, caps, expected_runs):
    found, errors = {}, []
    for path in sorted(input_dir.glob("cap*_run*_seed*.json")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        cap, run, seed = (int(match.group(k)) for k in ("cap", "run", "seed"))
        if cap not in caps:
            continue
        key = (cap, seed)
        if key in found:
            errors.append(f"duplicate cap/seed pair: {key}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: invalid JSON: {exc}")
            continue
        errors.extend(validate_result(path, data, (cap, run, seed)))
        found[key] = {"path": path, "cap": cap, "run": run, "seed": seed, "data": data}
    seed_sets = {}
    for cap in caps:
        group = [x for x in found.values() if x["cap"] == cap]
        if len(group) != expected_runs:
            errors.append(f"cap={cap}: expected {expected_runs} runs, found {len(group)}")
        seed_sets[cap] = {x["seed"] for x in group}
    if len(caps) == 2 and seed_sets[caps[0]] != seed_sets[caps[1]]:
        errors.append("Decode-cap groups do not have matching seed sets")
    if errors:
        raise ValueError("\n".join(errors))
    return sorted(found.values(), key=lambda x: (x["seed"], x["cap"]))


def make_run_rows(runs):
    rows = []
    for item in runs:
        data = item["data"]
        row = {
            "file": item["path"].name, "cap": item["cap"], "run": item["run"], "seed": item["seed"],
            "scenario": data.get("scenario"), "num_requests": data["config"].get("num_requests"),
            "prompt_lengths": ",".join(map(str, data["config"].get("prompt_lengths", []))),
            "output_tokens_per_request": data["config"].get("output_tokens"),
            "token_budget": data["config"].get("max_num_batched_tokens"),
            "prefill_cap": data["config"].get("max_num_prefill_seqs"),
        }
        for name, (path, _unit) in METRICS.items():
            row[name] = get_path(data, path)
        runner = data.get("model_runner_stats", {})
        for field in ("decode_calls", "decode_graph_calls", "decode_eager_calls",
                      "decode_graph_hit_rate", "prefill_calls"):
            row[field] = runner.get(field)
        row["kv_allocation_failures"] = sum(as_int(s.get("kv_allocation_failures")) for s in data["scheduler_steps"])
        row["decode_preemptions"] = sum(as_int(s.get("decode_preemptions")) for s in data["scheduler_steps"])
        used = sum(as_int(s.get("used_tokens")) for s in data["scheduler_steps"])
        budget = sum(as_int(s.get("token_budget")) for s in data["scheduler_steps"])
        row["aggregate_token_budget_utilization"] = used / budget if budget else None
        rows.append(row)
    return rows


def make_aggregate_rows(run_rows, caps):
    rows = []
    extra = {
        "decode_graph_hit_rate": "ratio",
        "aggregate_token_budget_utilization": "ratio",
        "kv_allocation_failures": "count", "decode_preemptions": "count",
        "decode_calls": "calls/run", "decode_eager_calls": "calls/run", "prefill_calls": "calls/run",
    }
    for cap in caps:
        group = [r for r in run_rows if r["cap"] == cap]
        metrics = {name: unit for name, (_path, unit) in METRICS.items()} | extra
        for metric, unit in metrics.items():
            rows.append({"cap": cap, "metric": metric, "unit": unit,
                         **stats([r[metric] for r in group])})
    return rows


def make_paired_rows(run_rows, caps):
    if len(caps) != 2:
        return []
    low, high = caps
    indexed = {(r["cap"], r["seed"]): r for r in run_rows}
    rows = []
    for seed in sorted({r["seed"] for r in run_rows}):
        a, b = indexed[(low, seed)], indexed[(high, seed)]
        for metric in PAIRED:
            x, y = float(a[metric]), float(b[metric])
            rows.append({"seed": seed, "metric": metric, f"cap{low}": x, f"cap{high}": y,
                         "delta_high_minus_low": y - x,
                         "delta_pct": (y - x) / x * 100 if x else None})
    return rows


def make_prompt_rows(runs):
    buckets = defaultdict(lambda: defaultdict(list))
    for item in runs:
        for request in item["data"]["requests"]:
            prompt = int(request["prompt_tokens"])
            for metric, field in REQUEST_FIELDS.items():
                if request.get(field) is not None:
                    buckets[(item["cap"], prompt)][metric].append(float(request[field]))
    rows = []
    for (cap, prompt), metrics in sorted(buckets.items()):
        row = {"cap": cap, "prompt_tokens": prompt}
        for metric, values in metrics.items():
            for key, value in stats(values).items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def make_report(runs, run_rows, aggregate_rows, paired_rows, prompt_rows, caps):
    agg = {(r["cap"], r["metric"]): r for r in aggregate_rows}
    lines = [
        "# Chunked Prefill Benchmark Report", "",
        f"- Validated JSON files: **{len(runs)}**",
        f"- Decode caps: **{', '.join(map(str, caps))}**",
        f"- Input directory: {runs[0]['path'].parent}",
        "- Run-level values are mean ± sample SD across runs.",
        "- Request-level fairness values pool requests across runs; percentiles use nearest-rank.",
        "", "## Overall performance", "",
        "| Metric | " + " | ".join(f"cap {c}" for c in caps) + " |",
        "|---|" + "---:|" * len(caps),
    ]
    for metric, label in (
        ("total_time_ms", "Total time (ms)"), ("output_throughput", "Output throughput (tokens/s)"),
        ("mean_ttft_ms", "Mean TTFT (ms)"), ("mean_tpot_ms", "Mean TPOT (ms)"),
        ("mean_request_latency_ms", "Mean request latency (ms)"), ("total_steps", "Engine Steps"),
        ("aggregate_token_budget_utilization", "Aggregate token budget utilization"),
    ):
        vals = [f"{fmt(agg[(c, metric)]['mean'])} ± {fmt(agg[(c, metric)]['sd'])}" for c in caps]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", "## Paired results", "",
              "Delta is higher cap minus lower cap; positive latency/time deltas are worse, positive throughput deltas are better.", ""]
    if paired_rows:
        lines += ["| Seed | Metric | " + " | ".join(f"cap {c}" for c in caps) + " | Delta | Delta % |",
                  "|---:|---|" + "---:|" * (len(caps) + 2)]
        for row in paired_rows:
            lines.append(f"| {row['seed']} | {row['metric']} | " +
                         " | ".join(fmt(row[f"cap{c}"]) for c in caps) +
                         f" | {fmt(row['delta_high_minus_low'])} | {fmt(row['delta_pct'])}% |")
    lines += ["", "## Prefill fairness by prompt length (pooled requests)", "",
              "| Cap | Prompt | Total wait mean / P95 / max (steps) | Max consecutive wait mean / max | Chunks mean | Prefill completion latency mean / P95 (ms) |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in prompt_rows:
        lines.append(
            f"| {row['cap']} | {row['prompt_tokens']} | "
            f"{fmt(row.get('total_wait_steps_mean'))} / {fmt(row.get('total_wait_steps_p95'))} / {fmt(row.get('total_wait_steps_max'))} | "
            f"{fmt(row.get('max_consecutive_wait_steps_mean'))} / {fmt(row.get('max_consecutive_wait_steps_max'))} | "
            f"{fmt(row.get('chunk_count_mean'))} | "
            f"{fmt(row.get('prefill_completion_latency_ms_mean'))} / {fmt(row.get('prefill_completion_latency_ms_p95'))} |"
        )
    lines += ["", "## Runtime diagnostics", "",
              "| Metric | " + " | ".join(f"cap {c}" for c in caps) + " |",
              "|---|" + "---:|" * len(caps)]
    for metric, label in (
        ("decode_graph_hit_rate", "Decode Graph hit rate"),
        ("decode_eager_calls", "Decode Eager calls / run"),
        ("kv_allocation_failures", "KV allocation failures / run"),
        ("decode_preemptions", "Decode preemptions / run"),
        ("peak_allocated_delta_mib", "Peak allocated delta (MiB)"),
        ("peak_reserved_delta_mib", "Peak reserved delta (MiB)"),
    ):
        lines.append(f"| {label} | " + " | ".join(
            fmt(agg[(c, metric)]["mean"], 4 if "rate" in metric else 2) for c in caps
        ) + " |")
    lines += ["", "## Validation notes", "",
              "- Every input passed filename/config, request completion, output-token, step-count, and Prefill cached-token continuity checks.",
              "- This is CPU-side JSON aggregation. It does not run the model or replace GPU validation.",
              "- Small differences from a limited number of runs must not be presented as statistically significant.",
              "", "## Input files", ""]
    lines.extend(f"- {item['path'].name}" for item in runs)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/benchmark/chunkprefill/decode_cap_sweep"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--caps", default="8,16", help="comma-separated Decode caps")
    parser.add_argument("--expected-runs", type=int, default=5)
    args = parser.parse_args()
    caps = [int(x) for x in args.caps.split(",") if x.strip()]
    if not caps or len(set(caps)) != len(caps):
        parser.error("--caps must contain unique integer values")
    output_dir = args.output_dir or args.input_dir / "analysis"
    try:
        runs = load_runs(args.input_dir, caps, args.expected_runs)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    run_rows = make_run_rows(runs)
    aggregate_rows = make_aggregate_rows(run_rows, caps)
    paired_rows = make_paired_rows(run_rows, caps)
    prompt_rows = make_prompt_rows(runs)
    report = make_report(runs, run_rows, aggregate_rows, paired_rows, prompt_rows, caps)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "run_summary.csv", run_rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(output_dir / "paired_comparison.csv", paired_rows)
    write_csv(output_dir / "prompt_fairness.csv", prompt_rows)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Validated {len(runs)} JSON files; reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
