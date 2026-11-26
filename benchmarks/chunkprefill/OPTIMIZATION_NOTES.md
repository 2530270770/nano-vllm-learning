# Chunked Prefill Scheduling Diagnostics

This benchmark exports scheduler and ModelRunner diagnostics in addition to end-to-end latency.

## Decode concurrency sweep

Use the same workload and seed, changing only `--max-num-decode-seqs`:

```bash
for decode_limit in 4 8 16; do
  python benchmarks/chunkprefill/benchmark.py \
    --variant optimized \
    --scenario staggered \
    --num-requests 32 \
    --prompt-lengths 128,512,1024 \
    --max-num-decode-seqs ${decode_limit} \
    --max-num-prefill-seqs 8 \
    --output-tokens 8 \
    --execution-mode cudagraph \
    --staggered-initial-requests 4 \
    --staggered-batch-size 4 \
    --staggered-interval-steps 2 \
    --seed 42 \
    --output outputs/benchmark/chunkprefill/decode_sweep/decode_${decode_limit}.json
done
```

Repeat each setting at least five times and compare paired medians/P95s. A larger Decode cap is not automatically better: it can improve Decode service while consuming more shared token budget and delaying Prefill.

## Scheduler diagnostics

Each `scheduler_steps` record includes:

- Decode limit/count and Prefill concurrency limit.
- Candidate request IDs and remaining Prefill demand.
- Integer water level and per-request Token allocations.
- Used/unused Token Budget and budget utilization.
- KV allocation failures, free/used KV blocks, and Decode preemptions.
- Queue sizes at the beginning/end of the scheduling step.

`prefill_total_wait_steps` on each Sequence is a monotonic cumulative queue-wait counter. Aging sorts by this cumulative value, so completing one Prefill chunk does not erase the wait accumulated before that chunk. With Aging disabled, FIFO ordering remains unchanged; the cumulative metric is still collected.

## ModelRunner diagnostics

The top-level `model_runner_stats` record includes Decode call count, CUDA Graph call count, Eager fallback count/reasons, Decode batch-size distribution, selected Graph-size distribution, Prefill call count/token count, and Graph hit rate.

These are call-count diagnostics, not GPU kernel timings. Do not interpret them as CUDA execution time. Use CUDA events or Nsight Systems for timing attribution.

## Interpretation

- Low budget utilization with non-empty Prefill demand suggests candidate/KV constraints or scheduling limits; inspect the per-step failure and allocation fields before changing Water-Filling.
- Repeated KV allocation failures plus low free-block counts indicate cache pressure; compare request lengths and preemption counts.
- A low Graph hit rate should be explained by Eager mode or oversized batches before changing graph capture.
- Keep Prefill Aging optional unless repeated A/B runs show a fairness benefit without unacceptable TTFT/throughput regression.
