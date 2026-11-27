# Chunked Prefill Benchmark Report

- Validated JSON files: **10**
- Decode caps: **8, 16**
- Input directory: outputs/benchmark/chunkprefill/decode_cap_sweep
- Run-level values are mean ± sample SD across runs.
- Request-level fairness values pool requests across runs; percentiles use nearest-rank.

## Overall performance

| Metric | cap 8 | cap 16 |
|---|---:|---:|
| Total time (ms) | 2903.02 ± 20.79 | 2909.44 ± 25.69 |
| Output throughput (tokens/s) | 88.19 ± 0.63 | 88.00 ± 0.77 |
| Mean TTFT (ms) | 962.10 ± 5.04 | 966.70 ± 4.21 |
| Mean TPOT (ms) | 60.48 ± 0.33 | 60.78 ± 0.78 |
| Mean request latency (ms) | 1385.48 ± 6.74 | 1392.16 ± 9.63 |
| Engine Steps | 44.00 ± 0.00 | 42.00 ± 0.00 |
| Aggregate token budget utilization | 0.78 ± 0.00 | 0.81 ± 0.00 |

## Paired results

Delta is cap16 minus cap8; positive latency/time deltas are worse, positive throughput deltas are better.

| Seed | Metric | cap 8 | cap 16 | Delta | Delta % |
|---:|---|---:|---:|---:|---:|
| 42 | total_time_ms | 2874.46 | 2928.36 | 53.90 | 1.88% |
| 42 | output_throughput | 89.06 | 87.42 | -1.64 | -1.84% |
| 42 | mean_ttft_ms | 955.30 | 966.15 | 10.85 | 1.14% |
| 42 | mean_tpot_ms | 59.96 | 60.69 | 0.73 | 1.22% |
| 42 | mean_request_latency_ms | 1375.01 | 1390.98 | 15.97 | 1.16% |
| 42 | total_steps | 44.00 | 42.00 | -2.00 | -4.55% |
| 43 | total_time_ms | 2892.19 | 2944.73 | 52.54 | 1.82% |
| 43 | output_throughput | 88.51 | 86.94 | -1.58 | -1.78% |
| 43 | mean_ttft_ms | 959.16 | 973.58 | 14.42 | 1.50% |
| 43 | mean_tpot_ms | 60.78 | 62.14 | 1.36 | 2.23% |
| 43 | mean_request_latency_ms | 1384.62 | 1408.54 | 23.92 | 1.73% |
| 43 | total_steps | 44.00 | 42.00 | -2.00 | -4.55% |
| 44 | total_time_ms | 2921.20 | 2897.29 | -23.91 | -0.82% |
| 44 | output_throughput | 87.64 | 88.36 | 0.72 | 0.83% |
| 44 | mean_ttft_ms | 964.87 | 966.67 | 1.80 | 0.19% |
| 44 | mean_tpot_ms | 60.55 | 60.54 | -0.01 | -0.02% |
| 44 | mean_request_latency_ms | 1388.73 | 1390.44 | 1.70 | 0.12% |
| 44 | total_steps | 44.00 | 42.00 | -2.00 | -4.55% |
| 45 | total_time_ms | 2924.60 | 2887.21 | -37.38 | -1.28% |
| 45 | output_throughput | 87.53 | 88.67 | 1.13 | 1.29% |
| 45 | mean_ttft_ms | 968.28 | 962.24 | -6.04 | -0.62% |
| 45 | mean_tpot_ms | 60.71 | 60.19 | -0.52 | -0.86% |
| 45 | mean_request_latency_ms | 1393.25 | 1383.57 | -9.68 | -0.69% |
| 45 | total_steps | 44.00 | 42.00 | -2.00 | -4.55% |
| 46 | total_time_ms | 2902.67 | 2889.58 | -13.09 | -0.45% |
| 46 | output_throughput | 88.19 | 88.59 | 0.40 | 0.45% |
| 46 | mean_ttft_ms | 962.91 | 964.87 | 1.96 | 0.20% |
| 46 | mean_tpot_ms | 60.41 | 60.34 | -0.07 | -0.12% |
| 46 | mean_request_latency_ms | 1385.79 | 1387.25 | 1.46 | 0.11% |
| 46 | total_steps | 44.00 | 42.00 | -2.00 | -4.55% |

## Prefill fairness by prompt length (pooled requests)

| Cap | Prompt | Total wait mean / P95 / max (steps) | Max consecutive wait mean / max | Chunks mean | Prefill completion latency mean / P95 (ms) |
|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 1.45 / 3.00 / 3.00 | 0.82 / 2.00 | 2.09 | 297.02 / 419.51 |
| 8 | 512 | 5.64 / 8.00 / 8.00 | 1.45 / 2.00 | 7.64 | 956.42 / 1231.76 |
| 8 | 1024 | 8.90 / 11.00 / 11.00 | 1.90 / 2.00 | 14.40 | 1699.94 / 1946.91 |
| 16 | 128 | 1.45 / 3.00 / 3.00 | 0.82 / 2.00 | 2.09 | 298.65 / 420.04 |
| 16 | 512 | 5.64 / 8.00 / 8.00 | 1.45 / 2.00 | 7.64 | 958.98 / 1232.04 |
| 16 | 1024 | 8.90 / 11.00 / 11.00 | 1.90 / 2.00 | 14.40 | 1710.05 / 1950.69 |

## Runtime diagnostics

| Metric | cap 8 | cap 16 |
|---|---:|---:|
| Decode Graph hit rate | 1.0000 | 1.0000 |
| Decode Eager calls / run | 0.0000 | 0.0000 |
| KV allocation failures / run | 0.0000 | 0.0000 |
| Decode preemptions / run | 0.0000 | 0.0000 |
| Peak allocated delta (MiB) | 15.0112 | 15.0112 |
| Peak reserved delta (MiB) | 6.0000 | 6.0000 |

## Validation notes

- All 10 JSON files were loaded; each contains 32 requests and all requests completed Prefill.
- Both step arrays match summary.total_steps; request output-token totals match summary totals.
- All Prefill chunk cached-token cursors are continuous and end at the request prompt length.
- This report is CPU-side JSON aggregation; it does not execute the model or replace GPU validation.
- Small differences from five runs must not be presented as statistically significant.

## Input files

- cap8_run1_seed42.json
- cap16_run1_seed42.json
- cap16_run2_seed43.json
- cap8_run2_seed43.json
- cap8_run3_seed44.json
- cap16_run3_seed44.json
- cap16_run4_seed45.json
- cap8_run4_seed45.json
- cap8_run5_seed46.json
- cap16_run5_seed46.json
