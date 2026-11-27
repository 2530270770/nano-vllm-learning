#!/usr/bin/env bash
set -euo pipefail

# Paired Decode-cap sweep for multi-request Chunked Prefill.
# Run from the repository root inside the nanovllm conda environment.

MODEL_PATH="${NANOVLLM_MODEL_PATH:-/home/chen17/huggingface/Qwen3-0.6B}"
OUTPUT_DIR="outputs/benchmark/chunkprefill/decode_cap_sweep"
REPEATS=5
BASE_SEED=42

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model directory not found: $MODEL_PATH" >&2
    echo "Set NANOVLLM_MODEL_PATH to your local Qwen3-0.6B directory." >&2
    exit 1
fi

python -m py_compile benchmarks/chunkprefill/benchmark.py

# Avoid silently overwriting any previous measurements.
for repeat in $(seq 1 "$REPEATS"); do
    seed=$((BASE_SEED + repeat - 1))
    for cap in 8 16; do
        output="$OUTPUT_DIR/cap${cap}_run${repeat}_seed${seed}.json"
        if [[ -e "$output" ]]; then
            echo "Refusing to overwrite existing result: $output" >&2
            exit 1
        fi
    done
done

echo "Starting paired Decode-cap sweep"
echo "Model: $MODEL_PATH"
echo "Repeats: $REPEATS"
echo "Seeds: $BASE_SEED-$((BASE_SEED + REPEATS - 1))"
echo "Output: $OUTPUT_DIR"
echo "Each run: 32 requests, prompts 128/512/1024, 8 output tokens"
echo "Scenario: staggered (4 initial, then 4 every 2 Engine steps)"
echo "Execution: CUDA Graph; Prefill cap: 8; Token Budget: 512"
echo

for repeat in $(seq 1 "$REPEATS"); do
    seed=$((BASE_SEED + repeat - 1))

    # Alternate order between repeats to reduce systematic warm/thermal bias.
    if (( repeat % 2 == 1 )); then
        caps=(8 16)
    else
        caps=(16 8)
    fi

    for cap in "${caps[@]}"; do
        output="$OUTPUT_DIR/cap${cap}_run${repeat}_seed${seed}.json"

        echo
        echo "============================================================"
        echo "Run $repeat/$REPEATS | Decode cap=$cap | seed=$seed"
        echo "Output: $output"
        echo "============================================================"

        python benchmarks/chunkprefill/benchmark.py \
            --variant optimized \
            --scenario staggered \
            --num-requests 32 \
            --prompt-lengths 128,512,1024 \
            --output-tokens 8 \
            --max-num-decode-seqs "$cap" \
            --max-num-prefill-seqs 8 \
            --execution-mode cudagraph \
            --seed "$seed" \
            --staggered-initial-requests 4 \
            --staggered-batch-size 4 \
            --staggered-interval-steps 2 \
            --output "$output"

        test -s "$output"
    done
done

echo
echo "All 10 runs completed."
echo "Results saved under: $OUTPUT_DIR"
echo "Do not compare only a single run; aggregate all five paired runs per cap."
