import sys
sys.path.insert(0, "/home/chen17/nano-vllm")

from nanovllm import LLM, SamplingParams

llm = LLM(
    "/home/chen17/huggingface/Qwen3-0.6B",
    trace_path="outputs/scheduler_trace.jsonl",
    enforce_eager=True,
    tensor_parallel_size=1,
    max_num_seqs=1,
)
params = SamplingParams(temperature=0.6, max_tokens=5)

seqs = [
    "Hello, i am God.",
    "Explain CUDA in simple terms.",
    "What is a KV cache?",
    "Explain continuous batching."
]
outputs = llm.generate(seqs, params)

seqs_len = len(seqs)
for i in range(0,seqs_len):
    print(outputs[i]["text"])