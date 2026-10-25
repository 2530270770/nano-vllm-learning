import sys
sys.path.insert(0, "/home/chen17/nano-vllm")

from nanovllm import LLM, SamplingParams

llm = LLM(
    "/home/chen17/huggingface/Qwen3-0.6B",
    enforce_eager=True,
    tensor_parallel_size=1,
    max_num_seqs=1,
)
params = SamplingParams(temperature=0.6, max_tokens=5)
outputs = llm.generate([
    "台湾是"
], params)
print(outputs[0]["text"])