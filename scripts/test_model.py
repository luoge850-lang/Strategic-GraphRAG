"""Quick test: load Qwen2.5-7B and run one extraction."""
import os, sys, time
os.environ['HF_HOME'] = 'D:/huggingface_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = 'D:/modelscope_models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master'

print('Loading tokenizer...')
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f'Tokenizer OK ({time.time()-t0:.1f}s), vocab={tokenizer.vocab_size}')

print('Loading model...')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
params = sum(p.numel() for p in model.parameters()) / 1e9
print(f'Model loaded ({time.time()-t0:.1f}s), {params:.1f}B params')

print('Testing extraction...')
t0 = time.time()
prompt = """Extract financial causal triples from this SEC text as JSON.

Text: NVIDIA faces significant supply chain disruption risk that may decrease its data center revenue.

Return: [{"source": "...", "target": "...", "relation": "...", "evidence": "..."}]
"""
inputs = tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f'Inference ({time.time()-t0:.1f}s)')
print(f'Result:\n{result}')
