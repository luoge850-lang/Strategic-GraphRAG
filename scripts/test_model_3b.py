"""Quick test: load Qwen2.5-3B and run extraction."""
import os, time
os.environ['HF_HOME'] = 'D:/huggingface_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = 'D:/modelscope_models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master'

print('Loading tokenizer...')
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f'  OK ({time.time()-t0:.1f}s)')

print('Loading model...')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
params = sum(p.numel() for p in model.parameters()) / 1e9
print(f'  OK ({time.time()-t0:.1f}s), {params:.1f}B params')

print('Testing extraction inference...')
t0 = time.time()
prompt = """<|im_start|>system
Extract financial causal triples from SEC text as JSON array.
Return ONLY: [{"source": "...", "target": "...", "relation": "...", "evidence": "..."}]
<|im_end|>
<|im_start|>user
Text: NVIDIA faces US export control restrictions that may decrease its China data center revenue.
<|im_end|>
<|im_start|>assistant
"""
inputs = tokenizer(prompt, return_tensors='pt')
import torch
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f'  Inference ({time.time()-t0:.1f}s)')
print(f'  Result:\n  {result}')

print('\nModel ready for extraction!')
