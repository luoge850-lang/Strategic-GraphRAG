"""Test local transformers LLM for extraction."""
import os
import sys
import time

# Use HF mirror for China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:/huggingface_cache"

print("Testing transformers pipeline for extraction...")
print(f"Model cache dir: D:/huggingface_cache")

# Test with a small model first
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Use Qwen2.5-3B-Instruct for fast test
model_id = "Qwen/Qwen2.5-3B-Instruct"

print(f"\nLoading {model_id}...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True,
)

print(f"Model loaded in {time.time() - t0:.1f}s")

# Test extraction
test_text = """
NVIDIA faces significant risks from US export controls on advanced semiconductors to China.
These restrictions may decrease our data center revenue in the China market by limiting our
ability to sell H100 and A100 GPUs. To mitigate this risk, we have developed compliant
products and are diversifying our supply chain.
"""

prompt = f"""Extract financial causal triples from this SEC filing text.
Return ONLY a JSON array of [source, relation, target, evidence_sentence] objects.

Text: {test_text}

JSON:"""

t1 = time.time()
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.1)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nInference time: {time.time() - t1:.1f}s")
print(f"\nResult:\n{result}")
