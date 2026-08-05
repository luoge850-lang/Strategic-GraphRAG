"""Setup local LLM via ModelScope (China CDN) + transformers."""
import subprocess
import sys
import os

print("Step 1: Install modelscope + accelerate...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'modelscope', 'accelerate',
     '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple/'],
    capture_output=True, text=True, timeout=120
)
print(f"Exit: {result.returncode}")
if result.stdout:
    for line in result.stdout.strip().split('\n')[-5:]:
        print(f"  {line}")

print("\nStep 2: Test ModelScope download...")
try:
    from modelscope import snapshot_download

    # Download to D drive
    cache = 'D:/modelscope_models'
    os.makedirs(cache, exist_ok=True)

    model_dir = snapshot_download(
        'Qwen/Qwen2.5-7B-Instruct',
        cache_dir=cache,
    )
    print(f"Model at: {model_dir}")

    # Test loading with transformers
    print("\nStep 3: Test loading with transformers...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    print(f"Tokenizer OK, vocab={tokenizer.vocab_size}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
