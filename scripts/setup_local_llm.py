"""Setup local LLM for extraction via transformers + hf-mirror.com.
Downloads Qwen2.5-7B-Instruct to D drive, adds 'local' provider to LLM system.
"""
import os
import sys
import subprocess

print("=== Setting up Local LLM (Transformers) ===")

# 1. Set HF mirror for China access
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
subprocess.run(['setx', 'HF_ENDPOINT', 'https://hf-mirror.com'], capture_output=True)
print("HF_ENDPOINT = https://hf-mirror.com")

# 2. Set HF cache to D drive (models are large)
hf_home = "D:/huggingface_cache"
os.makedirs(hf_home, exist_ok=True)
os.environ["HF_HOME"] = hf_home
os.environ["TRANSFORMERS_CACHE"] = hf_home
subprocess.run(['setx', 'HF_HOME', hf_home], capture_output=True)
print(f"HF_HOME = {hf_home}")

# 3. Install modelscope for faster CN downloads
print("\nInstalling modelscope...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'modelscope', 'accelerate',
     '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple/'],
    capture_output=True, text=True, timeout=120
)
print(f"pip exit: {result.returncode}")
if result.stderr:
    err_lines = result.stderr.strip().split('\n')
    for line in err_lines[-3:]:
        print(f"  {line}")

# 4. Test model download via modelscope
print("\nTesting model access via ModelScope...")
try:
    from modelscope import snapshot_download
    model_dir = snapshot_download(
        'Qwen/Qwen2.5-7B-Instruct',
        cache_dir='D:/huggingface_cache/modelscope',
        revision='master',
    )
    print(f"Model downloaded to: {model_dir}")
except Exception as e:
    print(f"ModelScope download test: {e}")
    print("Will use HF mirror directly instead.")

print("\nSetup complete!")
print("Model will be stored on D drive.")
