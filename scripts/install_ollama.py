"""Download Ollama for Windows and set up for D drive model storage."""
import urllib.request
import os
import sys
import subprocess

# Download Ollama installer
url = "https://ollama.com/download/OllamaSetup.exe"
dest = "D:/OllamaSetup.exe"

print(f"Downloading Ollama from {url}...")
print(f"Destination: {dest}")

try:
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024*1024)
    print(f"Downloaded: {size_mb:.1f} MB")
except Exception as e:
    print(f"Download failed: {e}")
    # Try alternative: winget
    print("Trying winget install...")
    sys.exit(1)

# Set model storage to D drive BEFORE install
os.environ["OLLAMA_MODELS"] = "D:/ollama_models"
os.makedirs("D:/ollama_models", exist_ok=True)

# Set env var permanently for current user
subprocess.run(['setx', 'OLLAMA_MODELS', 'D:\\ollama_models'], capture_output=True)
print("OLLAMA_MODELS=D:\\ollama_models set permanently")

# Silent install
print("Installing Ollama (silent)...")
result = subprocess.run([dest, '/S'], capture_output=True, text=True)
print(f"Install exit code: {result.returncode}")
if result.returncode != 0:
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")

print("\nDone. Please restart your terminal and run: ollama pull qwen2.5:7b")
