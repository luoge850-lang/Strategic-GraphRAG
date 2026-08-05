"""Download Ollama from Tsinghua mirror and install to D drive."""
import urllib.request
import os
import subprocess
import sys

# Tsinghua mirror URL for Ollama
# The latest version path format
url = "https://mirrors.tuna.tsinghua.edu.cn/github-release/ollama/ollama/latest/OllamaSetup.exe"
dest = "D:/OllamaSetup.exe"

print(f"Downloading Ollama from Tsinghua mirror...")
print(f"URL: {url}")

try:
    # Use a proper user-agent
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as f:
        data = f.read()
    with open(dest, 'wb') as f:
        f.write(data)
    size_mb = len(data) / (1024*1024)
    print(f"Downloaded: {size_mb:.1f} MB to {dest}")

    # Set model storage to D drive
    os.makedirs("D:/ollama_models", exist_ok=True)
    subprocess.run(['setx', 'OLLAMA_MODELS', 'D:\\ollama_models'], capture_output=True)
    os.environ['OLLAMA_MODELS'] = 'D:/ollama_models'
    print("OLLAMA_MODELS=D:\\ollama_models")

    # Install silently
    print("Installing Ollama...")
    result = subprocess.run([dest, '/S'], capture_output=True, text=True, timeout=120)
    print(f"Install completed with code {result.returncode}")

except Exception as e:
    print(f"Error: {e}")
    # Try winget as fallback
    print("Trying winget...")
    result = subprocess.run(['winget', 'install', 'Ollama.Ollama', '--silent'],
                          capture_output=True, text=True, timeout=120)
    print(f"winget result: {result.returncode}")
    print(result.stdout[:500] if result.stdout else "")
