"""Robust model fetch for a flaky/throttled network: small files via hf_hub (retries),
weights via aria2c with 16 connections against the resolved CDN URL (resumable)."""
import os, subprocess, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
import truststore; truststore.inject_into_ssl()
import requests
from huggingface_hub import HfApi, hf_hub_download

REPOS = {"q05b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "q15b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit", "q3b": "mlx-community/Qwen2.5-3B-Instruct-4bit", "q7b": "mlx-community/Qwen2.5-7B-Instruct-4bit"}


def retry(fn, n=20):
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            print("retry", i, repr(e)[:120], flush=True); time.sleep(5)
    raise SystemExit("gave up")


for d, repo in REPOS.items():
    if len(sys.argv) > 1 and d not in sys.argv[1:]:
        continue
    os.makedirs(f"models/{d}", exist_ok=True)
    files = retry(lambda: [s.rfilename for s in HfApi().model_info(repo).siblings])
    for f in files:
        if f.endswith(".safetensors"):
            if os.path.exists(f"models/{d}/{f}") and not os.path.exists(f"models/{d}/{f}.aria2"):
                continue
            def cdn():
                r = requests.head(f"https://huggingface.co/{repo}/resolve/main/{f}", allow_redirects=False, timeout=60)
                return r.headers["location"]
            url = retry(cdn)
            while subprocess.call(["aria2c", "-x16", "-s16", "-k4M", "-c", "--file-allocation=none", "--summary-interval=30",
                                   "--console-log-level=warn", "-d", f"models/{d}", "-o", f, url]) != 0:
                url = retry(cdn)
        else:
            retry(lambda: hf_hub_download(repo, f, local_dir=f"models/{d}"))
        print("ok", d, f, flush=True)
print("ALL_DONE", flush=True)
