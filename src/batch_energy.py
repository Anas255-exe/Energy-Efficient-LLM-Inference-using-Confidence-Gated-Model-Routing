"""RQ4-lite: measured J/query at batch size 1 vs B for one model (greedy, same questions).

usage: python src/batch_energy.py --model 3b --n 64 --batches 1 8
"""
import argparse, json, time
from mlx_lm import load, batch_generate
from energy import PowerMeter
from run import MODELS, SYSTEM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args()

qs = [json.loads(l) for l in open("data/questions.jsonl")][-a.n:]  # test-end questions
model, tok = load(MODELS[a.model])
prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question"]}],
                                   add_generation_prompt=True) for q in qs]
meter = PowerMeter()
batch_generate(model, tok, prompts[:8], max_tokens=8)  # warm-up
idle_soc, idle_sys = meter.idle_power(15)
out = []
for rep in range(a.repeats):
    for B in a.batches:
        t0 = time.time()
        for i in range(0, len(prompts), B):
            batch_generate(model, tok, prompts[i:i + B], max_tokens=24)
        t1 = time.time()
        soc, sys_ = meter.energy(t0, t1)
        r = dict(model=a.model, batch=B, rep=rep, n=len(prompts), seconds=t1 - t0,
                 J_per_query_sys_net=(sys_ - idle_sys * (t1 - t0)) / len(prompts),
                 J_per_query_soc_net=(soc - idle_soc * (t1 - t0)) / len(prompts))
        out.append(r); print(r, flush=True)
meter.close()
json.dump(out, open(f"results/batch_{a.model}.json", "w"), indent=2)
