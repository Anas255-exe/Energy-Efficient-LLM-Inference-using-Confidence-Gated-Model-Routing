"""Cost basis for the cascade (fixes measurement drift): both models resident, and per test question
3B answer, 7B answer and 3B prefill-only (max_tokens=1) are run in random order, so thermal and
battery state are shared across policies. Uses stock mlx-lm stream_generate (no probe instrumentation).
Energy = SoC + DRAM (macmon). Idle measured before and after; gross and net both saved.

usage: python src/cost.py [--n 300]
"""
import argparse, json, random, time
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from energy import PowerMeter
from run import MODELS, SYSTEM

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()

qs = [json.loads(l) for l in open("data/questions.jsonl")][-a.n:]  # = analyze.py test split (last 30%)
M = {k: load(MODELS[k]) for k in ("3b", "7b")}
greedy = make_sampler(temp=0.0)
jobs = {"3b": ("3b", 24), "7b": ("7b", 24), "3b_prefill": ("3b", 1)}


def run(job, q):
    m, mt = jobs[job]
    model, tok = M[m]
    p = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question"]}],
                                add_generation_prompt=True)
    text = "".join(r.text for r in stream_generate(model, tok, p, max_tokens=mt, sampler=greedy))
    return text


meter = PowerMeter()
for q in qs[:3]:
    for j in jobs:
        run(j, q)
idle0 = meter.idle_power(20)
rows, rnd = [], random.Random(0)
for i, q in enumerate(qs):
    for j in rnd.sample(list(jobs), len(jobs)):
        t0 = time.time(); ans = run(j, q); t1 = time.time()
        soc, sys_ = meter.energy(t0, t1)
        rows.append(dict(id=q["id"], job=j, seconds=t1 - t0, J=soc, J_sys=sys_, temp=meter.temp(), answer=ans.strip()))
    if i % 50 == 0:
        print(i, {j: round(np.mean([r["J"] for r in rows if r["job"] == j]), 2) for j in jobs}, "temp", round(meter.temp(), 1), flush=True)
idle1 = meter.idle_power(20)
meter.close()
idle = (idle0[0] + idle1[0]) / 2
for r in rows:
    r["J_net"] = r["J"] - idle * r["seconds"]
with open("results/cost.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
summ = {j: dict(J=float(np.mean([r["J"] for r in rows if r["job"] == j])),
               J_net=float(np.mean([r["J_net"] for r in rows if r["job"] == j])),
               s=float(np.mean([r["seconds"] for r in rows if r["job"] == j]))) for j in jobs}
summ["idle_before_W"], summ["idle_after_W"] = idle0[0], idle1[0]
json.dump(summ, open("results/cost_summary.json", "w"), indent=2)
print(json.dumps(summ, indent=2))
