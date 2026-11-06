"""Draft prefill-only cost vs full draft answer, interleaved per question so thermal state is shared.
A pre-generation router must pay the prefill to read hidden states; a cascade pays the full answer."""
import json, random, time
import numpy as np
from mlx_lm import load
from energy import PowerMeter
from run import MODELS, SYSTEM, generate

qs = [json.loads(l) for l in open("data/questions.jsonl")][-300:]  # test split
model, tok = load(MODELS["3b"])
ps = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question"]}],
                              add_generation_prompt=True) for q in qs]
meter = PowerMeter()
for p in ps[:3]:
    generate(model, tok, p, 8, True)
idle_soc, _ = meter.idle_power(15)
res = {0: [], 24: []}
rnd = random.Random(0)
for p in ps:
    for mt in rnd.sample([0, 24], 2):
        t0 = time.time(); generate(model, tok, p, mt, True); t1 = time.time()
        soc, _ = meter.energy(t0, t1); res[mt].append((soc - idle_soc * (t1 - t0), t1 - t0))
meter.close()
pre, full = np.array(res[0]), np.array(res[24])
out = dict(prefill_J=float(pre[:, 0].mean()), full_J=float(full[:, 0].mean()), prefill_s=float(pre[:, 1].mean()),
           full_s=float(full[:, 1].mean()), prefill_share=float(pre[:, 0].mean() / full[:, 0].mean()), n=len(ps), idle_soc=idle_soc)
json.dump(out, open("results/prefill_cost.json", "w"), indent=2); print(out)
