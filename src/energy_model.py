"""Token-level energy model per model size: E_net ≈ c + a·prompt_tokens + b·output_tokens.

16 GB-safe sweep: ONE model resident at a time, MLX buffer cache capped and cleared after every job, rows
appended to disk as they come (a crash keeps what ran). Before each model block the Mac cools to < COOL °C and
idle power is measured; within a block every (P, O) cell runs in random order per repeat. generate_step ignores
EOS, so exactly O tokens are decoded. Energy = SoC + DRAM (macmon). The 3B and 7B blocks end with held-out real
chat queries (stock stream_generate) for validation. Then: per-model least-squares fit, validation, scaling plot.

usage: PYTHONPATH=src python src/energy_model.py [--reps 3] [--fit-only]
"""
import argparse, gc, json, random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

R = "results"
PARAMS = {"0.5b": 0.49, "1.5b": 1.54, "3b": 3.09, "7b": 7.62}  # billions (Qwen2.5 model cards)
PATHS = {"0.5b": "models/q05b", "1.5b": "models/q15b", "3b": "models/q3b", "7b": "models/q7b"}
PS, OS = (16, 64, 256, 1024), (1, 8, 32, 128)
COOL = 60  # °C GPU temp before each model block; the first attempt hit 89 °C and ran the Mac out of memory

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--n-val", type=int, default=40)
ap.add_argument("--fit-only", action="store_true")
a = ap.parse_args()
greedy = make_sampler(temp=0.0)

if not a.fit_only:
    from mlx_lm import stream_generate
    from energy import PowerMeter
    from run import SYSTEM
    mx.set_cache_limit(256 * 2**20)
    mx.set_memory_limit(9 * 2**30)  # 7B 4-bit is ~4.3 GB; leave the rest of 16 GB to macOS and apps
    text = " ".join(json.loads(l)["question"] for l in open("data/questions.jsonl"))
    vqs = [json.loads(l) for l in open("data/questions.jsonl")][-a.n_val:]
    fs, fv = open(f"{R}/energy_sweep.jsonl", "w"), open(f"{R}/energy_validate.jsonl", "w")
    meter, rnd, idle_log = PowerMeter(), random.Random(0), {}

    def measure(fn):
        time.sleep(0.2)  # let power settle so one job does not bleed into the next window
        t0 = time.time(); out = fn(); t1 = time.time()
        mx.clear_cache()
        return out, t1 - t0, meter.energy(t0, t1)[0]

    for m in rnd.sample(list(PATHS), len(PATHS)):  # random block order so drift is not confounded with size
        while meter.temp() > COOL:
            time.sleep(5)
        idle = meter.idle_power(15)[0]
        idle_log[m] = idle
        model, tok = load(PATHS[m])
        ids = tok.encode(text)
        step = lambda P, O: [None for _ in generate_step(mx.array(ids[:P]), model, max_tokens=O, sampler=greedy)]
        step(64, 8)  # warm-up
        for rep in range(a.reps):
            for P, O in rnd.sample([(P, O) for P in PS for O in OS], len(PS) * len(OS)):
                _, sec, J = measure(lambda: step(P, O))
                fs.write(json.dumps(dict(model=m, P=P, O=O, rep=rep, seconds=sec, J=J, J_net=J - idle * sec, temp=meter.temp())) + "\n")
            fs.flush()
            print(m, "rep", rep, "temp", round(meter.temp(), 1), "peak GB", round(mx.get_peak_memory() / 2**30, 2), flush=True)
        if m in ("3b", "7b"):  # held-out validation: real chat prompts, short answers
            for q in vqs:
                p = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question"]}],
                                            add_generation_prompt=True)
                txt, sec, J = measure(lambda: "".join(r.text for r in stream_generate(model, tok, p, max_tokens=24, sampler=greedy)))
                n = len(tok.encode(txt))
                fv.write(json.dumps(dict(id=q["id"], model=m, P=len(p), O=24 if n >= 24 else n + 1,  # +1: EOS step computed, not yielded
                                         seconds=sec, J=J, J_net=J - idle * sec)) + "\n")
            fv.flush()
        del model, tok; gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
    meter.close(); fs.close(); fv.close()
    json.dump(dict(idle_W_per_block=idle_log), open(f"{R}/energy_sweep_idle.json", "w"))

# --- fit E = c + a·P + b·O (and the same for seconds) per model
rows = [json.loads(l) for l in open(f"{R}/energy_sweep.jsonl")]
fit = {}
for m in PARAMS:
    d = [r for r in rows if r["model"] == m]
    X = np.array([[1, r["P"], r["O"]] for r in d], float)
    out = {}
    for k in ("J", "J_net", "seconds"):
        yv = np.array([r[k] for r in d])
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        pred = X @ coef
        out[k] = dict(c=coef[0], a_per_prompt_tok=coef[1], b_per_output_tok=coef[2],
                      r2=1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum(),
                      mape=float(np.mean(np.abs(pred - yv) / np.abs(yv))))
    fit[m] = out

# --- validate on held-out real chat queries (energy_validate.jsonl), gross and net
vr = [json.loads(l) for l in open(f"{R}/energy_validate.jsonl")]
val = {}
for m in ("3b", "7b"):
    d = [r for r in vr if r["model"] == m]
    for k in ("J", "J_net", "seconds"):
        f = fit[m][k]
        meas = np.array([r[k] for r in d])
        pred = np.array([f["c"] + f["a_per_prompt_tok"] * r["P"] + f["b_per_output_tok"] * r["O"] for r in d])
        val[f"{m}_{k}"] = dict(measured_mean=meas.mean(), predicted_mean=pred.mean(), mean_error=pred.mean() / meas.mean() - 1,
                               per_query_mape=float(np.mean(np.abs(pred - meas) / np.abs(meas))))
val["ratio_3b_over_7b"] = dict(validate=val["3b_J_net"]["measured_mean"] / val["7b_J_net"]["measured_mean"],
                               model=val["3b_J_net"]["predicted_mean"] / val["7b_J_net"]["predicted_mean"],
                               cost_run_nov2_1am=0.44)  # results/cost_summary.json (net); gross 0.43

# --- scaling: per-token energy vs parameters (log-log power law)
p = np.array([PARAMS[m] for m in PARAMS])
scal = {}
for k in ("a_per_prompt_tok", "b_per_output_tok", "c"):
    v = np.array([fit[m]["J_net"][k] for m in PARAMS])
    ok = v > 0
    slope, icpt = np.polyfit(np.log(p[ok]), np.log(v[ok]), 1)
    scal[k] = dict(exponent=slope, J_at_1B=float(np.exp(icpt)))
json.dump(dict(fit=fit, validation=val, scaling=scal), open(f"{R}/energy_model.json", "w"), indent=2, default=float)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, lab in (("b_per_output_tok", "J per output token"), ("a_per_prompt_tok", "J per prompt token")):
    v = [fit[m]["J_net"][k] for m in PARAMS]
    ax[0].loglog(p, v, "o-", label=f"{lab} (∝ N^{scal[k]['exponent']:.2f})")
ax[0].set_xlabel("parameters (B)"); ax[0].set_ylabel("J/token (SoC+DRAM, net)"); ax[0].legend(fontsize=8)
for m in PARAMS:
    d = [r for r in rows if r["model"] == m and r["P"] == 64]
    O = sorted({r["O"] for r in d})
    ax[1].plot(O, [np.mean([r["J_net"] for r in d if r["O"] == o]) for o in O], "o-", label=m)
ax[1].set_xlabel("output tokens (prompt = 64)"); ax[1].set_ylabel("J per query (net)"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{R}/energy_scaling.png", dpi=150)
print(json.dumps(dict(fit=fit, validation=val, scaling=scal), indent=2, default=float))
