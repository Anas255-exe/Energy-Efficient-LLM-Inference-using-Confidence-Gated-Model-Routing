"""Energy, latency, hallucination rate and carbon at 95% retention for every measured pair
-> results_eval/eval_summary.json, results_eval/EVAL.md

Hallucination rate = share of questions answered wrongly (models never abstain). Gate = draft token entropy.
Latency is sequential: the draft always answers, the target runs after it when the query is escalated.
Carbon = device energy x grid factor (no PUE, no host CPU). Splits and grids match the paper:
M2 and T4 fp16 report the last 30% on the 21-point grid, T4 4-bit uses every answered question and every k/n.
usage: python src/eval_extra.py
"""
import json, os
import numpy as np

GRID = {  # g CO2 per kWh of electricity
    "India": 727,   # CEA CO2 Baseline Database v20.0 (Dec 2024), weighted average incl. RE, FY2023-24
    "World": 458,   # Ember, Global Electricity Review 2026 (2025 data), gCO2e/kWh
    "France": 19.6,  # RTE, Bilan electrique 2025, direct emissions per kWh generated
}
load = lambda f: [json.loads(l) for l in open(f)]
meanof = lambda f, k: float(np.mean([r[k] for r in load(f)]))


def replay(cs, cl, score, es, el, ts, tl, grid=False):
    """Escalate the top-k by score for every allowed k: escalation rate, accuracy, mean J and seconds per query."""
    n = len(cs); o = np.argsort(-score, kind="stable")
    cum = lambda x: np.r_[0, np.cumsum(np.broadcast_to(np.asarray(x, float), (n,))[o])] / n
    ks = np.round(np.linspace(0, 1, 21) * n).astype(int) if grid else np.arange(n + 1)
    return ks / n, (cs.sum() / n - cum(cs) + cum(cl))[ks], (np.mean(es + 0 * cs) + cum(el))[ks], (np.mean(ts + 0 * cs) + cum(tl))[ks]


def pair(name, platform, cs, cl, ent, es, el, ts, tl, grid):
    J_l, s_l = float(np.mean(el + 0 * cl)), float(np.mean(tl + 0 * cl))
    r, acc, J, s = replay(cs, cl, ent, es, el, ts, tl, grid=grid)
    co2 = lambda J: {g: J / 3.6e6 * f * 1000 for g, f in GRID.items()}  # g CO2 per 1,000 queries
    at = lambda R: int(np.argmax(acc >= R * cl.mean() - 1e-12))  # smallest k keeping R of always-large accuracy
    i = at(0.95)
    curve = [dict(R=R, escalate=r[j], err=1 - acc[j], saving=1 - J[j] / J_l, latency_change=s[j] / s_l - 1)
             for R, j in ((R, at(R)) for R in np.round(np.arange(0.80, 1.0001, 0.005), 3))]
    return name, dict(platform=platform, n=len(cs), err_small=1 - cs.mean(), err_large=1 - cl.mean(),
                      J_small=float(np.mean(es + 0 * cs)), J_large=J_l, s_small=float(np.mean(ts + 0 * cs)), s_large=s_l,
                      cascade=dict(escalate=r[i], err=1 - acc[i], J=J[i], s=s[i]), saving=1 - J[i] / J_l,
                      latency_change=s[i] / s_l - 1, gCO2_per_1k=dict(large=co2(J_l), cascade=co2(J[i])), curve=curve)


out = {}
# Apple M2, 3B->7B: per-query energy and time from the interleaved cost run (results/cost.jsonl), test = last 30%
s, l = load("results/3b.jsonl"), {r["id"]: r for r in load("results/7b.jsonl")}
s = [r for r in s if r["id"] in l][700:]
C = {(r["id"], r["job"]): r for r in load("results/cost.jsonl")}
v = lambda m, k: np.array([C[(r["id"], m)][k] for r in s])
k, o = pair("3B->7B", "Apple M2, MLX 4-bit", np.array([r["correct"] for r in s], float),
            np.array([l[r["id"]]["correct"] for r in s], float), np.array([r["ent_mean"] for r in s]),
            v("3b", "J_net"), v("7b", "J_net"), v("3b", "seconds"), v("7b", "seconds"), grid=True)
out[f"M2 {k}"] = o

# T4 fp16: mean of 100 real batch-1 queries per model, test = last 30%
D = "results_kaggle"
R = {m: load(f"{D}/{m}.jsonl")[2100:] for m in ("0.5b", "1.5b", "3b", "7b")}
cor = lambda m: np.array([r["correct"] for r in R[m]], float)
for a, b in (("0.5b", "7b"), ("1.5b", "7b"), ("3b", "7b"), ("0.5b", "3b"), ("1.5b", "3b")):
    f = lambda m, key: meanof(f"{D}/energy_validate_{m}.jsonl", key)
    k, o = pair(f"{a.upper()}->{b.upper()}", "2x T4, fp16", cor(a), cor(b), np.array([r["ent_mean"] for r in R[a]]),
                f(a, "J"), f(b, "J"), f(a, "seconds"), f(b, "seconds"), grid=True)
    out[f"T4 fp16 {k}"] = o

# T4 4-bit: energy run A for 7B->14B, run C for the 32B pairs (as in the paper), every answered question
D = "results_big"
R = {"7b": load(f"{D}/a/7b.jsonl"), "14b": load(f"{D}/a/14b.jsonl"), "32b": load(f"{D}/b/32b.jsonl")}
for a, b, run in (("7b", "14b", "a"), ("7b", "32b", "c"), ("14b", "32b", "c")):
    n = min(len(R[a]), len(R[b]))
    cor = lambda m: np.array([r["correct"] for r in R[m][:n]], float)
    f = lambda m, key: meanof(f"{D}/{run}/energy_validate_{m}.jsonl", key)
    k, o = pair(f"{a.upper()}->{b.upper()}", "2x T4, 4-bit", cor(a), cor(b), np.array([r["ent_mean"] for r in R[a][:n]]),
                f(a, "J"), f(b, "J"), f(a, "seconds"), f(b, "seconds"), grid=False)
    out[f"T4 4-bit {k}"] = o

os.makedirs("results_eval", exist_ok=True)
json.dump(dict(grid_gCO2_per_kWh=GRID, pairs=out), open("results_eval/eval_summary.json", "w"), indent=1, default=float)

pct = lambda x: f"{x:+.0%}"
lines = ["| Pair | Platform | Hallucination rate: small / large / cascade | Escalated | J/query large → cascade (saving) "
         "| s/query large → cascade (change) | g CO2 per 1,000 queries, India grid: large → cascade |", "|---|---|---|---|---|---|---|"]
for k, o in out.items():
    c, g = o["cascade"], o["gCO2_per_1k"]
    lines.append(f"| {k.split()[-1]} | {o['platform']} (n = {o['n']}) | {o['err_small']:.1%} / {o['err_large']:.1%} / {c['err']:.1%} "
                 f"| {c['escalate']:.0%} | {o['J_large']:.1f} → {c['J']:.1f} ({pct(o['saving'])}) "
                 f"| {o['s_large']:.2f} → {c['s']:.2f} ({pct(o['latency_change'])}) | {g['large']['India']:.2f} → {g['cascade']['India']:.2f} |")
b = out["T4 4-bit 14B->32B"]["gCO2_per_1k"]
lines += ["", "Carbon for 14B→32B, g CO2 per 1,000 queries (always-large → cascade):", "",
          "| Grid | g CO2 / kWh | Always-large | Cascade | Saved |", "|---|---|---|---|---|"]
lines += [f"| {g} | {f} | {b['large'][g]:.2f} | {b['cascade'][g]:.2f} | {b['large'][g] - b['cascade'][g]:.2f} |" for g, f in GRID.items()]
open("results_eval/EVAL.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
# dashboard: one offline HTML file with the data inlined (file:// pages cannot fetch)
rf = "results_eval/router_eval_summary.json"  # live M2 router run (src/route.py --eval), if done
router = json.load(open(rf)) if os.path.exists(rf) else None
html = open("dashboard/template.html").read().replace("/*DATA*/null", json.dumps(dict(grid=GRID, pairs=out, router=router), default=float))
open("dashboard/greengate_dashboard.html", "w").write(html)
