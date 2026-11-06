"""Big pairs on Kaggle 2x T4 (results_big/): Qwen2.5 7B, 14B, 32B, all 4-bit (bnb), same 3,000 questions as results_kaggle
-> results_big/big_summary.json

Energy: 7B and 14B measured (NVML gross J, mean of 100 real batch-1 queries; also net of cold idle).
32B ran out of Kaggle time before its energy step in run B; run C measured it (C_gross / C_net_cold). beta_4bit keeps the
earlier projection rho = (N_s/N_l)^beta from the measured 4-bit 7B->14B ratio, for comparison. Gates are free (token entropy, log-prob): nothing is fit,
so every answered question is used (32B answered 2,664 of 3,000 before the time cap).
usage: python src/analyze_big.py
"""
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score

D = "results_big"
load = lambda f: [json.loads(l) for l in open(f)]
R = {"7b": load(f"{D}/a/7b.jsonl"), "14b": load(f"{D}/a/14b.jsonl"), "32b": load(f"{D}/b/32b.jsonl")}
N = {"7b": 7.62, "14b": 14.7, "32b": 32.5}  # billions, model cards
col = lambda m, k, n: np.array([r[k] for r in R[m][:n]], float)
rng = np.random.default_rng(0)

meter = lambda f: {k: np.mean([r[k] for r in load(f)]) for k in ("J", "J_net_cold", "seconds")}
E = {m: meter(f"{D}/a/energy_validate_{m}.jsonl") for m in ("7b", "14b")}
# run C: energy only, 32B + 14B + 7B on the same 100 queries in one session (B hit the Kaggle time limit before energy)
EC = {m: meter(f"{D}/c/energy_validate_{m}.jsonl") for m in N if os.path.exists(f"{D}/c/energy_validate_{m}.jsonl")}
beta_4bit = np.log(E["14b"]["J"] / E["7b"]["J"]) / np.log(N["14b"] / N["7b"])
out = dict(energy=E, energy_C=EC, beta_4bit_gross=beta_4bit, accuracy={m: np.mean([r["correct"] for r in R[m]]) for m in R},
           n={m: len(R[m]) for m in R}, pairs={})


def at(cs, cl, score, rho, Rr):
    """Smallest escalation rate keeping Rr of always-large accuracy, and the cascade saving 1 - rho - r."""
    n = len(cs); o = np.argsort(-score, kind="stable")
    acc = (cs.sum() - np.r_[0, np.cumsum(cs[o])] + np.r_[0, np.cumsum(cl[o])]) / n  # acc with top-k escalated, k = 0..n
    k = np.argmax(acc >= Rr * cl.mean() - 1e-12)
    return k / n, 1 - rho - k / n


for s, l in (("7b", "14b"), ("7b", "32b"), ("14b", "32b")):
    n = min(len(R[s]), len(R[l]))
    cs, cl = col(s, "correct", n), col(l, "correct", n)
    ent, lp = col(s, "ent_mean", n), -col(s, "lp_mean", n)
    y, resc = 1 - cs, (1 - cs) * cl
    rhos = {"beta_4bit": (N[s] / N[l]) ** beta_4bit}
    if s in EC and l in EC:
        rhos = {"C_gross": EC[s]["J"] / EC[l]["J"], "C_net_cold": EC[s]["J_net_cold"] / EC[l]["J_net_cold"], **rhos}
    if l == "14b":
        rhos = {"measured_gross": E[s]["J"] / E[l]["J"], "measured_net_cold": E[s]["J_net_cold"] / E[l]["J_net_cold"], **rhos}
    po = dict(n=n, acc_s=cs.mean(), acc_l=cl.mean(), coverage=(cs * cl).sum() / cl.sum(),
              auroc_entropy=roc_auc_score(y, ent), auroc_logprob=roc_auc_score(y, lp),
              auroc_rescuable_vs_all=roc_auc_score(resc, ent), rho=rhos, save={})
    for name, rho in rhos.items():
        for Rr in (0.95, 0.99):
            r, sv = at(cs, cl, ent, rho, Rr)
            boot = [at(cs[b], cl[b], ent[b], rho, Rr)[1] for b in (rng.integers(0, n, n) for _ in range(300))]
            po["save"][f"{name}@{int(Rr * 100)}"] = dict(escalate=r, saving=sv, ci=list(np.percentile(boot, [2.5, 97.5])),
                                                        oracle=at(cs, cl, y + 1e-3 * ent, rho, Rr)[1])
    out["pairs"][f"{s}->{l}"] = po

json.dump(out, open(f"{D}/big_summary.json", "w"), indent=1, default=float)
print("acc", {m: round(v, 3) for m, v in out["accuracy"].items()}, "n", out["n"])
print("J/query A", {m: {k: round(v, 2) for k, v in e.items()} for m, e in E.items()}, "C", {m: {k: round(v, 2) for k, v in e.items()} for m, e in EC.items()}, "beta_4bit", round(beta_4bit, 2))
for k, v in out["pairs"].items():
    print(f"\n{k} n={v['n']} acc {v['acc_s']:.3f}->{v['acc_l']:.3f} cover {v['coverage']:.3f} "
          f"AUROC ent {v['auroc_entropy']:.3f} lp {v['auroc_logprob']:.3f} rescuable {v['auroc_rescuable_vs_all']:.3f}")
    for name, x in v["save"].items():
        print(f"  {name:24s} rho {v['rho'][name.split('@')[0]]:.2f} esc {x['escalate']:.2f} save {x['saving']:+.1%} "
              f"[{x['ci'][0]:+.1%}, {x['ci'][1]:+.1%}] oracle {x['oracle']:+.1%}")
