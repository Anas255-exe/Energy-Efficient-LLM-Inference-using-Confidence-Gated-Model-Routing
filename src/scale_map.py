"""Zero-GPU scale map: replay cascades for every same-family pair up to 72B on Open LLM Leaderboard MMLU-Pro
per-question logs (12,032 questions, 10-option log-likelihood) -> results_olb/scale_summary.json, scale_map.png

Gate = 1 - max option probability (softmax over the 10 option log-likelihoods), free like token entropy.
Energy ratio rho = (N_s/N_l)^beta, beta = the range of per-query energy exponents we measured for drafts >= 3B:
0.86 (T4 fp16 3B->7B, gross) to 1.13 (T4 4-bit 14B->32B, run C; 1.1250); M2 3B->7B 0.91, 4-bit 7B->14B 0.95, 7B->32B 1.05.
Drafts < 3B measured lower on T4 (0.42-0.55, idle-dominated), so their projections are optimistic.
Split as analyze_pairs.py: nothing is fit, so all 12,032 questions are reported.
usage: python src/scale_map.py results_olb/mmlu_pro_min.json
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results_olb/mmlu_pro_min.json"))
OUT = "results_olb"
N = {"Qwen__Qwen2.5-0.5B-Instruct": 0.49, "Qwen__Qwen2.5-1.5B-Instruct": 1.54, "Qwen__Qwen2.5-3B-Instruct": 3.09,
     "Qwen__Qwen2.5-7B-Instruct": 7.62, "Qwen__Qwen2.5-14B-Instruct": 14.7, "Qwen__Qwen2.5-32B-Instruct": 32.5,
     "Qwen__Qwen2.5-72B-Instruct": 72.7, "meta-llama__Llama-3.2-1B-Instruct": 1.24, "meta-llama__Llama-3.2-3B-Instruct": 3.21,
     "meta-llama__Llama-3.1-8B-Instruct": 8.03, "meta-llama__Llama-3.1-70B-Instruct": 70.6}  # billions, model cards
short = lambda m: m.split("__")[1].replace("-Instruct", "")
ids = sorted(set.intersection(*[{r[0] for r in D[m]} for m in D]))
acc, risk = {}, {}
for m, rows in D.items():
    by = {r[0]: r for r in rows}
    acc[m] = np.array([by[i][2] for i in ids], float)
    ll = np.array([by[i][3] + [-1e9] * (10 - len(by[i][3])) for i in ids])  # some questions have < 10 options
    p = np.exp(ll - ll.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    risk[m] = 1 - p.max(1)
rates = np.linspace(0, 1, 101)
BETAS = (0.86, 1.13)  # measured range, see docstring


def replay(s, l, score):
    order = np.argsort(-score, kind="stable")
    a = []
    for r in rates:
        esc = np.zeros(len(ids), bool); esc[order[:int(round(r * len(ids)))]] = True
        a.append(np.where(esc, acc[l], acc[s]).mean())
    return np.array(a)


def saving(a, rho, R, router=False):
    ok = np.where(a >= R * a[-1])[0]  # a[-1] = always-large accuracy
    if not len(ok):
        return None
    r = rates[ok[0]]
    return 1 - ((1 - r) * rho + r if router else rho + r)


fam = lambda m: m.split("__")[0]
pairs = [(s, l) for s in D for l in D if fam(s) == fam(l) and N[s] < N[l]]
out = dict(n=len(ids), accuracy={short(m): acc[m].mean() for m in D}, pairs={})
for s, l in pairs:
    y = (1 - acc[s]).astype(int)
    a_gate, a_orc = replay(s, l, risk[s]), replay(s, l, y + 1e-3 * risk[s])
    both = (acc[s] * acc[l]).sum()
    po = dict(acc_s=acc[s].mean(), acc_l=acc[l].mean(), coverage=both / acc[l].sum(), auroc=roc_auc_score(y, risk[s]),
              size_ratio=N[s] / N[l])
    for beta in BETAS:
        rho = (N[s] / N[l]) ** beta
        po[f"beta{beta}"] = dict(rho=rho, break_even=1 - rho,
                                 **{f"save{int(R * 100)}": saving(a_gate, rho, R) for R in (0.90, 0.95, 0.99)},
                                 oracle_save95=saving(a_orc, rho, 0.95))
    out["pairs"][f"{short(s)}->{short(l)}"] = po
json.dump(out, open(f"{OUT}/scale_summary.json", "w"), indent=1, default=float)

f = lambda x: "  –  " if x is None else f"{x * 100:+5.0f}%"
print("n", len(ids), {k: round(v, 3) for k, v in out["accuracy"].items()})
print(f"{'pair':28s} acc_s  acc_l  cover  AUROC  rho.86  save95  save99  oracle95 | beta{BETAS[1]}: save95")
for k, v in out["pairs"].items():
    b, b1 = v[f"beta{BETAS[0]}"], v[f"beta{BETAS[1]}"]
    print(f"{k:28s} {v['acc_s']:.3f}  {v['acc_l']:.3f}  {v['coverage']:.3f}  {v['auroc']:.3f}  {b['rho']:.2f}  "
          f"{f(b['save95'])}  {f(b['save99'])}  {f(b['oracle_save95'])} | {f(b1['save95'])}")

# heat-map: Qwen draft (rows) x Qwen large (cols), saving at 95% retention, conservative beta
Q = [m for m in N if m.startswith("Qwen")]
H = np.full((len(Q), len(Q)), np.nan)
for i, s in enumerate(Q):
    for j, l in enumerate(Q):
        v = out["pairs"].get(f"{short(s)}->{short(l)}")
        if v and v[f"beta{BETAS[0]}"]["save95"] is not None:
            H[i, j] = v[f"beta{BETAS[0]}"]["save95"] * 100
fig, ax = plt.subplots(figsize=(6.4, 5))
im = ax.imshow(H, cmap="RdYlGn", vmin=-60, vmax=60)
for i in range(len(Q)):
    for j in range(len(Q)):
        if not np.isnan(H[i, j]):
            ax.text(j, i, f"{H[i, j]:+.0f}", ha="center", va="center", fontsize=8)
lab = [short(m).replace("Qwen2.5-", "") for m in Q]
ax.set_xticks(range(len(Q)), lab); ax.set_yticks(range(len(Q)), lab)
ax.set_xlabel("large model (Qwen2.5)"); ax.set_ylabel("draft model")
ax.set_title(f"Projected energy saving at 95% of large-model accuracy\nMMLU-Pro, max-prob gate, $\\rho=(N_S/N_L)^{{{BETAS[0]}}}$")
fig.colorbar(im, label="% energy saved vs always-large"); fig.tight_layout(); fig.savefig(f"{OUT}/scale_map.png", dpi=150)
