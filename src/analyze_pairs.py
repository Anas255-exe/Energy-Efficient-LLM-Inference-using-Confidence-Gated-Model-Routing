"""Kaggle run (results_kaggle/): every draft->target pair, gates incl. semantic entropy, energy from the GPU
token-level model -> results_kaggle/pairs_summary.json, pairs_pareto.png

Per-query energy E_m(q) = c + a·P + b·O from each model's NVML sweep (O = answer tokens + EOS step).
Splits as analyze.py: probes fit on the first 70% (layer by 5-fold CV), everything reported on the last 30%.
usage: python src/analyze_pairs.py [dir]
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = sys.argv[1] if len(sys.argv) > 1 else "results_kaggle"
MODELS, DRAFTS = ("0.5b", "1.5b", "3b", "7b"), ("0.5b", "1.5b", "3b")
PAIRS = [("0.5b", "7b"), ("1.5b", "7b"), ("3b", "7b"), ("0.5b", "3b"), ("1.5b", "3b")]
rng = np.random.default_rng(0)
load = lambda f: [json.loads(l) for l in open(f)]
qs = load(f"{D}/questions.jsonl")
n = len(qs)
fit_idx, te = np.arange(n) < 0.7 * n, np.arange(n) >= 0.7 * n
R = {m: load(f"{D}/{m}.jsonl") for m in MODELS}
assert all([r["id"] for r in R[m]] == [q["id"] for q in qs] for m in MODELS)
col = lambda m, k: np.array([r[k] for r in R[m]], dtype=float)
acc = {m: col(m, "correct") for m in MODELS}

# --- token-level energy model per model (NVML gross; J_net unreliable: idle was read on a hot GPU, 0.5b validate err -87%)
out = dict(n=n, n_test=int(te.sum()), accuracy={m: acc[m][te].mean() for m in MODELS}, energy_model={}, pairs={})
E = {}
for m in MODELS:
    sw = load(f"{D}/energy_sweep_{m}.jsonl")
    X = np.array([[1, r["P"], r["O"]] for r in sw], float)
    coef, *_ = np.linalg.lstsq(X, np.array([r["J"] for r in sw]), rcond=None)
    va = load(f"{D}/energy_validate_{m}.jsonl")
    pv = np.array([coef @ [1, r["P"], r["O"]] for r in va]); mv = np.array([r["J"] for r in va])
    out["energy_model"][m] = dict(c=coef[0], a=coef[1], b=coef[2], validate_mean_error=pv.mean() / mv.mean() - 1,
                                  validate_measured_J=mv.mean())
    # ponytail: per-query cost = measured mean on 100 real batch-1 queries (the linear fit misses 7b by -39%);
    # router prefill share from the time model, which validates within 7% on the drafts
    tc, *_ = np.linalg.lstsq(X, np.array([r["seconds"] for r in sw]), rcond=None)
    E[m] = np.full(n, mv.mean())
    E[m + "_prefill"] = E[m] * (tc @ np.array([np.ones(n), col(m, "P"), np.ones(n)])) / (tc @ np.array([np.ones(n), col(m, "P"), col(m, "n_tokens") + 1]))
p = np.log([0.49, 1.54, 3.09, 7.62])
for k in ("a", "b"):
    out["energy_model"][f"exponent_{k}"] = np.polyfit(p, np.log([out["energy_model"][m][k] for m in MODELS]), 1)[0]


# --- gates for each draft
def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


G = {}
for d in DRAFTS:
    y = (1 - acc[d]).astype(int)
    H = np.load(f"{D}/{d}_hidden.npy").astype(np.float32)  # [n, layers(every 2nd), 2, d]
    cv = {(pos, l): cross_val_score(probe(), H[fit_idx, l, pos], y[fit_idx], cv=5, scoring="roc_auc").mean()
          for pos in (0, 1) for l in range(1, H.shape[1])}
    pa = max((k for k in cv if k[0] == 1), key=cv.get); pp = max((k for k in cv if k[0] == 0), key=cv.get)
    se = {r["id"]: r["se"] for r in load(f"{D}/{d}_se.jsonl")}
    G[d] = {"random": rng.random(n), "oracle": y + rng.random(n) * 1e-3,
            "entropy_mean": col(d, "ent_mean"), "neg_logprob_mean": -col(d, "lp_mean"),
            "semantic_entropy": np.array([se[q["id"]] for q in qs]) + rng.random(n) * 1e-6,
            "probe_answer": probe().fit(H[fit_idx, pa[1], 1], y[fit_idx]).predict_proba(H[:, pa[1], 1])[:, 1],
            "prefill_probe": probe().fit(H[fit_idx, pp[1], 0], y[fit_idx]).predict_proba(H[:, pp[1], 0])[:, 1]}
    out.setdefault("probe_layers", {})[d] = dict(answer=dict(layer_idx=pa[1], cv=cv[pa]), prompt=dict(layer_idx=pp[1], cv=cv[pp]))

rates = np.linspace(0, 1, 21)


K_SE = 10  # semantic entropy draws K extra draft samples; charged as K x draft (upper bound, batching makes it cheaper)


def cascade(s, l, score, ii, router=False, extra=0.0):
    order = np.argsort(-score[ii], kind="stable")
    a, e = [], []
    for r in rates:
        esc = np.zeros(len(ii), bool); esc[order[:int(round(r * len(ii)))]] = True
        a.append(np.where(esc, acc[l][ii], acc[s][ii]).mean())
        draft = np.where(esc, E[s + "_prefill"][ii], E[s][ii]) if router else E[s][ii]
        e.append((draft * (1 + extra) + esc * E[l][ii]).mean())
    return np.array(a), np.array(e)


ii = np.where(te)[0]
fig, axes = plt.subplots(1, len(PAIRS), figsize=(4 * len(PAIRS), 3.6), sharey=False)
for ax, (s, l) in zip(axes, PAIRS):
    y = (1 - acc[s]).astype(int)
    El = E[l][te].mean()
    po = dict(acc_small=acc[s][te].mean(), acc_large=acc[l][te].mean(), J_small=E[s][te].mean(), J_large=El,
              break_even=1 - E[s][te].mean() / El, gates={})
    for g, sc in G[s].items():
        a, e = cascade(s, l, sc, ii, extra=K_SE if g == "semantic_entropy" else 0.0)
        boots = [roc_auc_score(y[b], sc[b]) for b in (rng.choice(ii, len(ii)) for _ in range(500))]
        ret = a / po["acc_large"]
        ok = np.where(ret >= 0.95)[0]
        po["gates"][g] = dict(auroc=roc_auc_score(y[te], sc[te]), auroc_ci=list(np.percentile(boots, [2.5, 97.5])),
                              aurc=float(np.trapezoid(a, rates)),
                              at95=None if not len(ok) else dict(escalation=rates[ok[0]], saving=1 - e[ok[0]] / El),
                              best_saving_at_90=max([1 - ee / El for aa, ee in zip(a, e) if aa / po["acc_large"] >= 0.9], default=None))
        if g in ("random", "oracle", "entropy_mean", "semantic_entropy", "probe_answer"):
            ax.plot(e, a, marker=".", lw=2 if g == "probe_answer" else 1, label=g)
    a, e = cascade(s, l, G[s]["prefill_probe"], ii, router=True)
    ax.plot(e, a, ls="--", label="router (prefill probe)")
    # paired ΔAUROC: semantic entropy and probe vs token entropy
    for g in ("semantic_entropy", "probe_answer"):
        dd = [roc_auc_score(y[b], G[s][g][b]) - roc_auc_score(y[b], G[s]["entropy_mean"][b]) for b in (rng.choice(ii, len(ii)) for _ in range(500))]
        po[f"d_auroc_{g}_vs_entropy"] = dict(mean=float(np.mean(dd)), ci=list(np.percentile(dd, [2.5, 97.5])))
    out["pairs"][f"{s}->{l}"] = po
    ax.axhline(po["acc_large"], c="k", ls=":", lw=0.8); ax.axvline(El, c="k", ls=":", lw=0.8)
    ax.set_title(f"{s} → {l}"); ax.set_xlabel("J/query (NVML gross, measured)")
axes[0].set_ylabel("accuracy (test)"); axes[-1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{D}/pairs_pareto.png", dpi=150)
json.dump(out, open(f"{D}/pairs_summary.json", "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1, default=float))
for k, v in out["pairs"].items():
    print(k, "acc", round(v["acc_small"], 3), round(v["acc_large"], 3), "BE", round(v["break_even"], 2),
          {g: (round(x["auroc"], 3), x["at95"] and round(x["at95"]["saving"], 2)) for g, x in v["gates"].items()})
