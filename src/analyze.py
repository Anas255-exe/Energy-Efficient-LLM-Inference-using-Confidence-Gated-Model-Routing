"""Probes + offline cascade replay -> results/summary.json, results/*.png, results/report_tables.md

Label y = 1 when the draft (3B) answer is wrong. Every gate gives a risk score;
the cascade escalates the top-r fraction of test queries to the 7B model.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
R = "results"
load = lambda f: [json.loads(l) for l in open(f)]
s, l = load(f"{R}/3b.jsonl"), load(f"{R}/7b.jsonl")
L = {r["id"]: r for r in l}
s = [r for r in s if r["id"] in L]
H = np.load(f"{R}/3b_hidden.npy").astype(np.float32)[: len(s)]  # [n, layers, 2, d]
n = len(s)
y = np.array([not r["correct"] for r in s], dtype=int)            # 1 = draft wrong
c_s = np.array([r["correct"] for r in s], dtype=float)
c_l = np.array([L[r["id"]]["correct"] for r in s], dtype=float)
# Energy cost basis: results/cost.jsonl (src/cost.py) - 3B, 7B and 3B-prefill run interleaved per test question with
# stock mlx-lm, SoC+DRAM power, idle (before/after mean) subtracted. The sequential runs drifted (3B 1.07->1.67 J/q), so
# their per-query energies are kept only as a secondary check. Non-test rows get NaN (energy is only used on test).
cost = load(f"{R}/cost.jsonl")
CJ = {(r["id"], r["job"]): r["J_net"] for r in cost}
E_s = np.array([CJ.get((r["id"], "3b"), np.nan) for r in s])
E_l = np.array([CJ.get((r["id"], "7b"), np.nan) for r in s])
E_p = np.array([CJ.get((r["id"], "3b_prefill"), np.nan) for r in s])
E_g = np.zeros(n)  # probe reads hidden states captured during generation (inside metered window); linear head ~0 J
ds = np.array([r["dataset"] for r in s])

tr, ca, te = np.arange(n) < 0.6 * n, (np.arange(n) >= 0.6 * n) & (np.arange(n) < 0.7 * n), np.arange(n) >= 0.7 * n
fit = tr | ca  # ponytail: calibration split folded into training; threshold is chosen by escalation rate, not by probability


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


# --- layer selection by 5-fold CV on training data only (test untouched)
cv = {}
for pos, pname in [(0, "prompt"), (1, "answer")]:
    for layer in range(0, H.shape[1], 2):
        cv[(pname, layer)] = cross_val_score(probe(), H[fit, layer, pos], y[fit], cv=5, scoring="roc_auc").mean()
best = max(cv, key=cv.get)
pos = 0 if best[0] == "prompt" else 1
clf = probe().fit(H[fit, best[1], pos], y[fit])
probe_score = clf.predict_proba(H[:, best[1], pos])[:, 1]
best_prompt_layer = max((l for p, l in cv if p == "prompt"), key=lambda l: cv[("prompt", l)])
pre_score = probe().fit(H[fit, best_prompt_layer, 0], y[fit]).predict_proba(H[:, best_prompt_layer, 0])[:, 1]

f = lambda k: np.array([r[k] for r in s])
gates = {
    "random": rng.random(n),
    "oracle": y + rng.random(n) * 1e-3,
    "neg_logprob_mean": -f("lp_mean"),
    "neg_logprob_min": -f("lp_min"),
    "entropy_mean": f("ent_mean"),
    "entropy_max": f("ent_max"),
    f"probe_{best[0]}_L{best[1]}": probe_score,
    f"prefill_probe_L{best_prompt_layer}": pre_score,  # pre-generation router (Hybrid-LLM style signal)
}
PROBE = f"probe_{best[0]}_L{best[1]}"


def auroc_ci(score, idx, B=1000):
    vals = []
    ii = np.where(idx)[0]
    for _ in range(B):
        b = rng.choice(ii, len(ii))
        if y[b].min() != y[b].max():
            vals.append(roc_auc_score(y[b], score[b]))
    return roc_auc_score(y[idx], score[idx]), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def cascade(score, ii, rates, gate_cost, router=False):
    """accuracy and mean J/query over index array ii (duplicates allowed, for bootstrap) when escalating
    the top-r fraction by score. Cascade: escalated pays draft + large. Router: escalated pays draft prefill + large."""
    order = np.argsort(-score[ii], kind="stable")
    acc, en = [], []
    for r in rates:
        esc = np.zeros(len(ii), bool); esc[order[:int(round(r * len(ii)))]] = True
        acc.append(np.where(esc, c_l[ii], c_s[ii]).mean())
        draft = np.where(esc, E_p[ii], E_s[ii]) if router else E_s[ii]
        en.append((draft + gate_cost[ii] + esc * E_l[ii]).mean())
    return np.array(acc), np.array(en)


rates = np.linspace(0, 1, 21)
gate_cost = {g: E_g for g in gates}

out = dict(n=int(n), n_test=int(te.sum()), best_probe=dict(position=best[0], layer=int(best[1]), cv_auc=cv[best]),
           acc_small=c_s[te].mean(), acc_large=c_l[te].mean(),
           acc_small_strict=f("correct_strict")[te].mean(), acc_large_strict=np.array([L[r["id"]]["correct_strict"] for r in s])[te].mean(),
           J_small=E_s[te].mean(), J_large=E_l[te].mean(), J_gate=E_g[te].mean(),
           lat_small=f("latency")[te].mean(), lat_large=np.array([L[r["id"]]["latency"] for r in s])[te].mean(),
           break_even=1 - (E_s[te].mean() + E_g[te].mean()) / E_l[te].mean(),
           break_even_no_gate=1 - E_s[te].mean() / E_l[te].mean(),
           gates={})
curves = {}
for g, sc in gates.items():
    au = auroc_ci(sc, te)
    acc, en = cascade(sc, np.where(te)[0], rates, gate_cost[g])
    curves[g] = (acc, en)
    ret = acc / out["acc_large"]
    def cheapest(th):
        ok = np.where(ret >= th)[0]
        return None if not len(ok) else dict(escalation=float(rates[ok[0]]), J=float(en[ok[0]]),
                                             saving_vs_large=float(1 - en[ok[0]] / out["J_large"]))
    out["gates"][g] = dict(auroc=au[0], auroc_ci=[au[1], au[2]], aurc=float(np.trapezoid(acc, rates)),
                           at95=cheapest(0.95), at100=cheapest(1.0), acc_at_30pct=float(acc[6]))

# Pre-generation router (Measured-Joules / Hybrid-LLM style): escalated queries skip the draft answer.
# Escalated queries still pay the draft prefill (needed to read hidden states): E_p, measured per question
# (3B with max_tokens=1, interleaved in src/cost.py -> slight over-estimate of a pure prefill).
RG = f"prefill_probe_L{best_prompt_layer}"
r_acc, r_en = cascade(gates[RG], np.where(te)[0], rates, E_g, router=True)
curves["router_" + RG] = (r_acc, r_en)
es, el = E_s[te].mean(), E_l[te].mean()
PRE = float(E_p[te].mean() / es)
out["router"] = dict(gate=RG, prefill_share=PRE, break_even=float((el - es) / (el - es + PRE * es)), points=[dict(escalation=float(r), acc=float(a), retained=float(a / out["acc_large"]),
                                           J=float(e), saving_vs_large=float(1 - e / out["J_large"]))
                                      for r, a, e in zip(rates, r_acc, r_en)])
out["operating_points"] = {g: [dict(escalation=float(r), acc=float(a), retained=float(a / out["acc_large"]),
                                    J=float(e), saving_vs_large=float(1 - e / out["J_large"]))
                               for r, a, e in zip(rates, *curves[g])][:11]
                           for g in (PROBE, "entropy_mean", "oracle")}

# paired bootstrap: probe vs best non-oracle confidence baseline, area under accuracy-vs-escalation
base = max([g for g in gates if g not in ("random", "oracle", PROBE) and "prefill" not in g], key=lambda g: out["gates"][g]["aurc"])
diffs = []
ii = np.where(te)[0]
for _ in range(1000):
    b = rng.choice(ii, len(ii))  # true resample with replacement
    a1, _ = cascade(gates[PROBE], b, rates, E_g); a2, _ = cascade(gates[base], b, rates, E_g)
    diffs.append(np.trapezoid(a1, rates) - np.trapezoid(a2, rates))
dau = []
for _ in range(1000):
    b = rng.choice(ii, len(ii))
    dau.append(roc_auc_score(y[b], gates[PROBE][b]) - roc_auc_score(y[b], gates["entropy_mean"][b]))
out["probe_vs_best_baseline"] = dict(baseline=base, d_aurc=float(np.mean(diffs)), ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                     d_auroc_vs_entropy=float(np.mean(dau)), d_auroc_ci=[float(np.percentile(dau, 2.5)), float(np.percentile(dau, 97.5))])

# RQ5-lite: train probe on TriviaQA only, test on SciQ
tq, sq = (ds == "triviaqa") & fit, ds == "sciq"
pt = probe().fit(H[tq, best[1], pos], y[tq])
out["transfer_tqa_to_sciq"] = dict(auroc=roc_auc_score(y[sq & te], pt.predict_proba(H[sq & te, best[1], pos])[:, 1]),
                                   in_domain=roc_auc_score(y[sq & te], probe_score[sq & te]))
out["per_dataset_test"] = {d: dict(acc_small=c_s[te & (ds == d)].mean(), acc_large=c_l[te & (ds == d)].mean(),
                                   probe_auroc=roc_auc_score(y[te & (ds == d)], probe_score[te & (ds == d)]))
                           for d in np.unique(ds)}

json.dump(out, open(f"{R}/summary.json", "w"), indent=2, default=float)

# --- plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for g, (acc, en) in curves.items():
    ax[0].plot(rates, acc, label=g, lw=2.5 if g == PROBE else 1.2)
    ax[1].plot(en, acc, marker=".", label=g, lw=2.5 if g == PROBE else 1.2)
ax[0].axhline(out["acc_large"], ls="--", c="k", lw=0.8); ax[0].set_xlabel("escalation rate"); ax[0].set_ylabel("accuracy (test)")
ax[1].axhline(out["acc_large"], ls="--", c="k", lw=0.8); ax[1].axvline(out["J_large"], ls=":", c="k", lw=0.8)
ax[1].set_xlabel("measured energy per query (J, CPU+GPU+DRAM, idle-subtracted; interleaved cost run)"); ax[1].set_ylabel("accuracy (test)")
ax[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{R}/pareto.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 3.5))
for pname in ("prompt", "answer"):
    ks = sorted(l for p, l in cv if p == pname)
    ax.plot(ks, [cv[(pname, k)] for k in ks], marker="o", label=f"last {pname} token")
ax.set_xlabel("layer"); ax.set_ylabel("CV AUROC (train)"); ax.legend(); fig.tight_layout(); fig.savefig(f"{R}/auroc_by_layer.png", dpi=150)

print(json.dumps(out, indent=2, default=float))
