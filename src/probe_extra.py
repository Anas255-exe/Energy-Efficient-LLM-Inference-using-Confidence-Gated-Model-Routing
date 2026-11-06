"""Semantic-entropy probe (SEP) and transfer (RQ5) -> results_eval/probe_transfer.json, results_eval/PROBE_TRANSFER.md

1. SEP (Kossen et al. [11]): a linear probe on one forward pass's hidden state, trained to predict *high semantic entropy*
   (SE binarised at the split that minimises within-group squared error, fit split only), scored on draft *errors*.
   Layer/position by 5-fold CV on the fit split (first 70%); reported on the test split (last 30%), as analyze_pairs.py.
2. Cross-dataset transfer: correctness probe and SEP trained on the fit rows of two sources, tested on the third source's
   test rows, against the same probe trained on all sources (in-domain) and token entropy (needs no training).
3. Threshold transfer: entropy threshold t set for 95% retention on one set of questions, applied unchanged to another
   (held-out source, or another model pair). Energy = mean J/query per model, as in eval_extra.py.
usage: python src/probe_extra.py
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

load = lambda f: [json.loads(l) for l in open(f)]
meanJ = lambda f: float(np.mean([r["J"] for r in load(f)]))
probe = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
rng = np.random.default_rng(0)
D = "results_kaggle"
qs = load(f"{D}/questions.jsonl"); n = len(qs)
fit, te = np.arange(n) < 0.7 * n, np.arange(n) >= 0.7 * n
src = np.array([q["dataset"] for q in qs]); SRC = ("triviaqa", "sciq", "nq")
PS = json.load(open(f"{D}/pairs_summary.json"))
out = dict(sep={}, dataset_transfer={}, threshold_transfer={})


def binarise(x):
    """Kossen et al.: threshold minimising the summed within-group squared error."""
    xs = np.sort(x)
    sse = [xs[:i].var() * i + xs[i:].var() * (len(xs) - i) for i in range(1, len(xs))]
    return xs[int(np.argmin(sse)) + 1]


def at95(score, cs, cl, Es, El, ii):
    """21-point grid, as in the paper: saving at the first escalation rate keeping 95% of always-large accuracy."""
    o = np.argsort(-score[ii], kind="stable")
    for r in np.linspace(0, 1, 21):
        esc = np.zeros(len(ii), bool); esc[o[:int(round(r * len(ii)))]] = True
        if np.where(esc, cl[ii], cs[ii]).mean() >= 0.95 * cl[ii].mean():
            return dict(escalation=r, saving=1 - (Es + r * El) / El)


cl7 = np.array([r["correct"] for r in load(f"{D}/7b.jsonl")], float)
E7 = meanJ(f"{D}/energy_validate_7b.jsonl")
for d in ("0.5b", "1.5b", "3b"):
    R = load(f"{D}/{d}.jsonl")
    cs = np.array([r["correct"] for r in R], float); y = (1 - cs).astype(int)
    ent = np.array([r["ent_mean"] for r in R])
    se = {r["id"]: r["se"] for r in load(f"{D}/{d}_se.jsonl")}; se = np.array([se[q["id"]] for q in qs])
    hi = (se >= binarise(se[fit])).astype(int)
    H = np.load(f"{D}/{d}_hidden.npy", mmap_mode="r")  # [n, layers (every 2nd), 2 (prompt, answer), dim]
    X = lambda l, p: np.asarray(H[:, l, p], np.float32)
    cv = {(l, p): cross_val_score(probe(), X(l, p)[fit], hi[fit], cv=5, scoring="roc_auc").mean()
          for p in (0, 1) for l in range(1, H.shape[1])}
    bl, bp = max(cv, key=cv.get)
    sep = probe().fit(X(bl, bp)[fit], hi[fit]).predict_proba(X(bl, bp))[:, 1]
    ii = np.where(te)[0]
    dd = [roc_auc_score(y[b], sep[b]) - roc_auc_score(y[b], ent[b]) for b in (rng.choice(ii, len(ii)) for _ in range(500))]
    g = PS["pairs"][f"{d}->7b"]["gates"]
    out["sep"][d] = dict(layer_idx=int(bl), position="answer" if bp else "prompt", cv_auroc_high_se=cv[(bl, bp)],
                         se_threshold=float(binarise(se[fit])), auroc_high_se_test=roc_auc_score(hi[te], sep[te]),
                         auroc_errors=dict(sep=roc_auc_score(y[te], sep[te]), entropy=g["entropy_mean"]["auroc"],
                                           semantic_entropy=g["semantic_entropy"]["auroc"], correctness_probe=g["probe_answer"]["auroc"]),
                         d_auroc_sep_vs_entropy=dict(mean=float(np.mean(dd)), ci=list(np.percentile(dd, [2.5, 97.5]))),
                         saving95_to_7b=dict(sep=at95(sep, cs, cl7, meanJ(f"{D}/energy_validate_{d}.jsonl"), E7, ii),
                                             semantic_entropy_10_samples=g["semantic_entropy"]["at95"], entropy=g["entropy_mean"]["at95"]))

    # cross-dataset transfer: same layer as the paper's correctness probe / the SEP layer above
    la = PS["probe_layers"][d]["answer"]["layer_idx"]
    Xa, Xs = X(la, 1), X(bl, bp)
    full_c = probe().fit(Xa[fit], y[fit]).predict_proba(Xa)[:, 1]
    for h in SRC:
        tr, tt = fit & (src != h), te & (src == h)
        out["dataset_transfer"].setdefault(d, {})[h] = dict(
            n_test=int(tt.sum()), entropy=roc_auc_score(y[tt], ent[tt]),
            probe_in_domain=roc_auc_score(y[tt], full_c[tt]),
            probe_transfer=roc_auc_score(y[tt], probe().fit(Xa[tr], y[tr]).predict_proba(Xa[tt])[:, 1]),
            sep_in_domain=roc_auc_score(y[tt], sep[tt]),
            sep_transfer=roc_auc_score(y[tt], probe().fit(Xs[tr], hi[tr]).predict_proba(Xs[tt])[:, 1]))
    print(d, json.dumps(out["sep"][d], default=float)[:400], flush=True)


# --- threshold transfer on every question (entropy needs no training)
def threshold(cs, cl, ent, R=0.95):
    o = np.argsort(-ent, kind="stable"); m = len(cs)
    acc = (cs.sum() - np.r_[0, np.cumsum(cs[o])] + np.r_[0, np.cumsum(cl[o])]) / m
    k = int(np.argmax(acc >= R * cl.mean() - 1e-12))
    return np.inf if k == 0 else ent[o[k - 1]]


def apply(t, cs, cl, ent, Es, El):
    esc = ent >= t
    return dict(escalate=esc.mean(), retention=np.where(esc, cl, cs).mean() / cl.mean(), saving=1 - (Es + esc.mean() * El) / El)


B = "results_big"
RB = {"7b": load(f"{B}/a/7b.jsonl"), "14b": load(f"{B}/a/14b.jsonl"), "32b": load(f"{B}/b/32b.jsonl")}
EB = {"7b->14b": (meanJ(f"{B}/a/energy_validate_7b.jsonl"), meanJ(f"{B}/a/energy_validate_14b.jsonl")),
      "7b->32b": (meanJ(f"{B}/c/energy_validate_7b.jsonl"), meanJ(f"{B}/c/energy_validate_32b.jsonl")),
      "14b->32b": (meanJ(f"{B}/c/energy_validate_14b.jsonl"), meanJ(f"{B}/c/energy_validate_32b.jsonl"))}
P = {}
for k, (Es, El) in EB.items():
    a, b = k.split("->"); m = min(len(RB[a]), len(RB[b]))
    c = lambda x: np.array([r["correct"] for r in RB[x][:m]], float)
    P[k] = dict(cs=c(a), cl=c(b), ent=np.array([r["ent_mean"] for r in RB[a][:m]]),
                src=np.array([r["dataset"] for r in RB[a][:m]]), Es=Es, El=El)
P["3b->7b fp16"] = dict(cs=np.array([r["correct"] for r in load(f"{D}/3b.jsonl")], float), cl=cl7,
                        ent=np.array([r["ent_mean"] for r in load(f"{D}/3b.jsonl")]), src=src,
                        Es=meanJ(f"{D}/energy_validate_3b.jsonl"), El=E7)
for k, p in P.items():
    keys = ("cs", "cl", "ent")
    for h in SRC:
        cal, ev = p["src"] != h, p["src"] == h
        t = threshold(*(p[x][cal] for x in keys))
        out["threshold_transfer"].setdefault("held_out_source", {}).setdefault(k, {})[h] = dict(
            held_out=apply(t, *(p[x][ev] for x in keys), p["Es"], p["El"]),
            in_sample=apply(threshold(*(p[x][ev] for x in keys)), *(p[x][ev] for x in keys), p["Es"], p["El"]))
for a in EB:
    for b in EB:
        if a != b:
            t = threshold(*(P[a][x] for x in ("cs", "cl", "ent")))
            out["threshold_transfer"].setdefault("other_pair", {})[f"{a} => {b}"] = apply(t, *(P[b][x] for x in ("cs", "cl", "ent")), P[b]["Es"], P[b]["El"])

os.makedirs("results_eval", exist_ok=True)
json.dump(out, open("results_eval/probe_transfer.json", "w"), indent=1, default=float)

L = ["## Semantic-entropy probe (T4 fp16, test split, gate for →7B)", "",
     "| Draft | SEP layer | AUROC on errors: entropy / SE (10 samples) / correctness probe / SEP | ΔAUROC SEP − entropy [95% CI] | Saving @95%: entropy / SE / SEP |",
     "|---|---|---|---|---|"]
for d, s in out["sep"].items():
    a, v = s["auroc_errors"], s["saving95_to_7b"]
    L.append(f"| {d} | {s['position']} L{s['layer_idx']} | {a['entropy']:.3f} / {a['semantic_entropy']:.3f} / {a['correctness_probe']:.3f} / {a['sep']:.3f} "
             f"| {s['d_auroc_sep_vs_entropy']['mean']:+.3f} [{s['d_auroc_sep_vs_entropy']['ci'][0]:+.3f}, {s['d_auroc_sep_vs_entropy']['ci'][1]:+.3f}] "
             f"| {v['entropy']['saving']:+.0%} / {v['semantic_entropy_10_samples']['saving']:+.0%} / {v['sep']['saving']:+.0%} |")
L += ["", "## Cross-dataset transfer (AUROC on the held-out source's test rows)", "",
      "| Draft | Held-out source | n | Entropy | Probe in-domain → transfer | SEP in-domain → transfer |", "|---|---|---|---|---|---|"]
for d, x in out["dataset_transfer"].items():
    for h, s in x.items():
        L.append(f"| {d} | {h} | {s['n_test']} | {s['entropy']:.3f} | {s['probe_in_domain']:.3f} → {s['probe_transfer']:.3f} | {s['sep_in_domain']:.3f} → {s['sep_transfer']:.3f} |")
L += ["", "## Threshold transfer: entropy threshold set for 95% retention, applied unchanged", "",
      "| Pair | Held-out source | Retention achieved (target 95%) | Saving held-out vs in-sample |", "|---|---|---|---|"]
for k, x in out["threshold_transfer"]["held_out_source"].items():
    for h, s in x.items():
        L.append(f"| {k} | {h} | {s['held_out']['retention']:.1%} | {s['held_out']['saving']:+.0%} vs {s['in_sample']['saving']:+.0%} |")
L += ["", "| Threshold from → applied to | Retention achieved | Escalated | Saving |", "|---|---|---|---|"]
for k, s in out["threshold_transfer"]["other_pair"].items():
    L.append(f"| {k} | {s['retention']:.1%} | {s['escalate']:.0%} | {s['saving']:+.0%} |")
open("results_eval/PROBE_TRANSFER.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
