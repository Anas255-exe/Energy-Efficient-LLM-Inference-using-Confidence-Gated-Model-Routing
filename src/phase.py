"""Predictive model for when a confidence-gated cascade saves energy -> results_kaggle/phase_summary.json, phase_diagram.png

Inputs per pair (all observable without running a cascade):
  rho  = E_small / E_large (measured J/query; GPU gross, M2 net as in analyze.py)
  acc_s, acc_l, coverage = P(small right | large right), gate AUROC (token entropy; prefill probe for the router)
Model: binormal gate over the four joint-correctness cells. "Both right" scores ~ N(0,1); "rescuable" (small wrong, large
right), "harmed" (small right, large wrong) and "both wrong" ~ N(d_cell,1), d_cell = sqrt(2)*Phi^-1(AUROC of that cell vs both
right). Escalating the top-r fraction gives accuracy(r) in closed form. The independence model (score depends only on draft
correctness: d_rescuable = d_bothwrong from the plain draft-error AUROC, d_harmed = 0) is the special case; it is too
optimistic because uncertainty gates track question hardness, so they also rank "both wrong" and "harmed" queries high.
  cascade energy = rho + r      router energy (pre-generation, signal free, Jev-style) = (1 - r) * rho + r
  saving at retention R = 1 - energy at the smallest r with accuracy(r) >= R * acc_l;  break-even escalation = 1 - rho.
Validation: predicted vs measured replay on the 5 GPU pairs (results_kaggle), the M2 3B->7B pair (results/) and the 3 big
4-bit T4 pairs (results_big: 7B->14B, 7B->32B, 14B->32B), on the same 21-point escalation grid the replays use.
Phase diagram choice: since r(R) does not depend on rho, saving = 1 - rho - r(R; coverage), so the map is exact in rho.
Coverage alone does not fix acc_s, so the background holds acc_l = 0.443 (GPU 7B) and q = P(small right | large wrong) at
the mean of the measured pairs: acc_s = coverage*acc_l + q*(1 - acc_l). Points use their own predicted savings in labels.
usage: python src/phase.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

RATES = np.linspace(0, 1, 21)  # same grid as analyze.py / analyze_pairs.py
FINE = np.linspace(0, 1, 1001)
load = lambda f: [json.loads(l) for l in open(f)]


def cascade_acc(r, acc_s, acc_l, cov, au_r, au_b=None, au_h=0.5):
    """Expected accuracy when the top-r fraction by gate score escalates (AUROCs per cell vs both-right; au_b=None: = au_r)."""
    b = cov * acc_l
    P = (acc_l - b, acc_s - b, 1 - acc_s - acc_l + b, b)  # rescuable, harmed, both wrong, both right
    d = [np.sqrt(2) * norm.ppf(min(a, 1 - 1e-9)) for a in (au_r, au_h, au_r if au_b is None else au_b, 0.5)]
    out = []
    for x in np.atleast_1d(r):
        if x <= 0: out.append(acc_s); continue
        if x >= 1: out.append(acc_l); continue
        t = brentq(lambda t: sum(p * norm.sf(t - dd) for p, dd in zip(P, d)) - x, -50, 50)
        out.append(acc_s + P[0] * norm.sf(t - d[0]) - P[1] * norm.sf(t - d[1]))
    return np.array(out)


def saving_at(acc, acc_l, rates, R, energy):
    ok = np.where(acc / acc_l >= R - 1e-12)[0]
    return 1 - energy(rates[ok[0]])


def replay(cs, cl, score):
    order, n = np.argsort(-score, kind="stable"), len(cs)
    acc = []
    for r in RATES:
        esc = np.zeros(n, bool); esc[order[:int(round(r * n))]] = True
        acc.append(np.where(esc, cl, cs).mean())
    return np.array(acc)


# --- measured pairs: 5 GPU pairs (Kaggle T4, gross J) + M2 3B->7B (net J, interleaved cost run)
pairs, rng = {}, np.random.default_rng(0)
D = "results_kaggle"
qs = load(f"{D}/questions.jsonl"); n = len(qs); te = np.arange(n) >= 0.7 * n
R = {m: load(f"{D}/{m}.jsonl") for m in ("0.5b", "1.5b", "3b", "7b")}
summ = json.load(open(f"{D}/pairs_summary.json"))
for s, l in [("0.5b", "7b"), ("1.5b", "7b"), ("3b", "7b"), ("0.5b", "3b"), ("1.5b", "3b")]:
    v = summ["pairs"][f"{s}->{l}"]
    pairs[f"GPU {s}->{l}"] = dict(cs=np.array([r["correct"] for r in R[s]], float)[te], cl=np.array([r["correct"] for r in R[l]], float)[te],
                                  ent=np.array([r["ent_mean"] for r in R[s]])[te], rho=v["J_small"] / v["J_large"],
                                  auroc_router=v["gates"]["prefill_probe"]["auroc"], ref95=v["gates"]["entropy_mean"]["at95"]["saving"])
m2 = json.load(open("results/summary.json"))
s3, l7 = load("results/3b.jsonl"), {r["id"]: r for r in load("results/7b.jsonl")}
s3 = [r for r in s3 if r["id"] in l7]; te2 = np.arange(len(s3)) >= 0.7 * len(s3)
pairs["M2 3b->7b"] = dict(cs=np.array([r["correct"] for r in s3], float)[te2], cl=np.array([l7[r["id"]]["correct"] for r in s3], float)[te2],
                          ent=np.array([r["ent_mean"] for r in s3])[te2], rho=m2["J_small"] / m2["J_large"],
                          auroc_router=0.727, ref95=m2["gates"]["entropy_mean"]["at95"]["saving_vs_large"])  # router AUROC: RESULTS.md
# big 4-bit pairs (results_big): no probe was fit, so all answered questions are used; rho = gross J, 32B from energy-only run C
big = json.load(open("results_big/big_summary.json"))
RB = {m: load(f"results_big/{d}/{m}.jsonl") for m, d in (("7b", "a"), ("14b", "a"), ("32b", "b"))}
for s, l, rk in (("7b", "14b", "measured_gross"), ("7b", "32b", "C_gross"), ("14b", "32b", "C_gross")):
    nb = min(len(RB[s]), len(RB[l]))
    pairs[f"4bit {s}->{l}"] = dict(cs=np.array([r["correct"] for r in RB[s][:nb]], float), cl=np.array([r["correct"] for r in RB[l][:nb]], float),
                                  ent=np.array([r["ent_mean"] for r in RB[s][:nb]]), rho=big["pairs"][f"{s}->{l}"]["rho"][rk],
                                  auroc_router=None, ref95=None)

out = {}
for k, p in pairs.items():
    cs, cl, rho = p["cs"], p["cl"], p["rho"]
    acc_s, acc_l = cs.mean(), cl.mean()
    cov, q = (cs * cl).sum() / cl.sum(), (cs * (1 - cl)).sum() / (1 - cl).sum()
    e, resc, bw = p["ent"], (1 - cs) * cl, (1 - cs) * (1 - cl)
    auroc = roc_auc_score(1 - cs, e)
    br, hm = cs * cl, cs * (1 - cl)
    au_r, au_b, au_h = (roc_auc_score(c[c + br > 0], e[c + br > 0]) for c in (resc, bw, hm))  # each cell vs both-right
    cas = lambda r: rho + r
    row = dict(rho=rho, break_even=1 - rho, acc_s=acc_s, acc_l=acc_l, coverage=cov, q=q, auroc_entropy=auroc,
               auroc_rescue=au_r, auroc_bothwrong=au_b, auroc_harmed=au_h, auroc_rescue_vs_all=roc_auc_score(resc, e),
               auroc_router=p["auroc_router"])
    meas = {"entropy": replay(cs, cl, e), "oracle": replay(cs, cl, (1 - cs) + rng.random(len(cs)) * 1e-3)}
    models = {"entropy": cascade_acc(RATES, acc_s, acc_l, cov, au_r, au_b, au_h), "entropy_indep": cascade_acc(RATES, acc_s, acc_l, cov, auroc),
              "oracle": cascade_acc(RATES, acc_s, acc_l, cov, 1.0)}
    row["acc_curve_max_abs_err"] = float(np.abs(models["entropy"] - meas["entropy"]).max())
    row["acc_curve_max_abs_err_indep"] = float(np.abs(models["entropy_indep"] - meas["entropy"]).max())
    for g, pa in models.items():
        for Rr in (0.95, 0.90):
            row[f"{g}_{int(Rr * 100)}_measured"] = saving_at(meas[g.split("_")[0]], acc_l, RATES, Rr, cas)
            row[f"{g}_{int(Rr * 100)}_predicted"] = saving_at(pa, acc_l, RATES, Rr, cas)
    pr = cascade_acc(FINE, acc_s, acc_l, cov, p["auroc_router"]) if p["auroc_router"] else None  # big runs have no prefill probe
    for Rr in (0.95, 0.90):  # predicted only, independence model (an upper bound): no prefill-free router replay exists
        row[f"router_{int(Rr * 100)}_predicted_upper"] = np.nan if pr is None else saving_at(pr, acc_l, FINE, Rr, lambda r: (1 - r) * rho + r)
    # cross-check: this replay must reproduce the analysis scripts' own entropy at-95% saving (GPU: same constant energy)
    if k.startswith("GPU"):
        assert abs(row["entropy_95_measured"] - p["ref95"]) < 1e-9, (k, row["entropy_95_measured"], p["ref95"])
    row["entropy_95_reference"] = p["ref95"]
    out[k] = row

# cross-check 2: Monte Carlo of the three-class binormal gate agrees with the closed form
mc_n, a_s, a_l, cv, ar, ab, ah, r0 = 400_000, 0.35, 0.45, 0.6, 0.78, 0.82, 0.72, 0.4
b = cv * a_l
cell = rng.choice(4, mc_n, p=[b, a_s - b, a_l - b, 1 - a_s - a_l + b])  # both, only small, only large, neither
cs_mc, cl_mc = np.isin(cell, (0, 1)).astype(float), np.isin(cell, (0, 2)).astype(float)
sc = rng.normal(np.sqrt(2) * np.select([cell == 2, cell == 3, cell == 1], [norm.ppf(ar), norm.ppf(ab), norm.ppf(ah)], 0.0), 1)
k_esc = np.argsort(-sc)[:int(r0 * mc_n)]
acc_mc = cs_mc.copy(); acc_mc[k_esc] = cl_mc[k_esc]
assert abs(acc_mc.mean() - cascade_acc(r0, a_s, a_l, cv, ar, ab, ah)[0]) < 0.003, (acc_mc.mean(), cascade_acc(r0, a_s, a_l, cv, ar, ab, ah))

# --- phase diagram: saving at 95% retention over (rho, coverage)
ACC_L, Q = 0.443, float(np.mean([v["q"] for v in out.values()]))
AU_R, AU_B, AU_H = (float(np.mean([v[k] for v in out.values()])) for k in ("auroc_rescue", "auroc_bothwrong", "auroc_harmed"))
covs = np.linspace(0.0, 0.999, 100)


def r_needed(cov, au_r, au_b, au_h, Rr=0.95):
    acc_s = cov * ACC_L + Q * (1 - ACC_L)
    x, a = FINE[::5], cascade_acc(FINE[::5], acc_s, ACC_L, cov, au_r, au_b, au_h) / ACC_L - Rr  # 0.005 steps, then interpolate
    i = np.where(a >= -1e-12)[0][0]
    return x[i] if i == 0 else x[i - 1] + (x[i] - x[i - 1]) * -a[i - 1] / (a[i] - a[i - 1])


r80, rorc = np.array([r_needed(c, AU_R, AU_B, AU_H) for c in covs]), np.array([r_needed(c, 1.0, 1.0, 0.5) for c in covs])
rhos = np.linspace(0, 1, 200)
S = 1 - rhos[None, :] - r80[:, None]
fig, ax = plt.subplots(figsize=(6.4, 5))
cm = matplotlib.colors.LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
im = ax.pcolormesh(rhos, covs, S, cmap=cm, norm=matplotlib.colors.TwoSlopeNorm(0, -1, 1), shading="auto")
cs_ = ax.contour(rhos, covs, S, levels=[0.2, 0.4, 0.6], colors="#555", linewidths=0.8)
ax.clabel(cs_, fmt=lambda v: f"{v:+.0%}", fontsize=7)
ax.plot(1 - r80, covs, c="#1a1a19", lw=2, label="break-even, entropy gate")
ax.plot(1 - rorc, covs, c="#1a1a19", lw=1.5, ls="--", label="break-even, oracle gate")
for k, v in out.items():
    ax.scatter(v["rho"], v["coverage"], s=40, c="#1a1a19", marker="D" if k.startswith("4bit") else "o", edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
    dx, dy, ha = {"GPU 3b->7b": (8, 8, "left"), "GPU 1.5b->3b": (8, -12, "right"), "GPU 0.5b->3b": (-6, 7, "right"), "4bit 14b->32b": (8, 6, "left"),
                  "4bit 7b->32b": (0, 9, "center")}.get(k, (8, -10, "left"))
    name = (k.replace("GPU ", "T4 fp16 ").replace("4bit ", "T4 4-bit ").replace("->", "→").replace("5b", "5B").replace("3b", "3B")
            .replace("7b", "7B").replace("14b", "14B").replace("32b", "32B"))
    ax.annotate(f"{name}: {v['entropy_95_measured']:+.0%} / {v['entropy_95_predicted']:+.0%}", (v["rho"], v["coverage"]),
                xytext=(dx, dy), textcoords="offset points", fontsize=7, color="#1a1a19", ha=ha)
c95 = (0.95 * ACC_L - Q * (1 - ACC_L)) / ACC_L  # above this coverage the small model alone keeps 95%: saving = 1 - rho
ax.axhline(c95, c="#555", lw=0.8, ls=":")
ax.text(0.01, c95 + 0.01, "small model alone ≥ 95% of large", fontsize=6.5, color="#fcfcfb", ha="left")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("energy ratio ρ = E_small / E_large"); ax.set_ylabel("coverage = P(small right | large right)")
ax.set_title("Energy saving at 95% of large-model accuracy (labels: measured / model)", fontsize=9)
fig.colorbar(im, ax=ax, label="saving vs always-large (entropy gate, model)", format=matplotlib.ticker.PercentFormatter(1))
ax.legend(loc="lower right", fontsize=7, frameon=True)
ax.text(0.01, 0.99, f"background: acc_large = {ACC_L}, q = {Q:.2f}, mean measured cell AUROCs",
        transform=ax.transAxes, fontsize=6, color="#fcfcfb", va="top")
fig.tight_layout(); fig.savefig(f"{D}/phase_diagram.png", dpi=200)

json.dump(dict(model=__doc__.split("\n")[0], background=dict(acc_large=ACC_L, q=Q, auroc_rescue=AU_R, auroc_bothwrong=AU_B, auroc_harmed=AU_H), pairs=out), open(f"{D}/phase_summary.json", "w"), indent=2, default=float)
cols = ["rho", "coverage", "auroc_rescue_vs_all", "entropy_95_measured", "entropy_95_predicted", "entropy_indep_95_predicted",
        "entropy_90_measured", "entropy_90_predicted", "oracle_95_measured", "oracle_95_predicted", "acc_curve_max_abs_err", "acc_curve_max_abs_err_indep", "router_95_predicted_upper"]
print("pair".ljust(14), *[c.replace("_measured", "_m").replace("_predicted", "_p")[:12].rjust(12) for c in cols])
for k, v in out.items():
    print(k.ljust(14), *[f"{v[c]:12.3f}" for c in cols])
