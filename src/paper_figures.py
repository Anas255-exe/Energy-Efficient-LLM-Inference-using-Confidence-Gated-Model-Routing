"""Paper figures (static PNG, print) -> paper/figures/: architecture, energy scaling, four-cell model, gate hardness,
retention-vs-energy curves. Phase diagram and scale map come from phase.py / scale_map.py.
Palette: validated categorical slots (blue, orange, aqua, yellow); identity is also carried by direct labels.
usage: python src/paper_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, PANEL = "#1a1a19", "#5f5e58", "#e4e3df", "#f0efec"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})
load = lambda f: [json.loads(l) for l in open(f)]


def box(ax, x, y, w, h, text, fc="#ffffff", ec=MUTED, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal",
            linespacing=1.35)


def arrow(ax, a, b, label=None, lx=0, ly=0):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3, shrinkA=0, shrinkB=0))
    if label:
        ax.text((a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly, label, fontsize=8, color=INK, ha="center")


# --- Figure 1: architecture
fig, ax = plt.subplots(figsize=(9.5, 4.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 45); ax.axis("off")
box(ax, 1, 27, 11, 8, "Query q")
box(ax, 17, 24, 23, 14, "Draft model $S$\n(small LLM)\nanswer $\\hat{a}_S$ and token entropy", fc="#cde2fb", ec=BLUE)
box(ax, 46, 24, 17, 14, "Gate\nrisk score of $\\hat{a}_S$;\nin top fraction $r$?", fc=PANEL)
box(ax, 70, 33, 27, 8, "no (share $1-r$): return $\\hat{a}_S$")
box(ax, 70, 16, 27, 12, "yes (share $r$): Target model $L$\n(large LLM) → return $\\hat{a}_L$", fc="#fbe0d3", ec=ORANGE)
arrow(ax, (12.4, 31), (16.6, 31)); arrow(ax, (40.4, 31), (45.6, 31))
arrow(ax, (63.4, 33), (69.6, 37)); arrow(ax, (63.4, 29), (69.6, 22))
ax.add_patch(FancyBboxPatch((1, 1.5), 96, 9.5, boxstyle="round,pad=0.4,rounding_size=1.2", fc=PANEL, ec="none"))
ax.text(3, 8.3, "Energy meter (NVML on T4, macmon on M2), per query at batch 1:", fontsize=8.5, color=INK, fontweight="bold")
ax.text(3, 3.6, "$E_S$ paid on every query (wasted when escalated)  +  $E_L$ paid only when escalated   ⇒   "
        "$E\\,/\\,E_L = \\rho + r$;  saves energy iff $r < 1 - \\rho$", fontsize=8.5, color=INK)
ax.plot([28.5, 28.5], [11.5, 23.5], ls=":", c=BLUE, lw=1.2); ax.plot([83.5, 83.5], [11.5, 15.5], ls=":", c=ORANGE, lw=1.2)
ax.text(50, 43.5, "Answers, risk scores and energies are logged once per model; every escalation rate r is then evaluated by offline replay.",
        fontsize=8, color=MUTED, ha="center", style="italic")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_architecture.png"); plt.close(fig)

# --- Figure 2: energy scaling (measured J/query, batch 1, NVML gross)
P = json.load(open("results_kaggle/pairs_summary.json"))["energy_model"]
B = json.load(open("results_big/big_summary.json"))
N = {"0.5b": 0.49, "1.5b": 1.54, "3b": 3.09, "7b": 7.62, "14b": 14.7, "32b": 32.5}
fp16 = [(N[m], P[m]["validate_measured_J"]) for m in ("0.5b", "1.5b", "3b", "7b")]
q4 = [(N[m], B["energy_C"][m]["J"]) for m in ("7b", "14b", "32b")]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for pts, c, lab in ((fp16, BLUE, "fp16 (0.5B–7B)"), (q4, ORANGE, "4-bit bnb (7B–32B)")):
    x, y = map(np.array, zip(*pts))
    ax.plot(x, y, c=c, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=lab)
    for i in range(len(x) - 1):
        beta = np.log(y[i + 1] / y[i]) / np.log(x[i + 1] / x[i])
        ax.text(np.sqrt(x[i] * x[i + 1]), np.sqrt(y[i] * y[i + 1]) * (0.78 if c == BLUE else 1.22), f"β={beta:.2f}",
                fontsize=7.5, color=INK, ha="center")
    ax.text(x[-1] * 1.12, y[-1], lab, fontsize=8, color=INK, va="center")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.35, 120)
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
ax.set_yticks([10, 20, 50, 100, 200, 400]); ax.set_yticklabels(["10", "20", "50", "100", "200", "400"])
ax.set_xlabel("parameters (billions, log)"); ax.set_ylabel("measured J / query (log)")
ax.grid(True, which="major", color=GRID, lw=0.6); ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_energy_scaling.png"); plt.close(fig)

# --- Figure 3: four-cell model
PH = json.load(open("results_kaggle/phase_summary.json"))
bg = PH["background"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
a1.set_xlim(0, 2); a1.set_ylim(0, 2.25); a1.axis("off")
shares = {}
for k in ("GPU 0.5b->7b", "4bit 14b->32b"):
    v = PH["pairs"][k]; b = v["coverage"] * v["acc_l"]
    shares[k] = dict(both_right=b, harmed=v["acc_s"] - b, rescuable=v["acc_l"] - b, both_wrong=1 - v["acc_s"] - v["acc_l"] + b)
cells = [("both_right", 0, 1, "both right", BLUE, "#cde2fb"), ("harmed", 1, 1, "harmed", YELLOW, "#fbe7bd"),
         ("rescuable", 0, 0, "rescuable", AQUA, "#c3eedb"), ("both_wrong", 1, 0, "both wrong", ORANGE, "#fbe0d3")]
for key, cx, cy, name, ec, fc in cells:
    a1.add_patch(FancyBboxPatch((cx + 0.04, cy + 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=1.4))
    s0, s1 = shares["GPU 0.5b->7b"][key], shares["4bit 14b->32b"][key]
    a1.text(cx + 0.5, cy + 0.62, name, ha="center", fontsize=10, fontweight="bold", color=INK)
    a1.text(cx + 0.5, cy + 0.32, f"0.5B→7B: {s0:.0%}\n14B→32B: {s1:.0%}", ha="center", va="center", fontsize=8, color=INK)
a1.text(0.5, 2.05, "target right", ha="center", fontsize=9, color=MUTED); a1.text(1.5, 2.05, "target wrong", ha="center", fontsize=9, color=MUTED)
a1.text(-0.04, 1.5, "draft\nright", ha="right", va="center", fontsize=9, color=MUTED); a1.text(-0.04, 0.5, "draft\nwrong", ha="right", va="center", fontsize=9, color=MUTED)
x = np.linspace(-3.5, 5, 400)
d = {k: np.sqrt(2) * norm.ppf(bg[f"auroc_{k}"]) for k in ("rescue", "bothwrong", "harmed")}
for lab, mu, c in (("both right", 0.0, BLUE), ("harmed", d["harmed"], YELLOW), ("rescuable", d["rescue"], AQUA), ("both wrong", d["bothwrong"], ORANGE)):
    a2.plot(x, norm.pdf(x, mu), c=c, lw=2)
    a2.text(mu, norm.pdf(0) + {"both right": 0.012, "harmed": 0.045, "rescuable": 0.078, "both wrong": 0.111}[lab],
            f"{lab}", ha="center", fontsize=8, color=INK)
t = 2.0
a2.axvline(t, c=INK, lw=1, ls="--"); a2.axvspan(t, 5, color=PANEL, zorder=0)
a2.text(t + 0.15, 0.03, "escalated\n(top r)", fontsize=8, color=INK)
a2.set_yticks([]); a2.spines["left"].set_visible(False); a2.set_ylim(0, 0.56)
a2.set_xlabel("gate risk score (standardised; higher = escalate first)")
a2.set_title("Score per cell: mean measured AUROCs over 9 pairs", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_four_cell.png"); plt.close(fig)

# --- Figure 4: gates track hardness (cell AUROCs vs "both right"), dumbbell per pair
names = list(PH["pairs"])
lab = lambda k: (k.replace("GPU ", "T4 fp16 ").replace("4bit ", "T4 4-bit ").replace("->", "→")
                 .replace("5b", "5B").replace("3b", "3B").replace("7b", "7B").replace("14b", "14B").replace("32b", "32B"))
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for i, k in enumerate(names):
    v = PH["pairs"][k]; y = len(names) - 1 - i
    ax.plot([v["auroc_rescue"], v["auroc_bothwrong"]], [y, y], c=GRID, lw=3, zorder=1)
    ax.scatter(v["auroc_rescue"], y, s=46, c=AQUA, edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
    ax.scatter(v["auroc_bothwrong"], y, s=70, facecolors="#fcfcfb", edgecolors=ORANGE, linewidths=2, zorder=2)
top = PH["pairs"][names[0]]
ax.text(top["auroc_rescue"], len(names) - 0.45, "rescuable", fontsize=8, color=INK, ha="center")
ax.text(top["auroc_bothwrong"], len(names) - 0.45, "both wrong", fontsize=8, color=INK, ha="left")
ax.set_yticks(range(len(names))); ax.set_yticklabels([lab(k) for k in reversed(names)], fontsize=8)
ax.set_xlim(0.70, 0.88); ax.set_ylim(-0.6, len(names) + 0.1)
ax.set_xlabel("token-entropy AUROC against the \"both right\" cell")
ax.grid(True, axis="x", color=GRID, lw=0.6)
ax.scatter([], [], c=AQUA, label="rescuable (escalation helps)"); ax.scatter([], [], facecolors="#fcfcfb", edgecolors=ORANGE, linewidths=2, label="both wrong (escalation wasted)")
ax.legend(loc="lower right", fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_hardness.png"); plt.close(fig)

# --- Figure 5: retention vs energy (entropy gate, every escalation count)
def curve(cs, cl, ent, rho):
    o, n = np.argsort(-ent, kind="stable"), len(cs)
    acc = (cs.sum() - np.r_[0, np.cumsum(cs[o])] + np.r_[0, np.cumsum(cl[o])]) / n
    return rho + np.arange(n + 1) / n, acc / cl.mean()


R = {m: load(f"results_kaggle/{m}.jsonl") for m in ("0.5b", "3b", "7b")}
te = np.arange(len(R["7b"])) >= 0.7 * len(R["7b"])
col = lambda rows, k, sel=None: np.array([r[k] for r in rows], float)[sel if sel is not None else slice(None)]
RB = {m: load(f"results_big/{d}/{m}.jsonl") for m, d in (("7b", "a"), ("14b", "a"), ("32b", "b"))}
nb = len(RB["32b"])
series = []
for s, c in (("0.5b", BLUE), ("3b", ORANGE)):
    v = json.load(open("results_kaggle/pairs_summary.json"))["pairs"][f"{s}->7b"]
    series.append((f"T4 fp16 {s.upper()}→7B", c, *curve(col(R[s], "correct", te), col(R["7b"], "correct", te), col(R[s], "ent_mean", te), v["J_small"] / v["J_large"])))
series.append(("T4 4-bit 7B→14B", AQUA, *curve(col(RB["7b"], "correct"), col(RB["14b"], "correct"), col(RB["7b"], "ent_mean"), B["pairs"]["7b->14b"]["rho"]["measured_gross"])))
series.append(("T4 4-bit 14B→32B", YELLOW, *curve(col(RB["14b"], "correct")[:nb], col(RB["32b"], "correct"), col(RB["14b"], "ent_mean")[:nb], B["pairs"]["14b->32b"]["rho"]["C_gross"])))
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.axhline(0.95, c=MUTED, lw=0.8, ls="--"); ax.text(1.53, 0.935, "95% retention", fontsize=7.5, color=MUTED, ha="right")
ax.axvline(1.0, c=MUTED, lw=0.8, ls=":"); ax.text(1.02, 0.40, "always-large energy", fontsize=7.5, color=MUTED)
for name, c, e, ret in series:
    ax.plot(e, ret, c=c, lw=2, label=name)
    k = np.argmax(ret >= 0.95 - 1e-12)
    ax.scatter(e[k], ret[k], s=40, c=c, edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
    ax.text(e[0] - 0.02, ret[0], name, fontsize=7.5, color=INK, ha="right", va="center")
ax.set_xlim(-0.45, 1.55); ax.set_ylim(0.35, 1.12)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
ax.set_xlabel("cascade energy relative to always-large (ρ + r)"); ax.set_ylabel("accuracy retained vs always-large")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.grid(True, color=GRID, lw=0.6)
ax.legend(loc="upper left", fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_curves.png"); plt.close(fig)
print("figures ->", sorted(os.listdir(OUT)))
