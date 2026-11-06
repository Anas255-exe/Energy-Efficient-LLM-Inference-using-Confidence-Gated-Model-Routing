# GPU replication (4 Nov 2026, Kaggle 2x T4, fp16, `src/analyze_pairs.py results_kaggle`)

**Setup:** Qwen2.5 0.5B/1.5B/3B/7B-Instruct, 3,000 questions (1,000 each from TriviaQA, SciQ and NQ-open). Probes are fit on the first 70%, and everything is reported on the last 900 (test). Semantic entropy uses K=10 samples at T=1. Energy is NVML gross (both GPUs) measured per query on 100 real batch-1 test queries.

**Headline: the M2 negative result replicates on GPU and across 5 model pairs.** Keeping ≥95% of the large model's accuracy is energy-negative for every real gate on every pair except 3B→7B with entropy (+4%). Even an oracle saves at most 19%. Semantic entropy detects errors best, significantly so for the 0.5B and 1.5B drafts, but charging its 10 samples makes the cascade cost 4-10x as much as always using the large model. The probe is indistinguishable from token entropy.

| Pair | Acc draft / large | J/query draft / large | Break-even | AUROC entropy / SE / probe | Saving @95% (entropy / probe / oracle) | Best saving @90% (entropy / probe) |
|---|---|---|---|---|---|---|
| 0.5b → 7b | 17.0% / 44.3% | 12.1 / 38.8 | 69% | 0.744 / 0.818 / 0.762 | -26% / -21% / -6% | -16% / -16% |
| 1.5b → 7b | 28.6% / 44.3% | 16.1 / 38.8 | 58% | 0.808 / 0.836 / 0.836 | -22% / -12% / +3% | -2% / -2% |
| 3b → 7b | 34.6% / 44.3% | 17.9 / 38.8 | 54% | 0.802 / 0.814 / 0.812 | +4% / -1% / +19% | +14% / +19% |
| 0.5b → 3b | 17.0% / 34.6% | 12.1 / 17.9 | 32% | 0.744 / 0.818 / 0.762 | -58% / -58% / -33% | -43% / -43% |
| 1.5b → 3b | 28.6% / 34.6% | 16.1 / 17.9 | 10% | 0.808 / 0.836 / 0.836 | -40% / -25% / -10% | -15% / -10% |

| Draft | ΔAUROC SE − entropy [95% CI] | ΔAUROC probe − entropy [95% CI] | SE saving @95% (10 samples charged) |
|---|---|---|---|
| 0.5b | +0.074 [+0.026, +0.119] | +0.019 [-0.038, +0.073] | -333% |
| 1.5b | +0.029 [+0.005, +0.055] | +0.029 [-0.011, +0.065] | -432% |
| 3b | +0.012 [-0.010, +0.035] | +0.009 [-0.026, +0.045] | -466% |

| Model | Test acc | Gross J/query (100 real queries) | Linear-fit validate error |
|---|---|---|---|
| 0.5b | 17.0% | 12.1 | -17% |
| 1.5b | 28.6% | 16.1 | -12% |
| 3b | 34.6% | 17.9 | +3% |
| 7b | 44.3% | 38.8 | -39% |

**Caveats.**
- Idle power was read on a hot GPU (73-76 °C, 44-69 W vs 19.9 W true idle), so `J_net` is unreliable: the 0.5B linear fit misses by -87%. Gross J is therefore used.
- Gross J charges small models for an idle second T4. Subtracting the true idle (19.9 W × duration) moves break-even to 77/63/59% (→7B) and 43/9% (→3B). The conclusions are unchanged.
- The linear token model misses the 7B by -39% on real queries (short answers, two-GPU split), so per-query measured means are used instead.
- The semantic-entropy cost is an upper bound (10 sequential draft calls). Batched sampling would be cheaper but was not measured.

# M2 pilot results (2 Nov 2026, base Apple M2 16 GB, MLX 4-bit)

**Setup:** Qwen2.5-3B (draft) → Qwen2.5-7B (large), with 1,000 questions (600 TriviaQA + 400 SciQ). Probes are fit on the first 700 questions, and everything is reported on the last 300 (test). The energy basis is `results/cost.jsonl`: 3B, 7B and 3B-prefill run interleaved per test question, measuring CPU+GPU+ANE+DRAM with idle power subtracted.

## Headline: a negative result, scoped to this setup
On this hardware and model pair, a hallucination-gated cascade saves energy only if you accept accuracy loss. Keeping ≥95% of the 7B's accuracy costs **more** energy than always using the 7B with every real gate tested. Only an oracle gate gets there, saving about 10%.

## Numbers (test, n=300)
| Quantity | Value |
|---|---|
| Accuracy, 3B / 7B (lenient; strict) | 35.0% / 51.7% (33.3% / 49.0%) |
| Energy per query, 3B / 7B (net; gross) | 1.13 J / 2.58 J (2.77 / 6.37 J) |
| 3B prefill-only energy | 0.77 J (68% of a 3B answer) |
| Break-even escalation: cascade / pre-generation router | 56% / 65% |
| Probe AUROC (answer token, layer 22, CV-selected) | 0.806 [0.75, 0.86] |
| Entropy / log-prob AUROC | 0.804 [0.75, 0.86] / 0.791 [0.74, 0.84] |
| ΔAUROC probe − entropy | +0.002 [−0.060, +0.057], **no difference** |
| ΔAURC probe − best baseline (accuracy vs escalation) | −0.002 [−0.019, +0.015], **no difference** |
| Pre-generation (prompt) probe AUROC | 0.727 [0.67, 0.78] |
| Escalation needed for 95% of 7B accuracy | probe 70% (−12% energy, i.e. costs more), oracle 45% (+10% saving) |
| TriviaQA→SciQ probe transfer | AUROC 0.80, similar to in-domain (0.78) and to entropy |
| Batch 8 vs 1 (CPU+GPU net) | −9% (3B), −6% (7B) energy per query; −31% (3B), −34% (7B) wall time |

### Operating points (probe cascade vs oracle cascade vs pre-generation router)
| Escalation | Probe: % of 7B acc / saving | Oracle | Router |
|---|---|---|---|
| 0% | 68% / +56% | 68% / +56% | 68% / +56% |
| 10% | 72% / +47% | 75% / +46% | 72% / +48% |
| 20% | 75% / +37% | 79% / +36% | 75% / +39% |
| 30% | 82% / +27% | 84% / +25% | 79% / +31% |
| 40% | 85% / +18% | 91% / +15% | 83% / +21% |
| 50% | 86% / +8% | 97% / +5% | 86% / +13% |


## Token-level energy model (`src/energy_model.py`, 2 Nov 2026, afternoon)
E_net ≈ c + a·prompt_tokens + b·output_tokens, fitted per model on a 4×4 grid (prompt 16–1024, output 1–128 tokens; 3 repeats). One model resident at a time (peak 4.5 GB for 7B), so it is safe on 16 GB.

| Model | J per output token (b) | J per prompt token (a) | R² |
|---|---|---|---|
| Qwen2.5-0.5B | 0.075 | 0.004 | 0.995 |
| Qwen2.5-1.5B | 0.181 | 0.018 | 0.948 |
| Qwen2.5-3B | 0.313 | 0.032 | 0.998 |
| Qwen2.5-7B | 0.700 | 0.076 | 0.962 |

- **Scaling:** J/output token ∝ N^0.81; J/prompt token ∝ N^1.07. A decoded token costs ~10× a prompt token.
- **Held-out check** (40 real chat queries per model): the model predicts 7B within −6% (per-query error 10%) and 3B within −20% (it misses fixed per-call overhead on short queries).
- **Cascade ratio:** measured 3B/7B energy = 0.41 (model 0.35; earlier cost run 0.44), so break-even escalation stays ~59% and the headline holds.
- **Correction to the cost run:** absolute J in `cost.jsonl` is ~3× too low. That run went 2.6× slower at ~3.6 W, and subtracting idle × time removed most of it. Ratios still agree, so the conclusions stand; use the token model for absolute J.
- Caveats: the GPU reached 87–94 °C during 3B/7B blocks (cool-down to 60 °C only before each block). Run on battery, with no wall meter.

## What this means for the research questions
- **RQ1:** a hidden-state probe predicts draft errors well (AUROC 0.81), but no better than free token-entropy.
- **RQ2:** no measurable gain from the probe at n=300. Gaps smaller than about 0.06 AUROC are undetectable at this size.
- **RQ3:** the break-even point is high (56%) because the 3B already costs 44% of the 7B. With short answers, prefill and per-call overhead dominate. The energy saving only shows up in the low-accuracy region.
- **RQ4 (lite):** batching barely changes per-query energy on the M2 at this scale.
- **Router vs cascade:** the router saves the remaining 32% of draft energy on escalated queries, but its prompt-only probe ranks worse. Net, it is slightly better below 50% escalation and similar above.

## Caveats (from an independent review)
- The sequential runs drifted: 3B energy rose 1.07 → 1.67 J over its run, and SMC idle moved between 2.3 and 4.9 W. That is why the interleaved cost run is the basis. Its first ~75 questions were inflated by warm-up and by the Mac being in use (medians: 0.93 J / 2.21 J).
- Power comes from macmon, not validated with a wall meter. The run used the Mac while it was also in use. Fix: run unattended, repeat 3×, check idle before and after.
- The lenient (plural-tolerant) matching was decided after the first look at the results, so both strict and lenient are reported.
- One machine, one model pair, one seed, n=300. The probe's CV AUROC (0.855) is above its test AUROC (0.806): selecting the best of 36 layer/position options flatters the CV score.

## Next experiments (highest value first)
1. ~~Token-level energy model~~: done (see above).
2. **Clean unattended re-run** of `src/cost.py` (3 repeats, `caffeinate`, apps closed, charger at 100%).
3. **A bigger accuracy gap** (a 1B draft, or a 70B target on a DGX Spark). Savings at high reliability need the draft to be right much more often relative to its cost.
4. **Larger n** (≥3,000 questions) plus HaluEval-QA, and semantic-entropy labels on Kaggle.
5. **Longer answers** (explanations, not short phrases), where decode dominates and the energy gap between models widens.
