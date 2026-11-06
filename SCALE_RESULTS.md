# Scale results (5 Nov 2026): the gate starts paying from 7B→14B up

## 1. Measured on Kaggle 2x T4: Qwen2.5 7B / 14B / 32B, 4-bit, open QA (same 3,000 questions as the 0.5B–7B run)
Source: `results_big/`, `src/analyze_big.py` → `results_big/big_summary.json`. Gate = token entropy (free). Nothing is fit, so all answered questions are used.

| Pair | n | acc small → large | coverage | ρ (energy ratio) | saving @95% [95% CI] | saving @99% | oracle @95% |
|---|---|---|---|---|---|---|---|
| 7B→14B | 3000 | 43.6% → 50.2% | 0.743 | **0.53 measured** | **+24%** [+12, +32] | −3% [−30, +17] | +29% |
| 7B→32B | 2664 | 43.6% → 52.8% | 0.726 | **0.22 measured** | **+39%** [+28, +50] | 0% [−9, +16] | +53% |
| 14B→32B | 2664 | 49.8% → 52.8% | 0.834 | **0.41 measured** | **+55%** [+41, +59] | +27% [−4, +49] | +55% |

- Energy, measured (gross NVML J/query, 100 real batch-1 queries): 7B 69.5 J, 14B 129.9 J. Net of cold idle (20.3 W): 50.7 J and 95.2 J. Both give ρ = 0.53.
- 32B energy: run B ran out of Kaggle time before its energy step (2,664 of 3,000 answers done). Run C (`results_big/c`, energy only) re-measured 32B, 14B and 7B on the same 100 queries in one session:
  - Gross J/query: 7B 84.4, 14B 157.9, 32B 385.5. Net of cold idle (19.9 W): 66.5, 124.2, 315.5.
  - ρ is the run-C ratio; the projection (0.47 for 14B→32B) was conservative.
  - Run C read ~22% higher absolute J than run A (hotter GPUs), but the 7B/14B ratio is the same: 0.53 in both. Only ratios are used.
  - 32B is split across both T4s (`device_map="auto"`); 7B and 14B use one GPU, with the second idle but still metered. The measured exponent is ~1.1 for 14B→32B and 0.95 for 7B→14B.
- The gate still tracks question difficulty: AUROC for spotting the draft's errors is 0.76–0.78, but for spotting the questions the large model can fix it is only 0.63–0.65.
- Caveats:
  - 4-bit costs more energy than fp16 at batch 1: 7B is 69 J at 4-bit against 39 J at fp16, a bitsandbytes dequant overhead. Only ratios are compared.
  - 14B→32B @95% escalates only 4% of queries, because 14B alone already reaches 94% of 32B's accuracy. Report 99% retention next to it.

## 2. Projected for 11 models up to 72B: Open LLM Leaderboard, MMLU-Pro, 12,032 questions
Source: `results_olb/mmlu_pro_min.json`, `src/scale_map.py` → `results_olb/scale_summary.json`, `scale_map.png`. Gate = 1 − max option probability. ρ = (N_s/N_l)^β, with β = 0.86–1.13: the range of per-query exponents measured for drafts ≥ 3B (T4 fp16 3B→7B 0.86, M2 0.91, 4-bit 0.95 / 1.05 / 1.13). Corrected 5 Nov: the earlier β = 0.77 had no measured source. Drafts < 3B measured lower on T4 (0.42–0.55), so their projections are optimistic.

| Pair | acc small → large | saving @95% (β 0.86 / 1.13) | saving @99% (β 0.86) |
|---|---|---|---|
| Qwen 0.5B→7B | 17.2 → 42.9% | −1% / +3% | −7% |
| Qwen 7B→14B | 42.9 → 49.0% | −1% / +8% | −29% |
| Qwen 7B→72B | 42.9 → 56.3% | +24% / +30% | +3% |
| Qwen 14B→72B | 49.0 → 56.3% | **+39% / +48%** | +9% |
| Qwen 32B→72B | 56.7 → 56.3% | +50% / +60% (32B already ≥ 72B) | +50% |
| Llama 8B→70B (cross-size, the "universal" angle) | 38.0 → 53.1% | +11% / +17% | −3% |

- The pattern holds: savings rise as both models grow and the accuracy gap shrinks. Small pairs lose energy.
- The task matters:
  - 7B→14B saves +24% on open QA (measured) but only −1% to +8% on MMLU-Pro.
  - Open-QA coverage is higher (0.74 vs 0.69) and so is gate AUROC (0.78 vs 0.75).

## Proof
| Number | Source | Independent cross-check |
|---|---|---|
| 7B→14B +23.9% @95% | `analyze_big.py` (vectorised replay) | Brute-force loop: same k = 679 escalations and +23.9% |
| 32B / 14B / 7B J/query (run C) | `analyze_big.py` | awk over the raw files: 385.47 / 157.89 / 84.35; power 110 / 93 / 94 W, plausible for 2x T4 (70 W TDP each) |
| ρ reproducible | Run A vs run C | 7B/14B = 0.53 both runs, savings @95% equal (+23.9%) |
| 4-bit vs fp16 7B accuracy | `results_big` vs `results_kaggle` | 43.6% vs 44.4%; 92.7% per-question agreement |
| All 11 MMLU-Pro accuracies | Our per-question pull | Match leaderboard published "MMLU-PRO Raw" exactly (`results_olb/leaderboard_crosscheck.json`) |
| Question alignment | `results_big` vs `results_kaggle` | Same ids, same order, same answer sets (only alias order differs) |

**Not verified:**
- 32B pairs use 2,664 answered questions (Kaggle time limit), not 3,000.
- Big-model energy at batched serving.
- MMLU-Pro energy uses a prefill-dominated β; the open-QA one is measured.
