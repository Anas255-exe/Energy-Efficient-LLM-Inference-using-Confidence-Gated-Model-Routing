# Energy-Efficient LLM Inference using Confidence-Gated Model Routing

**Mohammad Anas, Nishit Singhal, Krish Garg, Aarush Singh**
Department of Computer Science & Engineering, Delhi Technological University
Advisor: Dr. Neha Gupta

*Preprint, November 2026.*

---

## Abstract

A cascade answers with a small language model and escalates to a larger one only when a confidence gate flags the draft as unreliable. Whether this saves energy is usually argued from API prices, call counts or FLOPs; we measure it. We study nine draft→target pairs from Qwen2.5-0.5B to 32B on an Apple M2 laptop and on 2x NVIDIA T4 GPUs. We log every answer with its per-query hardware energy, including the draft compute wasted on escalated queries, and replay cascades at every escalation rate on up to 3,000 open-domain questions. For all six pairs up to 7B, keeping 95% of the large model's accuracy costs between 4% less and 58% more energy than always using the large model, and better gates do not change this: semantic entropy detects errors best (up to +0.074 AUROC over token entropy) but its sampling costs 4-10x the large model's energy, and hidden-state probes, including one trained to predict semantic entropy, only match token entropy. Large pairs save energy: at 95% retention, 7B→14B saves 24% (95% CI 12-32%), 7B→32B 39% (28-50%) and 14B→32B 55% (41-59%), all measured. They are also 23-48% faster than always-large. The reason is not a better gate, since token entropy detects draft errors about equally well at every size (AUROC 0.74-0.81). It is that larger drafts already answer most of what the target answers (coverage 0.73-0.83, against 0.32-0.65) at a fraction of its energy.

A four-cell binormal model predicts the saving from quantities observable without running a cascade, within 5 points on all nine pairs, and yields a phase diagram of where cascading pays. It also shows why gates matter little: uncertainty ranks queries that neither model answers as high as those the large model can fix. Replaying public Open LLM Leaderboard logs for 11 models up to 72B, with energy scaled by our measured exponents, projects savings that rise with scale (39-48% for Qwen2.5-14B→72B), while 99% retention remains hard at every size. An entropy threshold set on two question sources keeps 91.6-97.7% retention on a third, but it must be recalibrated for each model pair. A live router on the M2 gives results consistent with the replay. We propose reporting cascade energy at fixed accuracy retention as an inference-energy label.

**Keywords:** LLM cascades, energy efficiency, uncertainty estimation, model routing, green AI, hallucination.

---

## 1 Introduction

Inference now accounts for most of the energy used by deployed language models [4], [5]. Sending every query to the largest model is the reliable default, but small models answer many queries correctly. A cascade runs a small *draft* model first and escalates to a large *target* model only when a *gate* judges the draft unreliable. Every escalated query pays twice: the draft is computed and then thrown away.

Cascade and routing work usually reports cost as API price, call counts or compute proxies [6]-[9], [13], [14]. Measured-energy routing exists for pre-generation routers [12], but post-generation, confidence-gated cascades have mostly not been measured. Our earlier single-pair pilot on an Apple M2 found the cascade energy-negative at high accuracy retention (Appendix A).

This paper asks the general question: **for which model pairs does a confidence-gated cascade save energy, and what decides it?** The answer turns out to be simple and predictive.

**Contributions.**
1. **Measurements across scale.** Nine draft→target pairs (Qwen2.5 0.5B-32B, fp16 and 4-bit) on two platforms, with per-query hardware energy that includes wasted draft compute, replayed at every escalation rate on identical questions (Section 5.1, 5.3).
2. **A negative result for better gates.** Semantic entropy and hidden-state probes do not rescue small pairs. Gates track question *hardness*, not whether the large model can fix the answer (Section 5.2).
3. **A predictive model and phase diagram.** A four-cell binormal model predicts the saving at any accuracy retention from the energy ratio, accuracies, coverage and per-cell AUROCs. It is validated within 5 points on all nine pairs. The resulting phase diagram shows which pairs can save energy at all (Section 4, 5.4).
4. **A projection to 72B.** Public per-question leaderboard logs for 11 Qwen2.5 and Llama-3.x models give savings for 27 same-family pairs up to 72B, with energy scaled by the range of per-query exponents we measured (Section 5.5).
5. **A reporting proposal.** Cascade energy at fixed retention, as a candidate inference-energy label (Section 6.2).
6. **Evaluation beyond energy, and a working system.** Latency, hallucination rate and CO2 at three grid intensities; transfer of gates and thresholds to unseen datasets and model pairs; a live router with metered energy; and an interactive dashboard (Sections 5.6-5.8).

---

## 2 Related Work

**Energy of LLM inference.** Samsi et al. [5] benchmark LLaMA inference energy on V100/A100 GPUs. Luccioni et al. [4] measure 88 models across ten tasks. Argerich and Patino-Martinez [3] profile size, depth, batch size and quantisation as energy levers. These studies measure single models, not cascades in which escalated queries pay for two generations.

**Hallucination and uncertainty.** Surveys [1], [10] cover hallucination in LLMs. Semantic entropy [2] clusters several sampled answers by meaning to flag confabulations. Semantic-entropy probes [11] and PEP [15] approximate such signals from hidden states. All are evaluated as detectors, not as deferral signals with energy cost.

**Cascades and routing.** FrugalGPT [6] chains APIs with a learned stopping scorer. AutoMix [7] lets the small model self-verify. Hybrid LLM [9] routes before generation. Gupta et al. [8] show that token-level uncertainty beats sequence-level confidence for deferral. These works measure cost as price, calls or compute.

**Closest 2026 work.**
- *Measured Joules, Learned Routes* [12] measures A100 energy for pre-generation routing among nine models on MMLU and GSM8K.
- *Signed Rescue Routing* [13] studies harm-aware Qwen3 cascades, with cost modelled rather than measured. Its "rescue" framing is closest to our four-cell model.
- *UCCI* [14] calibrates cascade uncertainty and reports latency.
- *ReLope* [16] uses prefill hidden states as a router.

To our knowledge, none combines a post-generation confidence-gated cascade, measured energy including wasted drafts, multiple model scales, and a model that predicts when the cascade pays. We describe these preprints as their reports state; we have not re-verified their claims.

**Energy reporting and policy.** The EU AI Act requires providers of general-purpose AI models to document the "known or estimated energy consumption of the model" (Art. 53(1)(a), Annex XI §1 point 2(e)) [26]. Ebert et al. [27] argue that inference energy is left unaddressed and should be brought into scope. Our proposed label (Section 6.2) is one concrete metric for that gap.

---

## 3 Setup

### 3.1 Cascade accounting

A draft model *S* answers every query. A gate scores each (query, draft answer) pair, and the top fraction *r* of queries by score is escalated to the target *L*. Let ρ = E_S / E_L be the draft-to-target energy ratio per query. With per-query energy taken as each model's measured mean, cascade energy relative to always-large is

E_cascade / E_L = ρ + r,

so the cascade saves energy only if r < 1 − ρ (the **break-even escalation rate**). Accuracy *retention* R is cascade accuracy divided by always-large accuracy. The **saving at retention R** is 1 − ρ − r(R), where r(R) is the smallest escalation rate whose accuracy reaches R × acc_L. A pre-generation router skips the draft for escalated queries, (1 − r)ρ + r; we treat routers only in Appendix A.

Answers are logged once per model. Any escalation rate is then evaluated by offline replay: an escalated query takes the target's logged answer and pays both models' energy. Figure 1 summarises the pipeline.

<!-- FIG:fig1 -->

### 3.2 Models, data and platforms

*Table 1. Platforms, models and questions.*

| Platform | Energy meter | Models | Questions | Evaluated |
|---|---|---|---|---|
| Apple M2, 16 GB (MLX) | `macmon`: SoC + DRAM power at 100 ms, idle subtracted | Qwen2.5 3B, 7B Instruct (4-bit) | 1,000: TriviaQA + SciQ | last 30% (n = 300) |
| Kaggle 2x T4 (`transformers`, fp16) | NVML total-energy counter, both GPUs | Qwen2.5 0.5B, 1.5B, 3B, 7B Instruct | 3,000: TriviaQA + SciQ + NQ-open | last 30% (n = 900) |
| Kaggle 2x T4 (`transformers`, bitsandbytes 4-bit) | NVML, both GPUs | Qwen2.5 7B, 14B, 32B Instruct (`unsloth` bnb-4bit) | same 3,000 | all answered (3,000; 2,664 for 32B) |
| none (public logs) | projected, ρ = (N_S/N_L)^β | 7 Qwen2.5 (0.5B-72B), 4 Llama-3.x (1B-70B) Instruct | MMLU-Pro, 12,032 | all |

Questions are open-ended (SciQ without its options); the 3,000-question set takes the first 1,000 of each source and shuffles them with seed 0. Prompts use each model's chat template with the instruction "Answer the question with only the short answer (a few words). No explanation." Decoding is greedy with at most 24 new tokens. An answer is correct if, after normalisation, a reference alias equals it or appears in it as a whole word.

The small T4 pairs evaluate on the last 30% because probes were fit on the first 70%. The 4-bit runs fit nothing, so they use every answered question. The 32B run hit Kaggle's time limit after 2,664 of 3,000 questions, so pairs with 32B use those 2,664.

### 3.3 Gates

Every gate outputs a risk score where higher means more likely wrong.
- **Token entropy.** Mean next-token entropy over the generated answer; this is the default gate.
- **Log-probability.** Negative mean and minimum token log-probability.
- **Hidden-state probe.** L2-regularised logistic regression on the draft's hidden state at the last answer token, with layer chosen by 5-fold CV on the training split. A prefill-only variant at the last prompt token serves as a pre-generation router.
- **Semantic entropy (SE)** [2]. K = 10 samples at T = 1, clustered by normalised-answer match. Its 10 extra draft generations are charged in energy.
- **References.** Random, and an oracle that escalates exactly the wrong drafts first.

Entropy, log-probability and probes are read from the draft's own generation, so their gate energy is approximately zero. For MMLU-Pro (multiple choice, scored by option log-likelihood) the gate is 1 − the maximum softmax probability over the options.

### 3.4 Energy measurement

**T4.**
- Energy is the NVML total-energy counter summed over both GPUs, read around each batch-1 generation.
- Per-query energy is the mean over 100 real held-out queries per model with natural stopping.
- We use gross energy because in the fp16 run idle power was read on a hot GPU (44-69 W against 19.9 W at true idle).
- The 4-bit runs read idle on a cold GPU before any model loads (19.9-20.3 W). Their idle-subtracted ("net") ratios match the gross ones within 0.02.

**4-bit sessions.** The 32B energy step was cut off by the time limit, so a separate energy-only session re-measured 32B, 14B and 7B on the same 100 queries. That session's 7B/14B ratio, 0.534, matches the first session's 0.535, although absolute joules were about 22% higher (hotter GPUs). Only ratios are used.

**M2.** Power is sampled with `macmon` at 100 ms, and per-query energy comes from an interleaved run of both models (Appendix A).

**Model splits across GPUs.** 32B is split across both GPUs (`device_map="auto"`); 7B and 14B use one, with the second idle but metered.

Figure 2 plots the measured energy per query against model size. Small fp16 models are dominated by fixed overheads, while from 3B upward energy grows almost in proportion to size.

<!-- FIG:fig2 -->

### 3.5 Replay and statistics

Escalation rate is swept over 21 points (0, 0.05, …, 1) for the small pairs and over every escalation count for the 4-bit pairs. We report the saving at 95% and 99% retention. Confidence intervals use percentile bootstrap over queries: 300-500 resamples for savings, and 500-1,000 for AUROC and paired ΔAUROC.

---

## 4 A predictive model of cascade savings

Saving = 1 − ρ − r(R). The energy ratio ρ is a property of the hardware and serving stack; r(R) is a property of the models, the task and the gate. We model r(R) without running a cascade.

**Four cells.** Each query falls into one cell by the joint correctness of the two models:
- **both right** (probability b = coverage × acc_L);
- **rescuable**: draft wrong, target right (acc_L − b);
- **harmed**: draft right, target wrong (acc_S − b);
- **both wrong**: the rest.

Here *coverage* = P(draft right | target right).

**Gate.** The gate score is modelled as binormal per cell. "Both right" scores are N(0, 1), and each other cell c has scores N(d_c, 1), with d_c = √2 Φ⁻¹(AUROC_c), where AUROC_c is the gate's AUROC for that cell against "both right".

**Closed form.** Escalating the top fraction r means thresholding at t with Σ_c P_c Φ̄(t − d_c) = r. Accuracy is then

acc(r) = acc_S + P_rescuable Φ̄(t − d_rescuable) − P_harmed Φ̄(t − d_harmed).

A 400,000-sample Monte Carlo matches this within 0.003 (an assert in `phase.py`).

**Independence model.** The common shortcut assumes the score depends only on whether the draft is right, so d_rescuable = d_bothwrong comes from the plain draft-error AUROC and d_harmed = 0. This is the special case we test against.

**Phase diagram.** Because r(R) does not depend on ρ, the saving map over (ρ, coverage) is exact in ρ. The diagram fixes acc_L = 0.443, q = P(draft right | target wrong) = 0.09 and the mean measured cell AUROCs, and plots each measured pair at its own (ρ, coverage). Figure 3 illustrates the cells and their score distributions.

<!-- FIG:fig3 -->

---

## 5 Results

### 5.1 Small pairs lose energy

*Table 2. Small pairs (measured, test split). Saving is relative to always-large, at 95% retention; negative means the cascade costs more.*

| Pair | Platform | Acc draft / target | ρ | Coverage | Break-even r | Entropy AUROC | Saving @95%: entropy / oracle |
|---|---|---|---|---|---|---|---|
| 0.5B→7B | T4 fp16 | 17.0% / 44.3% | 0.31 | 0.32 | 69% | 0.744 | −26% / −6% |
| 1.5B→7B | T4 fp16 | 28.6% / 44.3% | 0.42 | 0.55 | 58% | 0.808 | −22% / +3% |
| 3B→7B | T4 fp16 | 34.6% / 44.3% | 0.46 | 0.64 | 54% | 0.802 | +4% / +19% |
| 0.5B→3B | T4 fp16 | 17.0% / 34.6% | 0.68 | 0.37 | 32% | 0.744 | −58% / −33% |
| 1.5B→3B | T4 fp16 | 28.6% / 34.6% | 0.90 | 0.65 | 10% | 0.808 | −40% / −10% |
| 3B→7B | M2 4-bit | 35.0% / 51.7% | 0.44 | 0.61 | 56% | 0.804 | −23% / +10% |

Retaining 95% of the target's accuracy is energy-negative on five of six pairs; only 3B→7B on T4 is marginally positive. Even an oracle gate saves at most 19%. The draft is cheap (ρ 0.31-0.46 against a 7B target), but it answers too little of what the target answers: coverage 0.32-0.65. Escalation must therefore exceed the break-even rate before 95% retention is reached.

### 5.2 Better gates do not rescue small pairs; gates track hardness

*Table 3. Gate quality against token entropy (T4 fp16, paired bootstrap). SE's saving charges its 10 draft samples.*

| Draft | ΔAUROC SE − entropy [95% CI] | ΔAUROC probe − entropy [95% CI] | ΔAUROC SEP − entropy [95% CI] | SE saving @95% |
|---|---|---|---|---|
| 0.5B | +0.074 [+0.026, +0.119] | +0.019 [−0.038, +0.073] | −0.001 [−0.052, +0.051] | −333% |
| 1.5B | +0.029 [+0.005, +0.055] | +0.029 [−0.011, +0.065] | +0.008 [−0.027, +0.045] | −432% |
| 3B | +0.012 [−0.010, +0.035] | +0.009 [−0.026, +0.045] | −0.023 [−0.058, +0.013] | −466% |

- **Semantic entropy** is the best detector, significantly so for the 0.5B and 1.5B drafts. But its sampling makes the cascade cost 4-10x always-large.
- **Hidden-state probe** matches token entropy on every draft (all CIs include 0), as on M2 (ΔAUROC +0.002, CI −0.060 to +0.057).
- **Semantic-entropy probe (SEP)** [11]. A linear probe on the draft's hidden state, trained on the fit split to predict *high* semantic entropy (SE binarised at the split minimising within-group squared error), costs no extra sampling. It predicts high SE well on the test split (AUROC 0.79-0.82), but on draft errors it only matches token entropy (all CIs include 0). Its saving at 95% retention is −26%, −17% and −6% for the 0.5B, 1.5B and 3B drafts into 7B. Cheap SE does not recover SE's edge.

The four-cell decomposition explains why gate quality is not the bottleneck. Token entropy separates "rescuable" queries from "both right" with AUROC 0.73-0.81, but it ranks "both wrong" queries just as high (0.76-0.85). The gate measures how hard a question is, not whether the target can answer it. Escalations are therefore spent on queries that neither model gets right. As a result, its AUROC for picking out exactly the rescuable queries from all others is only 0.52-0.59 on the small pairs. A gate with a higher draft-error AUROC raises both cells together and gains little (Figure 4).

<!-- FIG:fig4 -->

### 5.3 Large pairs save energy

*Table 4. Large pairs, Qwen2.5 4-bit on T4 (measured, all answered questions; entropy gate; 95% bootstrap CI).*

| Pair | n | Acc draft / target | ρ (measured) | Coverage | Entropy AUROC | r @95% | Saving @95% | Saving @99% | Oracle @95% |
|---|---|---|---|---|---|---|---|---|---|
| 7B→14B | 3,000 | 43.6% / 50.2% | 0.53 | 0.74 | 0.780 | 23% | **+24%** [+12, +32] | −3% [−30, +17] | +29% |
| 7B→32B | 2,664 | 43.6% / 52.8% | 0.22 | 0.73 | 0.777 | 39% | **+39%** [+28, +50] | 0% [−9, +16] | +53% |
| 14B→32B | 2,664 | 49.8% / 52.8% | 0.41 | 0.83 | 0.764 | 4% | **+55%** [+41, +59] | +27% [−4, +49] | +55% |

Energy per query (gross, batch 1) was 7B 69.5 J and 14B 129.9 J in the first session, and 84.4 J, 157.9 J and 32B 385.5 J in the energy-only session. Net of cold idle the ratios are 0.53 (7B/14B) and 0.39 (14B/32B), against 0.53 and 0.41 gross.

All three large pairs save energy at 95% retention. The gate is no better than for the small pairs (AUROC 0.76-0.78); what changed is coverage, 0.73-0.83. 14B→32B is the extreme case: 14B alone already reaches 94% of 32B's accuracy, so 95% retention needs only 4% escalation. At 99% retention the savings shrink to −3% to +27%, with CIs that include zero.

Quantisation barely changes the draft. The 4-bit 7B scores 43.6% against 44.4% for fp16 on the same 3,000 questions, with 92.7% per-question agreement. At batch 1, however, bitsandbytes 4-bit costs more energy per query than fp16 (7B: 69.5 J vs 38.8 J), so we compare ratios only within one precision.

Figure 5 shows the full accuracy-energy trade-off. Small drafts start far below 95% retention and reach it only past break-even, while large drafts start close to it.

<!-- FIG:fig5 -->

### 5.4 The phase diagram predicts all nine pairs

*Table 5. Saving at 95% retention: measured replay vs the four-cell model and the independence model (21-point grid).*

| Pair | ρ | Coverage | Rescuable-vs-all AUROC | Measured | Four-cell model | Independence model | Max accuracy-curve error, four-cell / independence |
|---|---|---|---|---|---|---|---|
| T4 0.5B→7B | 0.31 | 0.32 | 0.53 | −26% | −21% | −16% | 1.2 / 2.1 pts |
| T4 1.5B→7B | 0.42 | 0.55 | 0.55 | −22% | −17% | −7% | 1.0 / 2.6 pts |
| T4 3B→7B | 0.46 | 0.64 | 0.58 | +4% | −1% | +14% | 0.7 / 2.1 pts |
| T4 0.5B→3B | 0.68 | 0.37 | 0.52 | −58% | −53% | −48% | 0.7 / 1.3 pts |
| T4 1.5B→3B | 0.90 | 0.65 | 0.54 | −40% | −45% | −20% | 0.7 / 2.3 pts |
| M2 3B→7B | 0.44 | 0.61 | 0.59 | −24% | −24% | −4% | 1.6 / 2.9 pts |
| T4 4-bit 7B→14B | 0.53 | 0.74 | 0.65 | +22% | +17% | +22% | 0.5 / 1.6 pts |
| T4 4-bit 7B→32B | 0.22 | 0.73 | 0.65 | +38% | +33% | +43% | 0.5 / 1.6 pts |
| T4 4-bit 14B→32B | 0.41 | 0.83 | 0.63 | +54% | +54% | +54% | 0.5 / 2.1 pts |

On all nine pairs the four-cell model:
- reproduces the measured accuracy-vs-escalation curve within 0.5-1.6 accuracy points;
- predicts the saving at 95% and at 90% retention within 5 points, one step of the replay grid, with no consistent sign.

The independence model is optimistic by 10-20 points at 95% retention on the small pairs (0-5 on the large ones). It assumes the gate never wastes escalations on "both wrong" queries, which Section 5.2 shows it does.

Notes on the table. The measured column uses the 21-point grid, so 7B→14B reads +22% here against +24% with exact counts (Table 4). M2 reads −24% here, with a constant ρ, against −23% with per-query energies (Table 2).

**Figure 6** plots saving at 95% retention over the energy ratio ρ (x) and coverage (y). The solid line is break-even for the entropy gate and the dashed line for an oracle.
- All six small pairs lie on or below the break-even line.
- All three large pairs lie above it, in the region where the saving is 20-55%.
- Above a coverage of about 0.84 the draft alone keeps 95%, and the saving approaches 1 − ρ.

**Reading the diagram.** A practitioner needs only:
- ρ, from two short energy measurements;
- coverage and the two accuracies, from a labelled sample of a few hundred queries answered by both models.

The diagram then shows whether a cascade can pay before one is built.

<!-- FIG:phase -->

### 5.5 Projection to 72B with public leaderboard logs

The Open LLM Leaderboard [23] publishes per-question log-likelihoods for MMLU-Pro [22]. We replay cascades for every same-family pair among 11 models on all 12,032 questions.

**Accuracies.** All 11 match the leaderboard's own published "MMLU-PRO Raw" scores exactly (`results_olb/leaderboard_crosscheck.json`).

**Energy.** Energy is not measured here: ρ = (N_S/N_L)^β, with β spanning the per-query exponents we measured for drafts of 3B and up.
- β = 0.86 is the lowest (T4 fp16 3B→7B) and is used as the conservative value.
- β = 1.13 is the highest (T4 4-bit 14B→32B).
- In between: M2 3B→7B 0.91, 4-bit 7B→14B 0.95, 4-bit 7B→32B 1.05.

Drafts below 3B measured lower on T4 (0.42-0.55), because fixed overheads dominate tiny models. Their projections are therefore optimistic.

*Table 6. Projected saving on MMLU-Pro (max-probability gate).*

| Pair | Acc draft / target | Coverage | Gate AUROC | ρ (β = 0.86) | Saving @95%, β = 0.86 / 1.13 | Saving @99% (β = 0.86) |
|---|---|---|---|---|---|---|
| Qwen 0.5B→7B | 17.2% / 42.9% | 0.24 | 0.635 | 0.09 | −1% / +3% | −7% |
| Qwen 3B→7B | 32.5% / 42.9% | 0.57 | 0.716 | 0.46 | −14% / −4% | −35% |
| Qwen 7B→14B | 42.9% / 49.0% | 0.69 | 0.746 | 0.57 | −1% / +8% | −29% |
| Qwen 7B→72B | 42.9% / 56.3% | 0.65 | 0.746 | 0.14 | +24% / +30% | +3% |
| Qwen 14B→32B | 49.0% / 56.7% | 0.75 | 0.771 | 0.51 | +10% / +20% | −19% |
| Qwen 14B→72B | 49.0% / 56.3% | 0.74 | 0.771 | 0.25 | **+39% / +48%** | +9% |
| Qwen 32B→72B | 56.7% / 56.3% | 0.84 | 0.792 | 0.50 | +50% / +60% | +50% |
| Llama 3B→70B | 31.9% / 53.1% | 0.48 | 0.714 | 0.07 | +10% / +14% | +1% |
| Llama 8B→70B | 38.0% / 53.1% | 0.57 | 0.747 | 0.15 | +11% / +17% | −3% |

**Figure 7** shows all 21 Qwen pairs.

<!-- FIG:scale -->
- **Savings rise with scale.** They grow as both models grow and the accuracy gap narrows. Drafts below 3B save at most 11% (14% at β = 1.13) even with a 72B target, and these figures are optimistic.
- **Degenerate case.** 32B→72B is degenerate: the 32B is as accurate as the 72B on this task, so "cascading" means using the smaller model.
- **Gate AUROC.** Unlike open QA, gate AUROC on MMLU-Pro rises broadly with draft size: 0.64 for 0.5B, 0.72 for 1.5B and 3B, 0.79 for 32B.

**The task matters.** The same 14B→32B pair saves 55% on open QA (measured) but is projected at only 10-20% on MMLU-Pro. Two things differ:
- 14B reaches 94% of 32B's accuracy on open QA but 87% on MMLU-Pro.
- The projected ρ (0.51) is higher than the measured one (0.41).

Coverage is a property of the task, ρ of the hardware and serving stack, and both must be measured for the deployment at hand.

### 5.6 Latency, hallucination rate and carbon

*Table 7. Every measured pair at 95% retention, entropy gate. Hallucination rate is the share of wrong answers (the models never abstain). Latency is sequential: the draft always answers, and the target runs after it if the query is escalated. CO2 is device energy times the grid factor of India (727 g/kWh [29]).*

| Pair | Hallucination rate: draft / target / cascade | Escalated | Energy saving | Latency per query, target → cascade | g CO2 per 1,000 queries, target → cascade |
|---|---|---|---|---|---|
| M2 3B→7B | 65.0% / 48.3% / 50.3% | 80% | −23% | 1.75 → 2.15 s (+23%) | 0.52 → 0.64 |
| T4 0.5B→7B | 83.0% / 55.7% / 56.3% | 95% | −26% | 0.33 → 0.54 s (+67%) | 7.84 → 9.90 |
| T4 1.5B→7B | 71.4% / 55.7% / 57.2% | 80% | −22% | 0.33 → 0.47 s (+44%) | 7.84 → 9.53 |
| T4 3B→7B | 65.4% / 55.7% / 57.7% | 50% | +4% | 0.33 → 0.40 s (+23%) | 7.84 → 7.53 |
| T4 0.5B→3B | 83.0% / 65.4% / 67.0% | 90% | −58% | 0.24 → 0.45 s (+88%) | 3.61 → 5.70 |
| T4 1.5B→3B | 71.4% / 65.4% / 66.8% | 50% | −40% | 0.24 → 0.33 s (+37%) | 3.61 → 5.06 |
| T4 4-bit 7B→14B | 56.4% / 49.8% / 52.3% | 23% | +24% | 1.71 → 1.31 s (−23%) | 26.2 → 20.0 |
| T4 4-bit 7B→32B | 56.4% / 47.2% / 49.8% | 39% | +39% | 3.52 → 2.27 s (−36%) | 77.8 → 47.3 |
| T4 4-bit 14B→32B | 50.2% / 47.2% / 49.8% | 4% | +55% | 3.52 → 1.84 s (−48%) | 77.8 → 35.1 |

- **Hallucination.** By construction, 95% retention keeps the cascade's hallucination rate within 0.6-2.6 points of always-large, against 3-27 points for the draft alone.
- **Latency follows energy.** Small pairs are 23-88% slower than always-large, because half or more of the queries run both models. Large pairs are 23-48% faster, because the draft is 1.9-3.9x faster than the target and only 4-39% of queries are escalated.
- **Carbon.** CO2 is energy times a grid factor, so the percentage saving equals the energy saving. The grid sets the absolute amount. For 14B→32B the cascade saves 42.7 kg CO2 per million queries on India's grid (727 g/kWh [29]), 26.9 kg at the world average (458 gCO2e/kWh, 2025 [30]) and 1.2 kg in France (19.6 gCO2e/kWh, 2025 [31]). The grid matters more than the cascade: the same always-32B workload emits 37x less CO2 in France than in India, while the cascade cuts it 2.2x. These figures are device energy only, with no data-centre overhead (PUE).

### 5.7 Do gates and thresholds generalise?

**Gates on unseen datasets.** On the T4 fp16 drafts we train each learned gate on the fit rows of two sources and test it on the third source's test rows (n = 293-306). The correctness probe loses 0.031 AUROC on average against the same probe trained on all sources (up to 0.107, for 0.5B on NQ); the SEP loses 0.007. Token entropy needs no training and is at least as good as the transferred probe on 8 of 9 held-out sets (mean AUROC 0.777 against 0.738).

**Thresholds on unseen datasets.** Tables 2-7 choose the escalation threshold on the questions they report on. Here an entropy threshold is set for 95% retention on two sources and applied unchanged to the third.

*Table 8. Accuracy retention reached on a held-out source (target 95%) and the saving there, with the threshold set on the other two sources.*

| Pair | Retention: TriviaQA / SciQ / NQ | Saving: TriviaQA / SciQ / NQ |
|---|---|---|
| T4 4-bit 7B→14B | 93.7% / 97.7% / 91.6% | +18% / +35% / +16% |
| T4 4-bit 7B→32B | 93.8% / 97.2% / 93.2% | +36% / +47% / +30% |
| T4 4-bit 14B→32B | 94.9% / 96.6% / 91.7% | +55% / +57% / +59% |
| T4 fp16 3B→7B | 94.2% / 96.7% / 94.3% | −4% / +9% / −12% |

Held-out retention lands within 3.4 points of the target (91.6-97.7%, mean 94.6%), and every large pair still saves 16-59%. NQ gives the lowest retention for all three large pairs. **Across model pairs the threshold does not transfer:** reusing one large pair's threshold on another reaches 87.4-98.1% retention. The threshold must be recalibrated for each pair, which needs only a labelled calibration set of a few hundred queries.

### 5.8 A working router

`src/route.py` runs the cascade live on the M2 (Qwen2.5-3B → 7B, MLX 4-bit). Each query is answered by the 3B. Its mean token entropy, read from its own decoding, is compared with a threshold t, and the query goes to the 7B only if the entropy is above t. Per-query energy (SoC + DRAM, idle subtracted) and latency are metered around the whole call. t is set for 95% retention on the 700 M2 fit questions, so the evaluation questions never set it.

On the 300 M2 test questions (t = 0.469), run interleaved with always-7B on the same questions:
- The router escalated 60% of queries and kept 90.5% of always-7B accuracy (44.3% against 49.0%; bootstrap 95% CI 84.8-96.5%).
- It used 5.60 J per query against 6.88 J, an 18.7% saving (CI −12% to +46%). Mean latency was 0.60 s against 0.57 s (+6%).
- Its gate decisions matched the logged draft entropies on all 300 questions. Always-7B scored 49.0% live against 51.7% in the logged run (8 of 300 questions differ).

The threshold, set on other questions, undershoots the 95% target, consistent with Section 5.7. At the 90.5% it reached, the saving is in line with the replay, which saved about 18% at about 85% retention on this pair (Appendix A). Replay on the same test questions needs 80% escalation, and 23% more energy than always-7B, for a full 95%.

An offline dashboard (`dashboard/greengate_dashboard.html`, Figure 8) shows energy saved, escalation, hallucination rate, latency and CO2 for any measured pair, retention level and grid.

<!-- FIG:dash -->

---

## 6 Discussion

### 6.1 When to cascade

Our results reduce to one inequality: a cascade saves energy at retention R only if the escalation it needs, r(R), is below 1 − ρ.
- Making the draft smaller lowers ρ, but it raises r(R) faster, because coverage collapses. Every pair with a draft at or below 3B loses or at best breaks even on open QA (best: +4%).
- Improving the gate helps little, because uncertainty scores rank "both wrong" queries as high as rescuable ones.
- What makes cascades pay is a draft that is already almost as good as the target, at about half the cost or less (measured ρ 0.22-0.53). That is the regime of large models, where the accuracy gap between sizes narrows while the energy gap stays wide.

For deployments that serve 14B-70B models, a same-family draft one size down is a strong default. Its saving at 95% retention can be read off the phase diagram from a few hundred labelled queries.

### 6.2 An inference-energy label for cascades and routers

Energy documentation under the EU AI Act centres on the model's "known or estimated energy consumption" [26], and inference energy is not yet covered [27]. For systems that combine models, single-model figures are not enough: what matters is the energy per query at a stated accuracy.

We propose a label in three parts:
1. **J/query at 95% (and 99%) accuracy retention**, relative to always using the largest model in the system.
2. **The measurement setup**: hardware, batch size and meter.
3. **The evaluation task.**

Our measured pairs would read, for example, "14B→32B on 2x T4: 0.45 of always-32B energy at 95% retention, 0.73 at 99%". The label is cheap to compute with the replay method used here. It also prevents a cascade from being reported as "efficient" when, like every small pair in Table 2, it costs more than the large model at the accuracy users expect.

### 6.3 Limitations

- **Hardware.** Two platforms, both small: a laptop and two T4 GPUs. Batch-1 energy differs from batched serving; on M2, batching 8 queries cut energy by only 6-9% (Appendix A). Large-model results come from 4-bit models on T4, where 32B spans two GPUs.
- **Task.** Short factual answers (about 4 tokens), so prompt processing weighs heavily. Long-form generation, where decoding dominates, may give different ratios.
- **Scale beyond 32B is projected.** It rests on public logs, not on measurement, and uses a multiple-choice task. Energy is scaled by the measured exponent range, which itself comes from 2-4 sizes per platform.
- **Energy accounting.**
  - Gross energy on T4 charges small single-GPU models for an idle second GPU; net-of-idle ratios agree within 0.02 on the 4-bit runs.
  - The fp16 run's idle reading was taken hot, so only gross energy is used there.
- **Statistics.**
  - n is 300-3,000 per pair; savings at 99% retention have CIs that include zero.
  - The lenient (plural-tolerant) answer matcher on M2 was chosen after seeing results.
- **Routers.** Pre-generation routers are measured only on M2 (Appendix A). The four-cell model for routers is not yet validated.

---

## 7 Conclusion

Confidence-gated cascades do not save energy for small models, and better gates (semantic entropy, hidden-state probes) do not change that. They do save energy for large models: 24-55% at 95% accuracy retention for measured Qwen2.5 pairs from 7B to 32B. Public leaderboard logs project 39-48% for 14B→72B.

What decides the saving is not the gate but two numbers: the energy ratio ρ, and how much of the target's correct work the draft already does (coverage). A four-cell binormal model turns these into a phase diagram that predicts the saving within 5 points on all nine measured pairs.

Large-pair cascades are also 23-48% faster than always-large. The percentage CO2 saving equals the energy saving on any grid, though the grid sets the absolute amount. Entropy thresholds carry over to unseen datasets within 3.4 points of the retention target, but not to other model pairs.

Next steps are:
- measuring 70B-class pairs and batched serving directly;
- validating the model for pre-generation routers;
- extending the label to cross-provider routers.

**Code and data.** The pipeline, logs and analysis scripts (`analyze_pairs.py`, `analyze_big.py`, `phase.py`, `scale_map.py`, `eval_extra.py`, `probe_extra.py`), the live router (`route.py`) and the dashboard are available from the authors.

---

## Appendix A: Apple M2 pilot (3B→7B, Qwen2.5 4-bit, MLX)

**Setup.** 1,000 TriviaQA + SciQ questions, with probes fit on 70% and n_test = 300.

**Energy.** Energy is SoC + DRAM power from `macmon` at 100 ms, idle subtracted. It comes from an interleaved run in which the 3B answer, the 7B answer and the 3B prefill are executed in random order per question.
- 3B: 1.13 J net per query, 7B: 2.58 J, so ρ = 0.44. 3B prefill alone: 0.77 J.
- A clean re-measurement found these absolute joules about 3x too low (2.9x for 3B, 3.1x for 7B; machine state). The ratio barely moved (0.41 vs 0.44), shifting break-even from 56% to 59%.

**Gates.** The best hidden-state probe (answer position, layer 22) reaches AUROC 0.806 (0.750-0.857), against 0.804 for token entropy and 0.791 for mean log-probability (ΔAUROC probe − entropy +0.002 [−0.060, +0.057]). The prefill-only probe reaches 0.727.

**Operating points.**
- Break-even escalation is 56% for the cascade and 65% for a pre-generation router.
- Retaining 95% needs about 70-80% escalation with real gates (energy-negative, −12% probe, −23% entropy) and 45% with an oracle (+10%).
- The cascade saves about 18% at about 85% retention.

**Token-level energy model.** E ≈ c + a·P + b·O, fitted on 0.5B-7B. Energy per output token scales as N^0.81 and per prompt token as N^1.07. It predicts held-out 7B queries within 6% and 3B queries within 20%.

**Other checks.**
- Batching 1→8 cut per-query SoC energy by 6-9%.
- A probe trained on TriviaQA reaches AUROC 0.80 on SciQ.

Full tables are in the pilot report (`RESULTS.md`, M2 section).

## Appendix B: T4 fp16 small-pair details

*Table B1. Gate AUROC and savings (test, n = 900).*

| Pair | J/query draft / target | AUROC entropy / SE / probe | Saving @95% entropy / probe / oracle | Best saving @90% entropy / probe |
|---|---|---|---|---|
| 0.5B→7B | 12.1 / 38.8 | 0.744 / 0.818 / 0.762 | −26% / −21% / −6% | −16% / −16% |
| 1.5B→7B | 16.1 / 38.8 | 0.808 / 0.836 / 0.836 | −22% / −12% / +3% | −2% / −2% |
| 3B→7B | 17.9 / 38.8 | 0.802 / 0.814 / 0.812 | +4% / −1% / +19% | +14% / +19% |
| 0.5B→3B | 12.1 / 17.9 | 0.744 / 0.818 / 0.762 | −58% / −58% / −33% | −43% / −43% |
| 1.5B→3B | 16.1 / 17.9 | 0.808 / 0.836 / 0.836 | −40% / −25% / −10% | −15% / −10% |

**Idle subtraction.** Subtracting true idle (19.9 W) instead of using gross energy moves break-even to 77/63/59% (→7B) and 43/9% (→3B), without changing any conclusion.

---

## References

[1] L. Huang et al., "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions," *ACM Trans. Inf. Syst.*, vol. 43, no. 2, pp. 1-55, 2025, doi: 10.1145/3703155.

[2] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, "Detecting hallucinations in large language models using semantic entropy," *Nature*, vol. 630, no. 8017, pp. 625-630, 2024, doi: 10.1038/s41586-024-07421-0.

[3] M. F. Argerich and M. Patino-Martinez, "Measuring and improving the energy efficiency of large language models inference," *IEEE Access*, vol. 12, pp. 80194-80207, 2024, doi: 10.1109/ACCESS.2024.3409745.

[4] S. Luccioni, Y. Jernite, and E. Strubell, "Power hungry processing: Watts driving the cost of AI deployment?," in *Proc. ACM FAccT*, 2024, pp. 85-99, doi: 10.1145/3630106.3658542.

[5] S. Samsi et al., "From words to watts: Benchmarking the energy costs of large language model inference," in *Proc. IEEE HPEC*, 2023, pp. 1-9, doi: 10.1109/HPEC58863.2023.10363447.

[6] L. Chen, M. Zaharia, and J. Zou, "FrugalGPT: How to use large language models while reducing cost and improving performance," *Trans. Mach. Learn. Res.*, 2024.

[7] P. Aggarwal et al., "AutoMix: Automatically mixing language models," in *Proc. NeurIPS*, 2024.

[8] N. Gupta, H. Narasimhan, W. Jitkrittum, A. S. Rawat, A. K. Menon, and S. Kumar, "Language model cascades: Token-level uncertainty and beyond," in *Proc. ICLR*, 2024.

[9] D. Ding et al., "Hybrid LLM: Cost-efficient and quality-aware query routing," in *Proc. ICLR*, 2024.

[10] Z. Ji et al., "Survey of hallucination in natural language generation," *ACM Comput. Surv.*, vol. 55, no. 12, pp. 1-38, 2023, doi: 10.1145/3571730.

[11] J. Kossen, J. Han, M. Razzak, L. Schut, S. Malik, and Y. Gal, "Semantic entropy probes: Robust and cheap hallucination detection in LLMs," arXiv:2406.15927, 2024.

[12] M. A. R. Siddiqui, D. Rojas, C. Yang, W. Cui, Y. Shi, and Y. Chen, "Measured joules, learned routes: Learning to route for energy-efficient LLM serving," arXiv:2609.23085, Sep. 2026.

[13] Z. Wang, S. Li, P. Song, S. Chen, Q. Song, and Q. Liu, "Signed rescue routing: Harm-aware cascades for efficient LLM inference," arXiv:2609.07786, Sep. 2026.

[14] V. Kotte, "UCCI: Calibrated uncertainty for cost-optimal LLM cascade routing," arXiv:2605.18796, May 2026.

[15] Z. Mrykhin and V. Malykh, "Prompt embedding probes (PEP): Hallucination detection in LLMs from hidden states," arXiv:2608.08024, Aug. 2026.

[16] Y. Zeng, C. Wang, B. J. Chen, and L. Lin, "ReLope: From hidden-state probing to a decision module for multimodal LLM routing," arXiv:2603.24787, Mar. 2026.

[17] Qwen Team, "Qwen2.5 technical report," arXiv:2412.15115, 2024.

[18] MLX and `mlx-lm`, https://github.com/ml-explore/mlx; `macmon`, https://github.com/vladkens/macmon. MLX 0.32.4, mlx-lm 0.32.2, macmon 0.10.0; accessed Nov. 2026.

[19] M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer, "TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension," in *Proc. ACL*, 2017, pp. 1601-1611.

[20] J. Welbl, N. F. Liu, and M. Gardner, "Crowdsourcing multiple choice science questions," in *Proc. W-NUT*, 2017, pp. 94-106.

[21] T. Kwiatkowski et al., "Natural Questions: A benchmark for question answering research," *Trans. Assoc. Comput. Linguist.*, vol. 7, pp. 452-466, 2019. (Open-domain split `nq_open`.)

[22] Y. Wang et al., "MMLU-Pro: A more robust and challenging multi-task language understanding benchmark," in *Proc. NeurIPS Datasets and Benchmarks Track*, 2024.

[23] C. Fourrier, N. Habib, A. Lozovskaya, K. Szafer, and T. Wolf, "Open LLM Leaderboard v2," Hugging Face, 2024. Per-question details: `open-llm-leaderboard/<model>-details`; scores: `open-llm-leaderboard/contents`; accessed 4 Nov 2026.

[24] A. Grattafiori et al., "The Llama 3 herd of models," arXiv:2407.21783, 2024.

[25] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in *Proc. NeurIPS*, 2023. (4-bit NF4 via `bitsandbytes`; checkpoints `unsloth/Qwen2.5-{7B,14B,32B}-Instruct-bnb-4bit`.)

[26] Regulation (EU) 2024/1689 of the European Parliament and of the Council (Artificial Intelligence Act), *OJ L*, 12 Jul. 2024: Art. 53(1)(a) and Annex XI, Section 1, point 2(e).

[27] K. Ebert, N. Alder, R. Herbrich, and P. Hacker, "AI, climate, and regulation: From data centers to the AI Act," arXiv:2410.06681, 2024.

[28] NVIDIA Management Library (NVML), `nvmlDeviceGetTotalEnergyConsumption`, NVIDIA Corp., accessed Nov. 2026.

[29] Central Electricity Authority, Government of India, "CO2 Baseline Database for the Indian Power Sector, Version 20.0," Dec. 2024. Weighted average emission factor including renewables, FY2023-24: 0.727 tCO2/MWh.

[30] Ember, "Global Electricity Review 2026," 2026. Global electricity emissions intensity in 2025: 458 gCO2e/kWh.

[31] RTE, "Bilan électrique 2025: GHG emissions," 2026. Carbon intensity of French electricity generation in 2025: 19.6 gCO2e/kWh (direct).
