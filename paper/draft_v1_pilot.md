# Measured Energy of Hallucination-Risk-Gated Small-to-Large LLM Cascades: Apple Silicon and GPU

**Mohammad Anas, Nishit Singhal, Krish Garg**
Department of Computer Science & Engineering, Delhi Technological University
Advisor: Dr. Neha Gupta

*Workshop-paper draft.*

---

## Abstract

Sending every query to the largest available language model is a reliable but energy-expensive default. A cascade answers with a small model first and escalates to a larger one only when the small model's answer is likely to be wrong. Prior cascade and routing work usually reports cost as API price, call counts, FLOPs or modelled latency; measured energy, including draft compute wasted on escalated queries, is rarely reported for post-generation, hallucination-risk-gated cascades. We study such a cascade on a single base Apple M2 (16 GB) laptop: Qwen2.5-3B-Instruct (4-bit) drafts an answer and Qwen2.5-7B-Instruct (4-bit) is the escalation target, both served with MLX. Risk gates are random, oracle, token log-probability, token entropy, and a linear probe on the draft model's hidden states, which are captured inside the draft's own generation loop, so the probe needs no extra forward pass and its gate energy is approximately zero. We replay logged answers over a sweep of escalation rates on 1,000 open-ended factual questions (TriviaQA and SciQ; 300 held out for testing), using per-query energy (SoC + DRAM, idle subtracted) measured in a separate interleaved run. The probe's AUROC for predicting draft errors is 0.81 (95% CI 0.75-0.86), indistinguishable from token entropy (0.80) and mean log-probability (0.79); the paired AUROC difference against entropy is +0.002 (95% CI -0.060 to +0.057), and n_test = 300 cannot resolve gaps below about 0.06. The 3B draft costs about 44% of the 7B per query (1.13 vs 2.58 J net), so the break-even escalation rate is 56% for the cascade and 65% for a pre-generation router. Retaining at least 95% of the 7B's accuracy needs about 70% escalation with the probe (energy-negative) and 45% even with an oracle (about 10% saving); on this pair and hardware, energy savings come only with accuracy loss, for example about 18% saving at about 85% of the 7B's accuracy. Batching the models from 1 to 8 cuts per-query SoC energy by only about 6-9%. A token-level energy model fitted on Qwen2.5 0.5B-7B (E ≈ c + a·prompt tokens + b·output tokens) shows energy per output token scaling as N^0.81 and per prompt token as N^1.07, with an output token costing about 10x a prompt token; it predicts held-out 7B queries within 6% and 3B within 20%. A clean re-measurement found the interleaved run's absolute joules about 2.5x too low, but the 3B/7B ratio (0.41 vs 0.44) and the conclusions are unchanged. A probe trained on TriviaQA reaches an AUROC of 0.80 on SciQ, consistent with but not distinguishable from training-free baselines. A replication on 2x NVIDIA T4 GPUs with four Qwen2.5 sizes (0.5B-7B), five draft-target pairs, 3,000 questions and semantic entropy confirms the result. Retaining 95% of the large model's accuracy is energy-negative for every real gate except one (+4%), and an oracle saves at most 19%. Semantic entropy detects errors best (up to +0.074 AUROC over token entropy) but costs 4-10x the energy of always using the large model.


---

## 1 Introduction

Inference now accounts for most of the energy used by deployed language models [4], [5]. When a small model answers most questions correctly, routing everything to a larger model spends energy on queries that did not need it. A cascade addresses this by having a small *draft* model answer first and consulting a larger model only when a *gate* judges the draft unreliable. Whether a cascade saves energy depends on three quantities that are easy to overlook: the accuracy gap between the two models, the quality of the gate's ranking, and the cost of the work that escalation throws away, namely the draft generation (and any gate computation) that was paid for before the query was escalated.

Existing cascade and routing systems optimise task accuracy against cost measured as API price, call counts or compute proxies [6]-[9]. Uncertainty estimates for hallucination detection, such as semantic entropy [2] and its single-pass hidden-state approximation [11], have been evaluated as detectors rather than as deferral signals. Recent 2026 work has begun to close parts of this gap (Section 2), so we do not claim that uncertainty-based deferral, hidden-state probing, or energy-aware routing is new in itself.

We state our positioning narrowly. The project this pilot belongs to aims to measure energy for a *post-generation*, hallucination-risk-gated small-to-large cascade, counting wasted draft compute and gate overhead, studying batching, and evaluating on factual question answering where hallucination is the failure mode. This paper reports the first, single-machine stage of that programme. It makes three contributions:

1. A measurement and replay pipeline on a base Apple M2 that records, for every query and model, the answer, correctness, token-level uncertainty, and idle-subtracted energy integrated from 100 ms power samples, collected without root privileges (energy is measured in a separate interleaved run, Section 3.5), plus a token-level energy model across four model sizes (Section 3.7).
2. A comparison of confidence gates and a linear hidden-state probe as deferral signals, including a prefill-only probe usable before generation, evaluated at matched escalation rates on identical queries.
3. An energy-grounded analysis of the resulting cascade: Pareto curves from offline replay, a break-even escalation rate from measured per-query energies (for the cascade and for a pre-generation router), a batch-1 versus batch-8 comparison, and a TriviaQA-to-SciQ transfer check.

The pilot's limits are stated up front: one hardware configuration, one model pair, one 1,000-question set, one random seed, and ground-truth correctness labels only (no semantic-entropy labels).

---

## 2 Related Work

**Energy of LLM inference.** Samsi et al. [5] benchmarked LLaMA inference on V100 and A100 GPUs and showed that energy per response grows with model size and output length and depends on sharding and batch configuration. Luccioni et al. [4] measured inference energy for 88 models across ten tasks and found general-purpose generative models far more energy-intensive than task-specific ones. Argerich and Patino-Martinez [3] profiled how model size, depth and vocabulary affect inference energy, and identified batch size and quantisation as practical levers. These studies measure single models in isolation, not cascades in which escalated queries pay for two generations.

**Hallucination and its detection.** Huang et al. [1] and Ji et al. [10] survey hallucination in LLMs and NLG. Semantic entropy [2] samples several answers, clusters them by bidirectional entailment and uses the entropy over meaning clusters to flag confabulations without ground truth, at the price of multiple generations per query. Semantic Entropy Probes [11] approximate this with a linear probe on the hidden states of a single generation. Both are evaluated as detectors. PEP [15] (arXiv:2608.08024) likewise studies hidden-state probes for hallucination detection, without routing.

**Cascades and routing.** FrugalGPT [6] chains LLM APIs with a learned scorer that decides when to stop. Hybrid LLM [9] predicts query difficulty before generation and routes to a small or large model. Gupta et al. [8] show that sequence-level confidence suffers from length bias in generative cascades and that deferral rules over token-level uncertainty perform better. AutoMix [7] has the small model verify its own answer and uses a POMDP router to decide escalation. These works measure cost as API price, call counts or compute proxies.

**Recent work closest to ours.** Five 2026 preprints overlap with parts of our setting; we describe what each does and what it leaves open as stated in their reports, and we have not re-verified their claims beyond that.

- *Measured Joules, Learned Routes* [12] (arXiv:2609.23085) measures GPU energy (NVML at 200 ms, idle subtracted) on an A100 and routes each query to one of nine models *before* generation with a Qwen-1.5B router. It does not study a post-generation cascade or hallucination benchmarks (it uses MMLU and GSM8K), and it reports GPU energy on one hardware setup. Our prefill-only probe is a pre-generation router in this spirit, evaluated under the same measured-energy accounting.
- *Signed Rescue Routing* [13] (arXiv:2609.07786) builds a Qwen3-4B to 8B cascade from entropy and margin features on an A100, with cost modelled as a draft cost plus the escalation rate times the large-model cost rather than measured in joules.
- *UCCI* [14] (arXiv:2605.18796) routes cascades with calibrated uncertainty (isotonic regression) and reports H100 latency rather than energy.
- *ReLope* [16] (arXiv:2603.24787) uses a hidden-state probe on prefill states as a router in a multimodal setting, without energy measurement or a hallucination focus.
- *PEP* [15] studies hidden-state probes as hallucination detectors only.

**Positioning.** To our knowledge, and as of the literature check we performed in November 2026, none of these combines (i) a post-generation hallucination-risk gate, (ii) directly measured energy that includes wasted draft compute (gate overhead is approximately zero in our design), and (iii) a batching study, on factual-QA benchmarks. We do not claim that no routing work measures energy: [12] does so for one-shot routing. Our pilot instantiates this combination on one Apple-silicon machine with system-level (not wall-plug) power; the GPU, CPU and wall-plug measurements of the full programme are not part of this paper.

---

## 3 Method

### 3.1 Cascade and notation

A draft model *S* (Qwen2.5-3B-Instruct, 4-bit) answers every query. A gate assigns a risk score to the pair (query, draft answer), or to the query alone for the prefill-only router. A target model *L* (Qwen2.5-7B-Instruct, 4-bit) is called only for queries whose risk score ranks in the top fraction *r* (the escalation rate) of the evaluation set. Escalated queries therefore pay for the draft generation and the target generation. For a query *i*, with measured idle-subtracted energies E_s(i) and E_l(i) for draft and target generation, and a gate energy E_g(i) that is approximately zero for every gate here (Section 3.4), the cascade energy is

E(i; r) = E_s(i) + E_g(i) + 1[i escalated] · E_l(i),

and the cascade answer is the target's answer if escalated and the draft's otherwise. Both models' answers for every query are logged once, so any escalation rate can be evaluated by offline replay without re-running the models. Replay assumes that each escalated query costs and answers exactly as in its logged target run, i.e. it ignores model swap or load cost; both models fit in memory simultaneously on the test machine. For the pre-generation router (Section 3.4), an escalated query skips the draft answer but still pays the draft's prefill E_p, so E(i; r) = E_p(i) + E_l(i) when escalated and E_s(i) otherwise. Replaying the logs against live end-to-end runs at selected operating points is part of the full programme and not done in this pilot.

### 3.2 Serving and generation

Both models are served with MLX (`mlx-lm`) from the 4-bit `mlx-community` conversions, so the draft and target share one inference stack. Prompts use each model's chat template with the system instruction "Answer the question with only the short answer (a few words). No explanation." and the question as the user turn. Decoding is greedy (temperature 0) with a maximum of 24 new tokens. Three warm-up generations (8 tokens) are run and discarded before each run. The hidden-state and uncertainty logging run (`run.py`) uses a custom greedy loop (`step`/`generate`); the energy cost run (`cost.py`, Section 3.5) uses stock `mlx-lm` `stream_generate` with a greedy sampler.

### 3.3 Correctness and labels

A prediction is correct if, after normalisation (lower-casing, removing punctuation and the articles *a*, *an*, *the*, collapsing whitespace), it is non-empty and some normalised reference alias equals it or appears in it as a whole-word match (strict). After a first look at the results we added a plural-tolerant rescore (`rescore.py`), which also accepts plural/singular variants; we report both strict and lenient accuracy, and the lenient matcher was chosen post hoc. Labels and all analyses use the lenient matcher. The risk label is y = 1 when the *draft* answer is incorrect. Labels come from ground-truth correctness only; semantic-entropy labels are not used in this pilot (Section 4.7).

### 3.4 Gates

Each gate produces a risk score where higher means more likely wrong, and the cascade escalates the highest-scoring fraction *r* of the test queries.

- **Random.** Uniform random scores (lower reference).
- **Oracle.** The label y plus noise of magnitude below 10^-3 to break ties; escalates truly wrong draft answers first (upper reference).
- **Token log-probability.** Negative mean and negative minimum log-probability of the generated tokens.
- **Token entropy.** Mean and maximum, over generated tokens, of the entropy of the next-token distribution.
- **Hidden-state probe (post-generation).** A logistic-regression probe on the draft model's hidden state at either the last prompt token or the last answer token, at a single transformer layer. Hidden states are captured inside the draft's own greedy generation loop (the output of every decoder layer at the last prompt token and the last answer token, stored in 16-bit floating point), so no extra forward pass is needed. Features are standardised (zero mean, unit variance, fitted on the training data) and the classifier is L2-regularised logistic regression (C = 0.05, up to 2,000 iterations).
- **Prefill-only probe (pre-generation router).** The same probe type on the last-prompt-token state, with the layer chosen among prompt-position layers only. Because the prompt-position state is available from the draft's own prefill, this gate can decide before any answer is generated; we evaluate it as a router (below).

**Layer and position selection.** The probe's position (prompt or answer) and layer are chosen by 5-fold cross-validated AUROC on the training portion only, over every second layer; the test split is not used for selection. The prefill-only probe uses the best prompt-position layer by the same criterion.

**Gate cost.** Hidden states are read during generation, inside the metered window, and the probe is a linear head, so gate energy is approximately zero and there is no separately timed gate (E_g = 0 throughout). Confidence gates, the random and oracle references are likewise charged zero.

**Router policy.** For the prefill-only probe used as a router, a query scored in the top fraction *r* is sent to the 7B without waiting for the 3B answer. It still pays the 3B prefill E_p (needed to read the hidden state), measured per question with `max_tokens = 1` (a slight over-estimate of a pure prefill), plus the 7B; non-escalated queries pay the 3B answer E_s. The router break-even is p*_router = (E_l - E_s) / (E_l - E_s + E_p), and we report the prefill share E_p/E_s.

### 3.5 Energy measurement

Power is sampled without root privileges using `macmon` (an open-source tool reading the macOS IOReport and SMC interfaces) at 100 ms intervals. Each sample is the mean power over the interval ending at its timestamp, so window energy is the sum over samples of power times the overlap of the sample's interval with the query window (not a trapezoidal integral). Energy is SoC (CPU + GPU + ANE) plus DRAM power; whole-system SMC power is also recorded but is not used for the primary results. System power is not wall-plug power and has not been validated against an external meter.

*Cost basis.* Our first sequential runs (one model after the other, one query at a time) drifted: the 3B per-query energy rose from 1.07 to 1.67 J across its run, and SMC `sys_power` idle drifted between 2.3 and 4.9 W. The primary energies therefore come from a dedicated run (`cost.py`) on the 300 test questions with both models resident. For each question, the 3B answer, the 7B answer and the 3B prefill-only call (`max_tokens = 1`) are executed in random order (seed 0), so thermal and power state is shared across policies. Idle power is measured for 20 s before and after the loop and the mean is subtracted (idle power times each call's duration) to give net energy; we report both net and gross. The first ~75 questions of this run were inflated (warm-up and concurrent desktop use), so we also report medians. Per-query energies from the sequential runs are kept only as a secondary check.

### 3.6 Offline replay, break-even and statistics

Escalation rate *r* is swept over 21 values from 0 to 1 in steps of 0.05, and for each gate the top-*r* fraction of the test queries by risk score is escalated. We report cascade accuracy, mean energy per query, accuracy retained relative to always-large, and, for retention thresholds of 95% and 100%, the smallest swept escalation rate that reaches the threshold and the corresponding energy saving relative to always-large.

The break-even escalation rate is the rate at which cascade energy equals always-large energy:

p* = 1 - (E_s + E_g) / E_l,

with E_s, E_g, E_l the test-set means (E_g = 0, so the with- and without-gate values coincide); the router break-even is given in Section 3.4.

Uncertainty is quantified with the bootstrap. Gate AUROCs have percentile 95% CIs from 1,000 resamples of the test queries. To compare the probe with the best confidence baseline (selected by AURC), we compute the paired difference in AURC over 1,000 bootstrap resamples (drawn with replacement, same resample for both gates) and report the mean and 95% percentile interval. AURC here is the area under the accuracy-versus-escalation curve (trapezoid over the 21 rates), not a risk-coverage area. We likewise report the paired ΔAUROC of the probe against mean token entropy over 1,000 resamples. A gap whose interval contains zero is reported as no detectable difference.

### 3.7 Token-level energy model

To separate energy from query mix and machine state, we fit per model E_net ≈ c + a·P + b·O, with P prompt tokens and O generated tokens, for Qwen2.5-Instruct 0.5B, 1.5B, 3B and 7B (4-bit MLX). Each model is swept over P ∈ {16, 64, 256, 1024} and O ∈ {1, 8, 32, 128} (`generate_step`, which ignores end-of-sequence, so exactly O tokens are decoded), with three repeats in random cell order. To stay within 16 GB, one model is resident at a time, the MLX buffer cache is capped at 256 MB and cleared after every call, and peak memory is logged (4.5 GB for the 7B). Before each model block the GPU cools below 60 °C and idle power is measured for 15 s; model blocks run in random order. Coefficients come from ordinary least squares. For validation, the 3B and 7B blocks each end with 40 held-out real chat queries (the last 40 test questions, stock `stream_generate`, maximum 24 tokens), with O counted as answer tokens plus the end-of-sequence step. Scaling exponents are log-log slopes of a and b against parameter count.

### 3.8 Batching and transfer checks

*Batch energy.* For one model at a time, the last 64 questions of the shuffled set are generated greedily (maximum 24 tokens) with `mlx-lm` batched generation at batch size 1 and at batch size 8, three repeats each, after a warm-up and a 15 s idle measurement. We report net energy per query (system and SoC) and wall time for each batch size. Batch 1 always ran before batch 8 within a repeat, and these runs predate the DRAM-inclusive meter, so their SoC figures exclude DRAM.

*Transfer.* The probe (at the selected layer and position) is trained on TriviaQA training queries only and evaluated on SciQ test queries, and compared with the AUROC on the same SciQ test queries of the probe trained on the pooled data.

---

## 4 Experimental Setup

### 4.1 Hardware and software

All experiments run on a single base Apple MacBook with an M2 chip (10-core GPU, 16 GB unified memory). Inference uses MLX and `mlx-lm`; probes use scikit-learn. The models are `mlx-community/Qwen2.5-3B-Instruct-4bit` (draft) and `mlx-community/Qwen2.5-7B-Instruct-4bit` (target) [17]. Energy is sampled with `macmon` [18]. Software: MLX 0.32.4, mlx-lm 0.32.2, macmon 0.10.0, macOS 26.5.2, Python 3.11.17. [TODO: machine conditions (mains power, power mode, other applications closed) for the clean re-run.]

### 4.2 Data

We use 1,000 open-ended factual questions: the first 600 rows of the TriviaQA `rc.nocontext` validation split [19] and the first 400 rows of the SciQ test split [20], with SciQ used as open-ended QA (the question only, without the multiple-choice options or support passage). Reference answers are the normalised aliases and normalised value for TriviaQA and the lower-cased correct answer for SciQ. The combined set is shuffled with seed 0 and used in that order. Both models see identical prompts.

### 4.3 Splits

The first 60% of the shuffled questions form the training split, the next 10% a calibration split, and the final 30% the test split. In the analysis the calibration split is merged into training (70% fit, 30% test) because the escalation threshold is set by escalation rate rather than by a probability cut-off. The test split is not used for layer selection, probe fitting or any other choice. All reported test statistics are on the 30% split; this is n_test = 300 queries (about 195 draft errors), so confidence intervals are wide.

### 4.4 Procedure

Each model is run once over all 1,000 questions, one query at a time (batch size 1), for answers, correctness, token counts, latency and token-level uncertainty (mean, minimum and sum of token log-probabilities; mean and maximum token entropy). For the draft model, hidden states for all layers at the last prompt token and last answer token are recorded during generation. Analysis restricts to queries answered by both models. Per-query energy for the 300 test questions comes from the separate interleaved run of Section 3.5.

### 4.5 Baselines and metrics

Gates are those of Section 3.4. Metrics are: AUROC of the gate score for predicting draft error; accuracy of the cascade at each escalation rate; accuracy retained relative to always-large; area under the accuracy-versus-escalation curve; net energy per query and its Pareto relation to accuracy; the smallest swept escalation rate and the energy saving at 95% and 100% accuracy retention; the break-even escalation rate p* (cascade and router); the router's prefill share; and per-dataset accuracy and probe AUROC. Always-small (r = 0) and always-large (r = 1) are the endpoints of every curve.

### 4.6 Implementation notes affecting interpretation

- Hidden states are collected inside the draft's generation loop; there is no separately timed gate and gate energy is taken as zero.
- Instrumentation differs between runs: `run.py` uses a Python generation loop with probe instrumentation, whereas `cost.py` uses stock `mlx-lm`. Energies and the cascade replay use `cost.py`; answers, correctness and uncertainty come from `run.py`.
- The router charges E_p measured with `max_tokens = 1` in the interleaved run, a slight over-estimate of a pure prefill.
- Per-query windows are short (about 0.6-1.8 s), so each contains few 100 ms power samples and per-query energy is noisy. We report medians alongside means; no repeated-run variance was measured.

### 4.7 Out of scope for this pilot

Semantic-entropy labels (10 sampled answers per query with entailment clustering) are planned as future work, to be generated on cloud GPUs, and are not used here. Also not included: NVIDIA GPU measurements, wall-plug validation, cross-model transfer, other model pairs, a live deployment study, and a calibration analysis of the probe.

---

## 5 Results

**Models and cost.** On the 300 test questions the 3B draft and 7B target reach 35.0% and 51.7% accuracy (strict matching: 33.3% and 49.0%), so about 195 draft answers are wrong. Table 1 gives the interleaved cost run. The 3B costs 1.13 J net per query, 43.8% of the 7B's 2.58 J (gross: 2.77 vs 6.37 J); 3B prefill alone costs 0.77 J net, 68% of a 3B answer. Idle power was 2.38 W before and 1.94 W after the run. The first ~75 questions were inflated (3B net mean 1.61 J over the first 75 vs 0.97 J over the rest), and medians are lower than means (3B 0.93 J, 7B 2.21 J, prefill 0.62 J net over all 300 questions).

These absolute values turned out to be too low (Section 5, token-level energy model); the replay depends only on relative costs, which agree with the clean re-measurement.

*Table 1. Per-query cost, interleaved run, 300 test questions (mean).*

| Job | Gross J | Net J | Seconds |
|---|---|---|---|
| 3B answer | 2.77 | 1.13 | 0.76 |
| 7B answer | 6.37 | 2.58 | 1.75 |
| 3B prefill only | 2.04 | 0.77 | 0.59 |

**Gate quality.** Table 2 gives AUROC for predicting draft errors. The best CV-selected probe (answer position, layer 22) reaches 0.806 (95% CI 0.750-0.857), indistinguishable from mean token entropy (0.804, 0.748-0.855), mean log-probability (0.791, 0.739-0.843) and the other confidence scores. The paired ΔAUROC of the probe against entropy is +0.002 (95% CI -0.060 to +0.057), and the paired ΔAURC against the best AURC baseline (mean log-probability) is -0.002 (95% CI -0.019 to +0.015). With n_test = 300 (about 195 draft errors), differences below about 0.06 AUROC cannot be resolved. The probe's cross-validated training AUROC was 0.855, against 0.806 on test, a gap consistent with winner's-curse selection of the layer and position. The prefill-only probe (layer 32) is weaker, with AUROC 0.727 (0.670-0.780).

*Table 2. Gate AUROC (test, 95% bootstrap CI) and AURC (area under accuracy vs escalation).*

| Gate | AUROC | 95% CI | AURC |
|---|---|---|---|
| Random | 0.480 | 0.409-0.550 | 0.434 |
| Oracle | 1.000 | 1.000-1.000 | 0.481 |
| Neg. mean log-prob | 0.791 | 0.739-0.843 | 0.449 |
| Neg. min log-prob | 0.786 | 0.729-0.835 | 0.448 |
| Mean entropy | 0.804 | 0.748-0.855 | 0.446 |
| Max entropy | 0.802 | 0.749-0.855 | 0.445 |
| Probe (answer, L22) | 0.806 | 0.750-0.857 | 0.447 |
| Prefill probe (L32) | 0.727 | 0.670-0.780 | 0.437 |

**Break-even.** With E_g = 0, the cascade break-even escalation is 56.2% (the same with and without gate cost). The router, which pays the 3B prefill (68% of a 3B answer) on escalated queries instead of the full draft, breaks even at 65.3%. At full escalation the cascade costs 3.71 J per query (44% more than always-7B) and the router 3.35 J (30% more). An earlier estimate of the prefill share (`prefill_cost.json`, 0.71) is superseded by the interleaved measurement (0.68).

**Operating points.** Table 3 gives accuracy retained (relative to always-7B) and energy saving versus always-7B at escalation 0 to 0.5.

*Table 3. Retained accuracy (%) and saving vs always-7B (%), by escalation rate.*

| Escalation | Probe cascade: retained / saving | Oracle cascade: retained / saving | Router (prefill probe): retained / saving |
|---|---|---|---|
| 0 | 67.7 / 56.2 | 67.7 / 56.2 | 67.7 / 56.2 |
| 0.1 | 72.3 / 47.2 | 75.5 / 45.6 | 71.6 / 48.1 |
| 0.2 | 75.5 / 37.2 | 79.4 / 35.8 | 74.8 / 39.1 |
| 0.3 | 81.9 / 27.0 | 83.9 / 25.2 | 78.7 / 31.1 |
| 0.4 | 84.5 / 17.6 | 91.0 / 14.8 | 82.6 / 20.5 |
| 0.5 | 85.8 / 7.6 | 97.4 / 5.0 | 86.5 / 12.7 |

Retaining at least 95% of the 7B's accuracy requires about 70% escalation with the probe, at a saving of -12% (energy-negative), and 45% escalation even with an oracle, at a saving of about 10%; the router needs 85% (saving -17%). On this pair and hardware, energy savings therefore come only with accuracy loss: for example, the probe cascade saves about 18% at about 85% retained accuracy. The oracle cascade reaches 100% retention at 55% escalation and can exceed the 7B's accuracy (the two models err on different questions), visible in `results/pareto.png`; `results/auroc_by_layer.png` shows probe AUROC by layer.

**Batching.** Table 4 gives batch 1 vs 8 (64 questions, mean of three repeats, SoC net). Batch 8 reduces per-query SoC energy by 8.8% (3B) and 5.6% (7B) and wall time by 31% (3B) and 34% (7B). Batch 1 always ran before batch 8, and these runs predate the DRAM-inclusive meter.

*Table 4. Batching, 64 questions, mean of 3 repeats.*

| Model | Batch | SoC net J/query | Wall time (s) |
|---|---|---|---|
| 3B | 1 | 1.19 | 37.3 |
| 3B | 8 | 1.09 | 25.9 |
| 7B | 1 | 2.19 | 82.1 |
| 7B | 8 | 2.07 | 54.1 |

**Token-level energy model.** Table 5 gives the fitted coefficients. The linear model explains the sweeps well (R² 0.95-1.00). Energy per output token rises from 0.075 J (0.5B) to 0.70 J (7B), scaling as N^0.81; energy per prompt token rises from 0.004 to 0.076 J, scaling as N^1.07; an output token costs about 9-19x a prompt token (Figure: `results/energy_scaling.png`). On the held-out real queries the model predicts 7B net energy within -6% (per-query error 10%) and 3B within -20% (per-query error 20%), underestimating the per-call overhead of very short 3B calls. For a typical test query (45 prompt tokens, about 4 answer tokens), prompt and decode contribute comparably (7B: 3.4 J and 3.1 J, plus a 1.0 J intercept).

The held-out queries also re-measure the cascade's relative cost under cleaner conditions: 3.28 J (3B) and 8.03 J (7B) net, a ratio of 0.41 against 0.44 in Table 1. The interleaved run's absolute energies were therefore about 2.5-3x too low. In that run calls took about 2.6x longer at about 3.6 W gross (versus about 13 W now), so subtracting idle power times duration removed most of each call's energy. An in-process A/B check showed that stock `stream_generate` and `generate_step` draw the same energy for the same tokens, so the difference comes from machine state, not code path. With the new ratio, cascade break-even moves from 56% to 59%; with the model's prefill share (0.55) the router break-even is about 73%. The conclusions of this section do not change.

*Table 5. Token-level energy model (net J, SoC + DRAM; 3 repeats of a 4x4 grid).*

| Model | Params (B) | c (J) | a: J/prompt token | b: J/output token | R² |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.49 | -0.25 | 0.004 | 0.075 | 0.995 |
| Qwen2.5-1.5B | 1.54 | -0.15 | 0.018 | 0.181 | 0.948 |
| Qwen2.5-3B | 3.09 | 0.04 | 0.032 | 0.313 | 0.998 |
| Qwen2.5-7B | 7.62 | 1.04 | 0.076 | 0.700 | 0.962 |

**Transfer.** A probe trained on TriviaQA only reaches AUROC 0.801 on the SciQ test queries (n = 134), against 0.780 for the pooled probe on the same queries. This is consistent with, but not distinguishable from, training-free baselines at this sample size. Per dataset, the 3B/7B accuracies are 41.8%/61.2% on SciQ and 29.5%/44.0% on TriviaQA, and the pooled probe's AUROC is 0.780 and 0.811.

### 5.x GPU replication: four model sizes, five pairs, semantic entropy

To test whether the M2 result is specific to one machine and one pair, we re-ran the pipeline on Kaggle (2x NVIDIA T4, fp16, `transformers`). The run used Qwen2.5 0.5B/1.5B/3B/7B-Instruct on 3,000 questions (1,000 each from TriviaQA, SciQ and NQ-open). Probes are fit on the first 70% and everything is reported on the last 900. Semantic entropy (SE) clusters K = 10 samples at T = 1 by normalised-answer match. Energy is the NVML hardware counter summed over both GPUs, measured per query at batch 1 on 100 real test queries. We use gross joules because the idle baseline was read on a hot GPU (44-69 W against 19.9 W at true idle), which made idle-subtracted energy unreliable (Section 6).

*Table 6. Cascade economics per pair (test, n = 900). Saving is relative to always using the large model; negative values cost more.*

| Pair | Acc draft / large | J/query draft / large | Break-even | AUROC entropy / SE / probe | Saving @95% (entropy / probe / oracle) | Best saving @90% (entropy / probe) |
|---|---|---|---|---|---|---|
| 0.5b → 7b | 17.0% / 44.3% | 12.1 / 38.8 | 69% | 0.744 / 0.818 / 0.762 | -26% / -21% / -6% | -16% / -16% |
| 1.5b → 7b | 28.6% / 44.3% | 16.1 / 38.8 | 58% | 0.808 / 0.836 / 0.836 | -22% / -12% / +3% | -2% / -2% |
| 3b → 7b | 34.6% / 44.3% | 17.9 / 38.8 | 54% | 0.802 / 0.814 / 0.812 | +4% / -1% / +19% | +14% / +19% |
| 0.5b → 3b | 17.0% / 34.6% | 12.1 / 17.9 | 32% | 0.744 / 0.818 / 0.762 | -58% / -58% / -33% | -43% / -43% |
| 1.5b → 3b | 28.6% / 34.6% | 16.1 / 17.9 | 10% | 0.808 / 0.836 / 0.836 | -40% / -25% / -10% | -15% / -10% |

*Table 7. Gate quality against token entropy (paired bootstrap, 500 resamples). SE energy charges its 10 draft samples.*

| Draft | ΔAUROC SE − entropy [95% CI] | ΔAUROC probe − entropy [95% CI] | SE saving @95% (10 samples charged) |
|---|---|---|---|
| 0.5b | +0.074 [+0.026, +0.119] | +0.019 [-0.038, +0.073] | -333% |
| 1.5b | +0.029 [+0.005, +0.055] | +0.029 [-0.011, +0.065] | -432% |
| 3b | +0.012 [-0.010, +0.035] | +0.009 [-0.026, +0.045] | -466% |

The M2 conclusion holds on every pair: retaining 95% of the large model's accuracy is energy-negative for all real gates except entropy on 3B→7B (+4%), and even an oracle saves at most 19%. The 3B→7B pair, the closest analogue of the M2 pilot, saves 14-19% only at 90% retention. Semantic entropy is the best detector. Its advantage over token entropy is significant for the 0.5B (+0.074) and 1.5B (+0.029) drafts and shrinks as the draft grows. However, sampling 10 extra answers makes the cascade cost 4-10x as much as always using the large model, so its better ranking cannot pay for itself. The hidden-state probe again matches token entropy (all CIs include 0). Weak drafts make cascading worse, not better: the 0.5B is cheap (break-even 69%) but answers too few questions correctly (17.0% against 44.3%) to avoid heavy escalation.

## 6 Discussion & Limitations

The probe adds no measurable detection power over token entropy or log-probability on this pair, so its practical advantage is limited to reading a signal already computed during generation at no extra cost. The more consequential finding is economic: the 3B costs 44% of the 7B per query, the accuracy gap is 16.7 points, and escalated queries pay twice. Because answers are short (about 4 tokens), each call is dominated by prompt processing plus a handful of decode steps. The token model shows that per-token energy scales almost in proportion to size (N^0.81 to N^1.07), so the 3B saves only what its 2.5x smaller size allows (a 0.41 energy ratio) while recovering just 68% of the 7B's accuracy. With few decode steps per call, batching also helps little. The token model also makes the economics portable: break-even for any pair and prompt/answer length can be computed from (c, a, b) without re-running the cascade, and for long answers the ratio approaches b_small/b_large (0.45 for 3B/7B, 0.26 for 1.5B/7B). Cascading pays off only if one accepts accuracy loss; high retention is energy-negative even for a perfect gate. The oracle result also shows complementarity between the models, which a stronger gate could exploit.

*Limitations.* (i) One machine, one model pair, one dataset mix and n_test = 300. (ii) Short answers, so conclusions may not transfer to long-form generation, where decode dominates. (iii) `macmon` power is not validated against a wall meter. (iv) Instrumentation differs between runs: a Python loop in `run.py` versus stock `mlx-lm` in `cost.py`. (v) The lenient plural-tolerant rescoring was decided post hoc after seeing results. (vi) Semantic entropy is measured only in the GPU run, with K = 10 sequential samples charged as an upper bound. (vii) The first ~75 interleaved questions were inflated by warm-up and concurrent desktop use, and the probe winner's-curse gap (0.855 CV vs 0.806 test) suggests mild optimism in layer selection. (viii) Batch comparisons have a fixed order and a different meter than the primary energies. (ix) The interleaved run's absolute energies were about 2.5-3x too low (Section 5); only ratios from it are used. The token-model sweep ran with the GPU at 87-94 °C during the 3B and 7B blocks and on battery power, and its idle baseline differs by block (1.5-2.0 W). In the GPU run, idle was read at 73-76 °C (44-69 W against 19.9 W at true idle), so only gross energy is used there. Gross energy charges small models for an idle second GPU. Subtracting the true idle instead moves break-even to 77/63/59% (→7B) and 43/9% (→3B) without changing any conclusion. The GPU linear token model misses the 7B by -39% on real queries, so per-query measured means are used.

*Future work.* A token-level energy model E ≈ c + a·prompt_tokens + b·output_tokens that also validates on GPU; batched semantic-entropy sampling with its energy measured; a larger n; a bigger accuracy gap (a 1B draft or a 70B target); wall-plug validation; and unattended clean runs.

## 7 Conclusion

On a base M2 with Qwen2.5-3B and 7B, a hidden-state probe is statistically indistinguishable from entropy and log-probability for flagging draft errors (ΔAUROC +0.002, 95% CI -0.060 to +0.057), and the 3B draft costs about 44% of the 7B. Energy-positive cascading requires escalating less than 56% of queries (65% with a router) and therefore sacrifices accuracy: about 18% saving at about 85% retained, and retaining 95% of the 7B's accuracy is energy-negative with the probe and saves only about 10% with an oracle. A token-level energy model (J per output token ∝ N^0.81) reproduces held-out query energy within 6-20% and lets break-even be computed for other pairs and answer lengths. A GPU replication across five Qwen2.5 pairs (0.5B-7B) and semantic entropy confirms the result: semantic entropy ranks errors best but its sampling cost outweighs any saving. Larger accuracy gaps, longer outputs and better-validated power measurement are the next steps.

---

## References

[1] L. Huang et al., "A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions," *ACM Trans. Inf. Syst.*, vol. 43, no. 2, pp. 1-55, 2025, doi: 10.1145/3703155.

[2] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, "Detecting hallucinations in large language models using semantic entropy," *Nature*, vol. 630, no. 8017, pp. 625-630, 2024, doi: 10.1038/s41586-024-07421-0.

[3] M. F. Argerich and M. Patino-Martinez, "Measuring and improving the energy efficiency of large language models inference," *IEEE Access*, vol. 12, pp. 80194-80207, 2024, doi: 10.1109/ACCESS.2024.3409745.

[4] S. Luccioni, Y. Jernite, and E. Strubell, "Power hungry processing: Watts driving the cost of AI deployment?," in *Proc. ACM Conf. Fairness, Accountability, and Transparency (FAccT)*, 2024, pp. 85-99, doi: 10.1145/3630106.3658542.

[5] S. Samsi et al., "From words to watts: Benchmarking the energy costs of large language model inference," in *Proc. IEEE High Performance Extreme Computing Conf. (HPEC)*, 2023, pp. 1-9, doi: 10.1109/HPEC58863.2023.10363447.

[6] L. Chen, M. Zaharia, and J. Zou, "FrugalGPT: How to use large language models while reducing cost and improving performance," *Trans. Mach. Learn. Res.*, 2024.

[7] P. Aggarwal et al., "AutoMix: Automatically mixing language models," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

[8] N. Gupta, H. Narasimhan, W. Jitkrittum, A. S. Rawat, A. K. Menon, and S. Kumar, "Language model cascades: Token-level uncertainty and beyond," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2024.

[9] D. Ding et al., "Hybrid LLM: Cost-efficient and quality-aware query routing," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2024.

[10] Z. Ji et al., "Survey of hallucination in natural language generation," *ACM Comput. Surv.*, vol. 55, no. 12, pp. 1-38, 2023, doi: 10.1145/3571730.

[11] J. Kossen, J. Han, M. Razzak, L. Schut, S. Malik, and Y. Gal, "Semantic entropy probes: Robust and cheap hallucination detection in LLMs," ICML 2024 Workshop on Foundation Models in the Wild, arXiv:2406.15927, 2024.

[12] M. A. R. Siddiqui, D. Rojas, C. Yang, W. Cui, Y. Shi, and Y. Chen, "Measured joules, learned routes: Learning to route for energy-efficient LLM serving," arXiv:2609.23085, Sep. 2026.

[13] Z. Wang, S. Li, P. Song, S. Chen, Q. Song, and Q. Liu, "Signed rescue routing: Harm-aware cascades for efficient LLM inference," arXiv:2609.07786, Sep. 2026.

[14] V. Kotte, "UCCI: Calibrated uncertainty for cost-optimal LLM cascade routing," arXiv:2605.18796, May 2026.

[15] Z. Mrykhin and V. Malykh, "Prompt embedding probes (PEP): Hallucination detection in LLMs from hidden states," arXiv:2608.08024, Aug. 2026.

[16] Y. Zeng, C. Wang, B. J. Chen, and L. Lin, "ReLope: From hidden-state probing to a decision module for multimodal LLM routing," arXiv:2603.24787, Mar. 2026.

[17] Qwen Team, "Qwen2.5 technical report," arXiv:2412.15115, 2024. Models used via the `mlx-community` 4-bit MLX conversions on Hugging Face.

[18] MLX and `mlx-lm`: Apple machine learning research, open-source array framework and LLM library for Apple silicon, https://github.com/ml-explore/mlx. `macmon`: open-source Apple-silicon power monitor, https://github.com/vladkens/macmon. Versions MLX 0.32.4, mlx-lm 0.32.2, macmon 0.10.0; accessed Nov. 2026.

[19] M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer, "TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension," in *Proc. ACL*, 2017, pp. 1601-1611.

[20] J. Welbl, N. F. Liu, and M. Gardner, "Crowdsourcing multiple choice science questions," in *Proc. 3rd Workshop on Noisy User-generated Text (W-NUT)*, 2017, pp. 94-106.
