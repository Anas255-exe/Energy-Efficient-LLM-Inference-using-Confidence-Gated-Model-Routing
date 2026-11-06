# %% [markdown]
# # greengate on Kaggle (GPU T4 x2): 4 model sizes, 3,000 questions, semantic entropy, NVML energy
# Settings: Accelerator = GPU T4 x2, Internet = on. "Save & Run All" runs it in the background (~1.5-3 h).
# Everything is written to /kaggle/working after each stage, so a crash keeps finished stages.

# %%
import os, json, re, string, time, random, subprocess, sys
try:
    import pynvml
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py"], check=True)
    import pynvml
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT = "/kaggle/working"
MODELS = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
          "3b": "Qwen/Qwen2.5-3B-Instruct", "7b": "Qwen/Qwen2.5-7B-Instruct"}
DRAFTS = ("0.5b", "1.5b", "3b")          # hidden states + semantic-entropy samples only for drafts
SYSTEM = "Answer the question with only the short answer (a few words). No explanation."
N_PER, K_SE, MAX_NEW = 1000, 10, 24
SMOKE = os.environ.get("GG_SMOKE")  # local CPU check: tiny data, 0.5B only, fake meter
if SMOKE:
    OUT, N_PER, K_SE, MODELS = "smoke", 4, 3, {"0.5b": MODELS["0.5b"]}
    os.makedirs(OUT, exist_ok=True)
GPU = torch.cuda.is_available()
sync = torch.cuda.synchronize if GPU else (lambda: None)
torch.manual_seed(0); random.seed(0)
print(torch.cuda.device_count(), "GPUs", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(pred, answers):
    p = norm(pred)
    return bool(p) and any(a and (a == p or re.search(rf"\b{re.escape(a)}\b", p)) for a in map(norm, answers))

# %% [markdown]
# ## 1. Data: 1,000 each from TriviaQA (rc.nocontext val), SciQ (test+val), NQ-open (val); shuffled, seed 0

# %%
qs = []
tq = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
for i, r in zip(range(N_PER), tq):
    qs.append(dict(id=f"tqa-{i}", dataset="triviaqa", question=r["question"],
                   answers=list({r["answer"]["value"], *r["answer"]["aliases"]})))
sq = [r for sp in ("test", "validation") for r in load_dataset("allenai/sciq", split=sp)][:N_PER]
qs += [dict(id=f"sciq-{i}", dataset="sciq", question=r["question"], answers=[r["correct_answer"]]) for i, r in enumerate(sq)]
nq = load_dataset("google-research-datasets/nq_open", split="validation").select(range(N_PER))
qs += [dict(id=f"nq-{i}", dataset="nq", question=r["question"], answers=r["answer"]) for i, r in enumerate(nq)]
random.Random(0).shuffle(qs)
with open(f"{OUT}/questions.jsonl", "w") as f:
    f.writelines(json.dumps(q) + "\n" for q in qs)
print(len(qs), "questions")

# %% [markdown]
# ## 2. NVML energy meter (sum over all GPUs, hardware energy counter in mJ)

# %%
if GPU:
    pynvml.nvmlInit()
    H = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    energy_mj = lambda: sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in H)
    temp = lambda: max(pynvml.nvmlDeviceGetTemperature(h, 0) for h in H)
else:
    energy_mj, temp = lambda: time.time() * 1000, lambda: 40


def measure(fn):
    sync(); e0, t0 = energy_mj(), time.time()
    out = fn()
    sync(); e1, t1 = energy_mj(), time.time()
    return out, t1 - t0, (e1 - e0) / 1000


def idle_w(sec=10):
    e0, t0 = energy_mj(), time.time(); time.sleep(sec)
    return (energy_mj() - e0) / 1000 / (time.time() - t0)

print("idle W", idle_w(1 if SMOKE else 5), "temp", temp())

# %% [markdown]
# ## 3. Per model: batched greedy answers + token uncertainty + hidden states, semantic-entropy samples,
# ## token-level energy sweep (batch 1) and held-out real-query energy (batch 1)

# %%
PS, OS = ((16, 64), (1, 4)) if SMOKE else ((16, 64, 256, 1024), (1, 8, 32, 128))
filler = " ".join(q["question"] for q in qs)


def prompt_ids(tok, q):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                   add_generation_prompt=True, return_dict=False)  # newer transformers return a BatchEncoding by default


def run_model(name):
    tok = AutoTokenizer.from_pretrained(MODELS[name], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], torch_dtype=torch.float16,
                                                 device_map="auto" if name == "7b" else {"": 0 if GPU else "cpu"}).eval()  # 7B fp16 (15 GB) needs both T4s
    dev = model.get_input_embeddings().weight.device
    bs = {"0.5b": 64, "1.5b": 48, "3b": 32, "7b": 16}[name]
    rows, hid = [], []
    for b in range(0, len(qs), bs):
        batch = qs[b:b + bs]
        enc = tok([tok.decode(prompt_ids(tok, q["question"])) for q in batch], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            # one teacher-forced pass over prompt+answer: token log-probs, entropies, hidden states
            L = enc.input_ids.shape[1]
            mask = torch.cat([enc.attention_mask, torch.ones_like(gen[:, L:])], 1)  # left pads masked; answer tail is causal-safe
            out = model(gen, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0), output_hidden_states=True)
        logp = torch.log_softmax(out.logits[:, L - 1:-1].float(), -1)
        ld, stop = logp.device, torch.tensor(tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]), device=dev)
        for i, q in enumerate(batch):
            ans = gen[i, L:]
            n = int((~torch.isin(ans, stop)).int().cumprod(0).sum())
            text = tok.decode(ans[:n], skip_special_tokens=True).strip()
            k = max(n, 1)
            lp = logp[i, torch.arange(k, device=ld), ans[:k].to(ld)].cpu().numpy()
            ent = -(logp[i, :k].exp() * logp[i, :k]).sum(-1).cpu().numpy()
            rows.append(dict(id=q["id"], dataset=q["dataset"], answer=text, n_tokens=n, P=L - int((enc.attention_mask[i] == 0).sum()),
                             correct=is_correct(text, q["answers"]), lp_mean=float(lp.mean()), lp_min=float(lp.min()),
                             lp_sum=float(lp.sum()), ent_mean=float(ent.mean()), ent_max=float(ent.max())))
            if name in DRAFTS:  # every 2nd layer; last prompt token and last answer token
                pos = L + max(n, 1) - 1
                hid.append(np.stack([np.stack([h[i, L - 1].float().cpu().numpy(), h[i, pos].float().cpu().numpy()])
                                     for h in out.hidden_states[::2]]).astype(np.float16))
        print(name, b, flush=True)
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    if hid:
        np.save(f"{OUT}/{name}_hidden.npy", np.stack(hid))
    print(name, "acc", np.mean([r["correct"] for r in rows]), flush=True)

    if name in DRAFTS:  # discrete semantic entropy: K samples at T=1, clustered by normalized-answer match/containment
        se_rows = []
        for b in range(0, len(qs), bs // 4):
            batch = qs[b:b + bs // 4]
            enc = tok([tok.decode(prompt_ids(tok, q["question"])) for q in batch], return_tensors="pt",
                      padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_p=0.95,
                                     num_return_sequences=K_SE, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for i, q in enumerate(batch):
                samp = [norm(t) for t in texts[i * K_SE:(i + 1) * K_SE]]
                clusters = []
                for s in samp:
                    for c in clusters:
                        if s == c[0] or (s and c[0] and (s in c[0] or c[0] in s)):
                            c.append(s); break
                    else:
                        clusters.append([s])
                p = np.array([len(c) for c in clusters]) / K_SE
                se_rows.append(dict(id=q["id"], se=float(-(p * np.log(p)).sum()), n_clusters=len(clusters),
                                    sample_acc=float(np.mean([is_correct(s, q["answers"]) for s in samp])), samples=samp))
        with open(f"{OUT}/{name}_se.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in se_rows)
        print(name, "SE done", flush=True)

    # token-level energy sweep, batch 1, exactly O new tokens (min_new_tokens forces past EOS)
    ids = tok(filler, add_special_tokens=False).input_ids
    gen1 = lambda x, O: model.generate(x, max_new_tokens=O, min_new_tokens=O, do_sample=False, pad_token_id=tok.pad_token_id)
    gen1(torch.tensor([ids[:64]], device=dev), 8)  # warm-up
    t0 = time.time()
    while temp() > 60 and time.time() - t0 < 300:  # ponytail: capped; v3 hung 11.8 h on a T4 that never cooled below 60 °C
        time.sleep(5)
    print(name, "sweep start temp", temp(), flush=True)
    idle = idle_w(1 if SMOKE else 10)
    sweep, rnd = [], random.Random(0)
    for rep in range(1 if SMOKE else 3):
        for P, O in rnd.sample([(P, O) for P in PS for O in OS], len(PS) * len(OS)):
            x = torch.tensor([ids[:P]], device=dev)
            _, sec, J = measure(lambda: gen1(x, O))
            sweep.append(dict(model=name, P=P, O=O, rep=rep, seconds=sec, J=J, J_net=J - idle * sec, temp=temp(), idle_W=idle))
    val = []
    for q in qs[-(3 if SMOKE else 100):]:  # held-out real chat queries, batch 1, natural stopping
        x = torch.tensor([prompt_ids(tok, q["question"])], device=dev)
        g, sec, J = measure(lambda: model.generate(x, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id))
        val.append(dict(id=q["id"], model=name, P=x.shape[1], O=g.shape[1] - x.shape[1], seconds=sec, J=J, J_net=J - idle * sec))
    with open(f"{OUT}/energy_sweep_{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in sweep)
    with open(f"{OUT}/energy_validate_{name}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in val)
    print(name, "energy done, idle W", round(idle, 1), flush=True)
    del model; torch.cuda.empty_cache()


for name in MODELS:
    if os.path.exists(f"{OUT}/energy_validate_{name}.jsonl"):
        continue
    run_model(name)
print("ALL_DONE")
