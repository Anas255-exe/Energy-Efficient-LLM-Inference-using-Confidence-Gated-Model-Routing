"""Answer every question once with one model; log answer, correctness, token
uncertainty, per-query measured energy, and (draft model) per-layer hidden states.

usage: python src/run.py --model 3b|7b [--limit N] [--hidden]
"""
import argparse, json, math, re, string, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache

from energy import PowerMeter

MODELS = {"3b": "models/q3b",   # mlx-community/Qwen2.5-3B-Instruct-4bit (src/get_models.py)
          "7b": "models/q7b"}   # mlx-community/Qwen2.5-7B-Instruct-4bit
SYSTEM = "Answer the question with only the short answer (a few words). No explanation."


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(pred, answers):
    p = norm(pred)
    if not p:
        return False
    for a in answers:
        a = norm(a)
        if a and (a == p or re.search(rf"\b{re.escape(a)}\b", p)):
            return True
    return False


def step(model, tokens, cache):
    """One forward over `tokens` with KV cache; returns (logprobs of last pos, per-layer hidden of last pos)."""
    inner = model.model
    h = inner.embed_tokens(mx.array(tokens)[None])
    mask = create_attention_mask(h, cache[0])
    hs = []
    for layer, c in zip(inner.layers, cache):
        h = layer(h, mask, c)
        hs.append(h[0, -1])
    out = inner.norm(h[:, -1])
    logits = inner.embed_tokens.as_linear(out) if model.args.tie_word_embeddings else model.lm_head(out)
    logits = logits[0].astype(mx.float32)
    return logits - mx.logsumexp(logits), hs


def generate(model, tok, prompt, max_tokens, keep_hidden):
    """Greedy decode that keeps the draft's own hidden states, so the probe needs no extra pass.
    Hidden returned as [layers, 2, d]: last prompt token, last answer token."""
    cache = make_prompt_cache(model)
    lp, hs = step(model, prompt, cache)
    h_prompt = hs
    toks, lps, ents, h_ans = [], [], [], hs
    for _ in range(max_tokens):
        t = mx.argmax(lp).item()
        if t in tok.eos_token_ids:
            break
        toks.append(t)
        lps.append(lp[t].item())
        ents.append(-(mx.exp(lp) * lp).sum().item())
        lp, h_ans = step(model, [t], cache)
    if not toks:  # empty answer: score the EOS step
        lps, ents = [lp.max().item()], [0.0]
    hidden = None
    if keep_hidden:
        hidden = mx.stack([mx.stack([a, b]) for a, b in zip(h_prompt, h_ans)]).astype(mx.float16)
        mx.eval(hidden)
        hidden = np.array(hidden)
    return tok.decode(toks), toks, lps, ents, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    qs = [json.loads(l) for l in open("data/questions.jsonl")][: a.limit]
    model, tok = load(MODELS[a.model])
    meter = PowerMeter()

    # warm-up (discarded), then idle baseline
    for q in qs[:3]:
        generate(model, tok, tok.apply_chat_template([{"role": "user", "content": q["question"]}], add_generation_prompt=True), 8, False)
    idle_soc, idle_sys = meter.idle_power(15)
    print(f"idle power soc={idle_soc:.2f}W sys={idle_sys:.2f}W", flush=True)

    name = f"{a.model}{a.tag}"
    rows, hs = [], []
    T0 = time.time()
    for i, q in enumerate(qs):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question"]}],
            add_generation_prompt=True)
        t0 = time.time()
        text, toks, lps, ents, hid = generate(model, tok, prompt, 24, a.hidden)
        t1 = time.time()
        soc, sys_ = meter.energy(t0, t1)
        row = dict(id=q["id"], dataset=q["dataset"], model=a.model, answer=text.strip(),
                   correct=is_correct(text, q["answers"]), n_prompt=len(prompt), n_gen=len(toks),
                   latency=t1 - t0, e_soc=soc, e_sys=sys_,
                   e_soc_net=soc - idle_soc * (t1 - t0), e_sys_net=sys_ - idle_sys * (t1 - t0),
                   lp_mean=float(np.mean(lps)), lp_min=float(np.min(lps)), lp_sum=float(np.sum(lps)),
                   ent_mean=float(np.mean(ents)), ent_max=float(np.max(ents)))
        if a.hidden:
            hs.append(hid)  # captured inside the timed/metered window: probe input costs nothing extra
        rows.append(row)
        if i % 25 == 0:
            acc = np.mean([r["correct"] for r in rows])
            print(f"{i}/{len(qs)} acc={acc:.3f} lat={row['latency']:.2f}s E={row['e_sys']:.1f}J  {q['question'][:50]!r} -> {row['answer'][:30]!r}", flush=True)

    T1 = time.time()
    tot_soc, tot_sys = meter.energy(T0, T1)  # run-level total: per-query sums should match it
    meter.close()
    with open(f"results/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(dict(idle_soc=idle_soc, idle_sys=idle_sys, run_seconds=T1 - T0, run_J_sys=tot_sys, run_J_soc=tot_soc,
                   sum_query_J_sys=sum(r["e_sys"] for r in rows), samples=len(meter.samples)),
              open(f"results/{name}_idle.json", "w"))
    if a.hidden:
        np.save(f"results/{name}_hidden.npy", np.stack(hs))
    print("done", name, "acc", np.mean([r["correct"] for r in rows]))


if __name__ == "__main__":
    main()
