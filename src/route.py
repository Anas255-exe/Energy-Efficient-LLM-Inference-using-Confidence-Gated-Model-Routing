"""Live confidence-gated router on Apple M2: Qwen2.5-3B drafts, Qwen2.5-7B answers only when the draft's token entropy
is above a threshold. Every query's SoC+DRAM energy (macmon, idle subtracted) and latency are measured.

The threshold is set for a target accuracy retention on the M2 *fit* split (first 70% of results/3b.jsonl, 7b.jsonl),
so the questions used by --eval (the last 30%) never set it.

usage: python src/route.py "Who wrote Hamlet?"           # one question
       python src/route.py --eval 100                    # router vs always-7B on 100 test questions, interleaved
       options: --retention 0.95 | --t 0.4 (fixed entropy threshold)
"""
import argparse, json, time
import numpy as np
import mlx.core as mx
from mlx_lm import load

from energy import PowerMeter
from run import MODELS, SYSTEM, generate, is_correct

rows = lambda f: [json.loads(l) for l in open(f)]


def calibrate(retention):
    """Smallest top-k escalation keeping `retention` of 7B accuracy on the fit split; returns that entropy threshold."""
    s, l = rows("results/3b.jsonl"), {r["id"]: r for r in rows("results/7b.jsonl")}
    s = [r for r in s if r["id"] in l][:700]
    cs, cl = np.array([r["correct"] for r in s], float), np.array([l[r["id"]]["correct"] for r in s], float)
    ent = np.array([r["ent_mean"] for r in s]); o = np.argsort(-ent, kind="stable")
    acc = (cs.sum() - np.r_[0, np.cumsum(cs[o])] + np.r_[0, np.cumsum(cl[o])]) / len(s)
    k = int(np.argmax(acc >= retention * cl.mean() - 1e-12))
    return np.inf if k == 0 else float(ent[o[k - 1]])


class Router:
    def __init__(self, t):
        mx.set_cache_limit(256 * 2**20)
        mx.set_memory_limit(10 * 2**30)  # 3B + 7B 4-bit ~6 GB resident; leave the rest of 16 GB to macOS
        self.t, self.m = t, {k: load(MODELS[k]) for k in ("3b", "7b")}

    def ask(self, model, question):
        mdl, tok = self.m[model]
        prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
                                         add_generation_prompt=True)
        text, _, _, ents, _ = generate(mdl, tok, prompt, 24, False)
        return text.strip(), float(np.mean(ents))

    def route(self, question):
        answer, ent = self.ask("3b", question)
        if ent < self.t:
            return dict(answer=answer, model="3b", entropy=ent, draft=answer)
        return dict(answer=self.ask("7b", question)[0], model="7b", entropy=ent, draft=answer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--eval", type=int, default=0)
    ap.add_argument("--retention", type=float, default=0.95)
    ap.add_argument("--t", type=float, default=None)
    a = ap.parse_args()
    t = a.t if a.t is not None else calibrate(a.retention)
    print(f"entropy threshold t = {t:.3f} ({'fixed' if a.t is not None else f'{a.retention:.0%} retention on the fit split'})")
    R, meter = Router(t), PowerMeter()
    for w in ("What is the capital of France?", "Who wrote Hamlet?"):  # warm-up, discarded
        R.ask("3b", w); R.ask("7b", w)
    idle, _ = meter.idle_power(10)

    def timed(fn):
        t0 = time.time(); res = fn(); t1 = time.time()
        return res, meter.energy(t0, t1)[0] - idle * (t1 - t0), t1 - t0

    if a.question:
        res, J, s = timed(lambda: R.route(a.question))
        print(json.dumps(dict(**res, joules=round(J, 2), seconds=round(s, 2)), ensure_ascii=False))
    if a.eval:
        qs = rows("data/questions.jsonl")[700:700 + a.eval]
        log = []
        for i, q in enumerate(qs):
            res, J, s = timed(lambda: R.route(q["question"]))
            (big, _), J7, s7 = timed(lambda: R.ask("7b", q["question"]))
            log.append(dict(id=q["id"], model=res["model"], entropy=res["entropy"], correct=is_correct(res["answer"], q["answers"]),
                            J=J, s=s, correct_7b=is_correct(big, q["answers"]), J_7b=J7, s_7b=s7))
            if i % 10 == 0:
                print(f"{i}/{len(qs)} {res['model']} ent={res['entropy']:.2f} J={J:.1f} vs 7B {J7:.1f}", flush=True)
        m =lambda k: float(np.mean([r[k] for r in log]))
        summ = dict(n=len(log), t=t, idle_W=idle, escalated=float(np.mean([r["model"] == "7b" for r in log])),
                    acc_router=m("correct"), acc_7b=m("correct_7b"), retention=m("correct") / max(m("correct_7b"), 1e-9),
                    J_router=m("J"), J_7b=m("J_7b"), saving=1 - m("J") / m("J_7b"), s_router=m("s"), s_7b=m("s_7b"))
        with open("results_eval/router_eval.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in log)
        json.dump(summ, open("results_eval/router_eval_summary.json", "w"), indent=1)
        print(json.dumps(summ, indent=1))
    meter.close()


if __name__ == "__main__":
    main()
