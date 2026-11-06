"""Re-score saved answers with a plural-tolerant matcher; writes `correct` (lenient) and keeps `correct_strict`.
usage: python src/rescore.py results/3b.jsonl results/7b.jsonl
"""
import json, re, sys
from run import norm

Q = {q["id"]: q for q in map(json.loads, open("data/questions.jsonl"))}


def stem(s):
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in norm(s).split())


def lenient(pred, answers):
    p = stem(pred)
    return bool(p) and any(a and (a == p or re.search(rf"\b{re.escape(a)}\b", p)) for a in map(stem, answers))


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        r.setdefault("correct_strict", r["correct"])
        r["correct"] = lenient(r["answer"], Q[r["id"]]["answers"])
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(path, "strict", sum(r["correct_strict"] for r in rows) / len(rows), "lenient", sum(r["correct"] for r in rows) / len(rows))
