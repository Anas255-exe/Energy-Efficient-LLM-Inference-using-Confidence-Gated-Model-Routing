"""Fetch small QA subsets via the HF datasets-server rows API -> data/questions.jsonl"""
import json, random, requests, truststore
truststore.inject_into_ssl()  # corporate proxy: use macOS trust store

API = "https://datasets-server.huggingface.co/rows"

def rows(dataset, config, split, n):
    out = []
    for off in range(0, n, 100):
        r = requests.get(API, params=dict(dataset=dataset, config=config, split=split, offset=off, length=min(100, n - off)), timeout=120)
        r.raise_for_status()
        out += [x["row"] for x in r.json()["rows"]]
    return out

qs = []
for r in rows("mandarjoshi/trivia_qa", "rc.nocontext", "validation", 600):
    a = r["answer"]
    qs.append(dict(id=f"tqa-{r['question_id']}", dataset="triviaqa", question=r["question"],
                   answers=sorted(set(a["normalized_aliases"] + [a["normalized_value"]]))))
for i, r in enumerate(rows("allenai/sciq", "default", "test", 400)):
    qs.append(dict(id=f"sciq-{i}", dataset="sciq", question=r["question"], answers=[r["correct_answer"].lower()]))

random.Random(0).shuffle(qs)
with open("data/questions.jsonl", "w") as f:
    for q in qs:
        f.write(json.dumps(q) + "\n")
print(len(qs), "questions")
