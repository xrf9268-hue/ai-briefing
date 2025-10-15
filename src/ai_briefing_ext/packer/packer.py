"""
Context packer with token budgeting and per-cluster quotas.
CLI:
  python -m ai_briefing_ext.packer.packer --in clusters_ranked.jsonl --out packed.json --budget 6000 --min 300 --max 1200
Input: JSONL, each line: {"topic_id","label","items":[{"text","urls","score"}]}
"""
from __future__ import annotations
import argparse, json, typing as t, re, time
from dataclasses import dataclass, field

def try_token_len(text: str) -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # heuristic: ~4 chars per token
        return max(1, int(len(text) / 4))

def sent_split(text: str) -> t.List[str]:
    sents = re.split(r"(?<=[。！？.!?])\s+", text.strip())
    return [s for s in sents if s]

@dataclass
class Excerpt:
    text: str
    urls: t.List[str] = field(default_factory=list)

@dataclass
class PackedTopic:
    topic_id: str
    label: str
    excerpts: t.List[Excerpt] = field(default_factory=list)

def pack_cluster(items: t.List[dict], min_tokens: int, max_tokens: int) -> t.List[Excerpt]:
    used = set()
    total = 0
    out: t.List[Excerpt] = []
    for it in items:
        sents = sent_split(it.get("text",""))[:8]
        urls = it.get("urls") or []
        for s in sents:
            s_norm = s.strip()
            if s_norm in used: 
                continue
            tl = try_token_len(s_norm)
            if total + tl > max_tokens and total >= min_tokens:
                return out
            out.append(Excerpt(text=s_norm, urls=urls))
            used.add(s_norm)
            total += tl
    return out

def pack(clusters: t.List[dict], token_budget: int, per_cluster_min: int, per_cluster_max: int, title: str, date_iso: str) -> dict:
    remaining = token_budget
    topics: t.List[PackedTopic] = []
    for c in clusters:
        cap = min(per_cluster_max, remaining) if remaining > per_cluster_min else remaining
        ex = pack_cluster(c.get("items", []), per_cluster_min, cap)
        topics.append(PackedTopic(topic_id=c.get("topic_id"), label=c.get("label",""), excerpts=ex))
        used = sum(try_token_len(e.text) for e in ex)
        remaining = max(0, remaining - used)
        if remaining <= 0:
            break
    return {
        "title": title,
        "date": date_iso,
        "topics": [{
            "topic_id": t.topic_id,
            "label": t.label,
            "excerpts": [{"text": e.text, "urls": e.urls} for e in t.excerpts]
        } for t in topics]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--min", dest="minc", type=int, default=300)
    ap.add_argument("--max", dest="maxc", type=int, default=1200)
    ap.add_argument("--title", default="Daily Engineering Briefing")
    ap.add_argument("--date", default="1970-01-01T00:00:00Z")
    args = ap.parse_args()
    clusters = [json.loads(l) for l in open(args.inp, "r", encoding="utf-8")]
    packed = pack(clusters, token_budget=args.budget, per_cluster_min=args.minc, per_cluster_max=args.maxc, title=args.title, date_iso=args.date)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(packed, f, ensure_ascii=False, indent=2)
    print(f"topics: {len(packed['topics'])}, tokens<=~{args.budget}")

if __name__ == "__main__":
    main()
