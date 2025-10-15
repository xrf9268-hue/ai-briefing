"""
Fingerprint deduplication: canonicalization + SimHash + banded LSH.
CLI:
  python -m ai_briefing_ext.dedup.fingerprint --in items.jsonl --out items.fp.jsonl --bands 8 --bits 64
"""
from __future__ import annotations
import argparse, json, hashlib, re, unicodedata, math
from typing import List, Dict, Tuple

def canonicalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    return text

def simhash(text: str, bits: int = 64) -> int:
    # simple token-based simhash
    v = [0]*bits
    for token in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    fp = 0
    for i, val in enumerate(v):
        if val >= 0:
            fp |= (1 << i)
    return fp

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def band_buckets(simhashes: List[int], bits: int = 64, bands: int = 8) -> Dict[Tuple[int,int], List[int]]:
    r = bits // bands
    buckets = {}
    for idx, sh in enumerate(simhashes):
        for b in range(bands):
            key = (b, (sh >> (b*r)) & ((1<<r)-1))
            buckets.setdefault(key, []).append(idx)
    return buckets

def group_families(items: List[Dict], bits: int = 64, bands: int = 8, ham_thresh: int = 3) -> List[List[int]]:
    texts = [canonicalize(x.get("text","")) for x in items]
    shs = [simhash(t, bits=bits) for t in texts]
    buckets = band_buckets(shs, bits=bits, bands=bands)
    seen = set()
    families = []
    for _, idxs in buckets.items():
        for i in idxs:
            if i in seen: 
                continue
            fam = [i]
            seen.add(i)
            for j in idxs:
                if j in seen: 
                    continue
                if hamming(shs[i], shs[j]) <= ham_thresh:
                    fam.append(j)
                    seen.add(j)
            if len(fam) > 1:
                families.append(sorted(fam))
    # merge overlapping families
    merged = []
    for fam in families:
        added = False
        for m in merged:
            if set(fam) & set(m):
                m[:] = sorted(set(m) | set(fam))
                added = True
                break
        if not added:
            merged.append(fam)
    return merged

def collapse_families(items: List[Dict], families: List[List[int]]) -> List[Dict]:
    id_to_family = {}
    for fid, fam in enumerate(families):
        for idx in fam:
            id_to_family[idx] = fid
    out = []
    used = set()
    for idx, it in enumerate(items):
        if idx in used:
            continue
        if idx in id_to_family:
            fid = id_to_family[idx]
            members = [i for i,j in id_to_family.items() if j==fid]
            # pick representative = longest text
            rep_idx = max(members, key=lambda k: len((items[k].get("text") or "")))
            rep = dict(items[rep_idx])
            rep["family_id"] = f"fam_{fid}"
            rep["rep"] = True
            rep["merged_sources"] = []
            for m in members:
                used.add(m)
                mitem = items[m]
                rep["merged_sources"].append({
                    "id": mitem.get("id"),
                    "url": mitem.get("url"),
                    "source": mitem.get("source"),
                    "ts": mitem.get("ts"),
                })
            out.append(rep)
        else:
            it2 = dict(it)
            it2["family_id"] = None
            it2["rep"] = True
            it2["merged_sources"] = [{"id": it.get("id"),
                                      "url": it.get("url"),
                                      "source": it.get("source"),
                                      "ts": it.get("ts")}]
            used.add(idx)
            out.append(it2)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--bits", type=int, default=64)
    ap.add_argument("--ham", type=int, default=3, help="Hamming threshold within band")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.inp, "r", encoding="utf-8")]
    fams = group_families(items, bits=args.bits, bands=args.bands, ham_thresh=args.ham)
    out = collapse_families(items, fams)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"families: {len(fams)}, out: {len(out)}")

if __name__ == "__main__":
    main()
