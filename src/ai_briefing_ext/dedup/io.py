"""
JSONL helpers (used by tests/CLI).
"""
from __future__ import annotations
import json, typing as t

def read_jsonl(path: str) -> t.List[dict]:
    return [json.loads(l) for l in open(path, "r", encoding="utf-8")]

def write_jsonl(path: str, rows: t.Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
