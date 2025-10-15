# Context Packer

- Inputs: clusters with ranked items and merged source URLs.
- Token budget: global budget (e.g., 6k) with per-cluster {min, max}.
- Strategy: iterate clusters by priority, take Top-N sentences from each until hitting cluster cap, sentence-level de-dup, attach merged URLs.
- Output: `{title, date, topics:[{topic_id,label,excerpts:[{text,urls[]}]}]}` for direct prompt injection.
