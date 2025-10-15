# 两级去重伪代码（指纹 → 语义）

```python
# 早期指纹
families = LSHSimHash.build(items, key=lambda x: x.canonical_text)
representatives = []
for fam in families:
    rep = select_representative(fam)   # 信息密度 + 来源可信 + 新鲜度
    rep.sources = merge_sources(fam)
    representatives.append(rep)

# 语义层（ANN）
index = ANN.build([emb(rep) for rep in representatives])
groups = []
visited = set()
for i, rep in enumerate(representatives):
    if i in visited: 
        continue
    neigh = index.neighbors(rep.vector, topk=topk)
    group = [rep]
    for j, score in neigh:
        if score >= τ_sem:
            group.append(representatives[j])
            visited.add(j)
    groups.append(merge_group(group))  # 代表样本 + sources 合并
```
