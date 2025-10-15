# 簇内精排 + MMR 伪代码

```python
ranked = [center]
selected = set([center])
while len(ranked) < N and candidates:
    best = argmax_cand(lambda c: rel(center, c) - (1-λ)*max_sim(c, selected))
    ranked.append(best)
    selected.add(best)
```
