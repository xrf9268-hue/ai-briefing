# AI‑Briefing 文本聚合→嵌入→去重→话题聚类/重排序→LLM 摘要与要点 改进方案（v1）

**版本**：1.0.0  
**日期**：2025-10-15  
**适用对象**：后端/数据/ML 工程、Prompt 工程、SRE/运维  
**范围**：面向“多源聚合技术简报”流水线（聚合→嵌入→两级去重→聚类→精排→重排序→上下文打包→结构化摘要）  
**非目标**：UI/运营侧大改、引入与业务无关的重型知识库

---

## 0. 执行摘要（Executive Summary）

**核心改造方向（3 维度）**  
1) **质量与可控性**：两级去重（指纹+语义）、聚类稳定化与主题命名、跨编码器精排+MMR、结构化输出与“工程师价值”排序、反幻觉护栏。  
2) **性能与成本**：批量嵌入与缓存、ANN/HNSW 加速、上下文打包器、单/多阶段自适应、失败重试与熔断。  
3) **工程化与运维**：Provider 抽象（OpenAI/Gemini/本地 TEI）、配置中心（Pydantic/YAML）、可观测性（日志/指标/追踪）、成本/质量仪表板、回放评测与灰度发布。

**验收口径（核心指标）**  
- 语义重复 F1↑、误杀↓；聚类纯度/一致性↑；端到端延迟↓；摘要事实错率↓；要点引用覆盖率↑；成本/条目↓。

---

## 1. 现状画像与痛点（简要）
- 聚合：多源、质量参差，格式不统一，早期去重不足；  
- 嵌入：模型/调用单一，批量与缓存不足，长度/语言敏感；  
- 去重：阈值经验化、O(n²) 瓶颈，仅丢弃不合并来源；  
- 聚类/排序：参数敏感，噪声点处理粗放；簇间顺序与“工程价值”弱耦合；  
- 摘要：上下文打包无策略，提示冗长且领域耦合强，结构化输出与反幻觉护栏不足；  
- 工程化：缺少统一 Provider 抽象、端到端可观测与成本治理、回放评测/灰度。

---

## 2. 目标与非目标
**目标**：提升去重/聚类/精排质量、摘要事实性与引用覆盖、整体吞吐与稳定性；降低延迟与 Token 成本；完善可观测/评测/灰度。  
**非目标**：不引入重型检索问答系统；不做 UI 侧大规模改造。

---

## 3. 总体技术方案（总览）

```text
Sources → 规范化(Normalizer) → 早期指纹去重(Fingerprint)
→ 嵌入(Embeddings) → 语义去重(ANN/LSH)
→ 话题聚类(HDBSCAN/KMeans) → 噪声点策略(二次吸附/独立小簇)
→ 簇内精排(Cross‑Encoder + MMR) → 簇间排序(工程师价值多因子)
→ 上下文打包(Context Packer) → LLM 摘要(JSON Schema/只用输入事实)
→ 要点二次去冗 & 引用检查 → 渲染/发布
```

---

## 4. 模块级详细方案

### 4.1 聚合与规范化（Adapters + Normalizer）
- 统一数据模型 `Item(id,title,text,url,author,source,lang,ts,meta)`；
- 规范化：HTML 清洗、空白合并、Unicode NFKC、语言检测、软截断；
- **早期去重（文本指纹）**：exact‑hash + SimHash/MinHash + LSH，合并为“重复族（family）”，保留代表样本+来源集合；
- 多源并发抓取与限流/退避。

**验收**：早期去重后样本减少，无明显误杀，来源合并信息完备。

---

### 4.2 嵌入层（Embedding Layer）
- **Provider 抽象**：TEI 本地（bge/e5/sentence‑transformers）、OpenAI（text‑embedding‑3*）、Gemini（embedding API）；
- 批量与缓存（LRU+持久化）；失败重试（指数退避+抖动）；
- 多语言/超长分块策略；向量存储（FAISS/HNSW/pgvector）。

**验收**：吞吐↑，缓存命中↑，多语言稳定性好。

---

### 4.3 两级去重（指纹 → 语义）
- 语义层用 ANN 近邻（top‑k）+ 阈值 `τ_sem` 聚合相似组；
- 代表样本选择：信息密度/来源可信/新鲜度；
- 合并所有 urls/titles/sources，**不丢信息**。

**验收**：查全率↑误杀↓；代表样本能覆盖重复族关键信息。

---

### 4.4 话题聚类与噪声点策略
- HDBSCAN 调参（`min_cluster_size/min_samples`），或 KMeans+自动选 K；
- 噪声点：不混入同簇；二次吸附（与最近簇中心相似度 ≥ `τ_attach`），否则独立小簇；
- **主题命名（可选 LLM 轻调用）**：3–8 字标签，不引入新事实。

**验收**：聚类纯度/一致性↑；孤立但高价值点不被淹没。

---

### 4.5 重排序（簇内与簇间）
- **簇内**：Cross‑Encoder（bge/cohere/本地）精排 + **MMR** 去冗；
- **簇间**：工程师价值多因子（Utility/Impact/Novelty/Reliability/Recency）。
  - 先在 Pipeline 侧给启发式初分，LLM 端只做同序校正并给“排序理由”（仅内部）。

**验收**：读者主观有用度↑；与工程价值排序一致性↑。

---

### 4.6 上下文打包（Context Packer）
- Token 预算控制：按簇间分值分配配额（min/max）；
- 簇内取 Top‑N + 引用合并；句级去冗，保留版本/指标/限制等“硬信息”。

**验收**：在预算内保持信息覆盖率。

---

### 4.7 摘要与要点生成（结构化输出与反幻觉）
- 严格 **JSON Schema**，仅输出 JSON；
- **事实护栏**：只用输入；每条 bullet 至少 1 个来源；不确定用“仍在评估/尚不确定”；
- 单/多阶段自适应：
  - 单阶段：快、低成本；
  - 多阶段：草拟→去冗→风格/结构化定稿（大样本/高风险时启用）。

**验收**：解析成功率≈100%；引用覆盖率↑；事实性投诉率↓。

---

## 5. Prompt 方案（可直接落地）

### 5.1 JSON Schema（节选，完整见 `/schema/summary.schema.json`）
见附带文件。

### 5.2 System Prompt（工程师向技术简报）
见 `/prompts/system_v2.txt`。

### 5.3 Task Prompt（含质量检查）
见 `/prompts/task_v2.txt`。

### 5.4 可选：簇间排序理由（内部使用）
见 `/prompts/sorting_reason_optional.txt`。

---

## 6. 数据与存储设计
建议表/集合：`runs`、`items`、`fingerprint`、`embeddings`、`clusters`、`rerank`、`summaries`（细节与字段示例见文末“附录 A”）。

---

## 7. 评测与验收

**自动化指标**：  
- 去重：P/R/F1；重复族信息保留率；
- 聚类：Purity/NMI/ARI；噪声点覆盖；
- 精排：NDCG@k、MMR 去冗率；
- 摘要：引用覆盖率、可解析率、重复 bullet 率、平均 bullet 长度；
- 性能成本：端到端延迟、Token/条目、缓存命中、失败重试率。

**人工评测**：事实性/可执行性/可读性抽样。  
**验收**：离线回放 A/B，对比基线达阈值后灰度放量。

---

## 8. 可观测性与成本治理
- 结构化日志；Prometheus 指标；OpenTelemetry 追踪（带 run_id/topic_id）；
- 成本与质量仪表板；
- 告警：解析失败率↑、引用覆盖骤降、成本异常。

---

## 9. 灰度与回滚
- 提示词/模型版本化：`prompt_version/schema_version/model_alias`；
- 灰度：按来源/用户/比例；影子运行；
- 回滚：稳定版本快回；最近 N 次 run 快照。

---

## 10. 风险与对策
- 过度去重：阈值分层+抽样校准；保留差异字段对齐；  
- 聚类不稳：固定随机种子；主题命名辅助；噪声二次吸附；  
- 幻觉：只用输入事实；每 bullet 带来源；质量检查失败→保守输出；  
- 成本/延迟波动：批量/缓存/退避；自适应阶段；上下文配额；  
- 外部 API 抖动：重试+熔断+降级（本地模型/缓存）。

---

## 11. 实施顺序（依赖关系）
1) 基础设施与抽象层；2) 两级去重；3) 聚类增强与噪声策略；  
4) 精排 + MMR；5) 上下文打包器；6) 结构化摘要与质量清单；  
7) 评测与仪表板；8) 优化与清理。

---

## 12. 关键配置与伪代码（节选）

### 12.1 配置（YAML，简化版）
详见 `/config/pipeline.example.yaml`。

### 12.2 两级去重（伪代码）
详见 `/docs/snippets_dedup.md`。

### 12.3 精排 + MMR（伪代码）
详见 `/docs/snippets_rerank_mmr.md`。

---

## 13. 交付物清单
- 代码骨架：`providers/`、`pipeline/`、`storage/`、`observability/`；
- 配置与脚本：YAML、CLI、Docker/Compose（可选）；
- 提示词：System/Task/Schema v2（版本化）；
- 评测：离线回放器、指标报表脚本；
- 文档：运行手册、参数调优指南、故障排查。

---

## 14. 下一步建议
- 用一份固定数据快照做离线回放，拿首轮曲线；
- 以“提示词 v2 + 两级去重 + 簇内精排”为最小增量上线点灰度；
- 逐步启用上下文打包器与簇间打分；
- 建立每周评测与参数固化机制。

---

### 附录 A：数据表字段建议（节选）

- `runs(run_id, started_at, config_hash, provider_versions, …)`  
- `items(run_id, item_id, source, title, canonical_text, lang, ts, meta)`  
- `fingerprint(run_id, item_id, sha1, simhash, family_id)`  
- `embeddings(run_id, item_id, model, vector_ref)`  
- `clusters(run_id, topic_id, label_llm, center_item_id, members[])`  
- `rerank(run_id, topic_id, scores)`  
- `summaries(run_id, json_blob, prompt_version, schema_version)`

---

**版权与使用**：本文档与附带文件仅供团队内部使用，可按需修改。
