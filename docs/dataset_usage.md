# 使用指南：繁體中文 RAG 評測資料集

本文件說明如何載入、使用本資料集 (`queries.json` 與 `corpus.json`) 來進行 RAG 系統的評測。

## 資料集概覽

| 項目 | 數量 | 說明 |
|------|------|------|
| **問題總數** | 60 題 | DRCD 20 + HotpotQA 20 + 2Wiki 20 |
| **單跳題 (Single-hop)** | 20 題 | 僅需單一文檔即可回答 |
| **多跳題 (Multi-hop)** | 40 題 | 需整合多篇文檔資訊 |
| **文檔總數** | 600 篇 | 含 Gold Contexts + Hard/Random Negatives |

## 1. 資料載入

資料集為標準 JSON 格式，可使用 Python `json` 套件直接載入。

```python
import json

# 1. 載入評測題庫 (60 題)
with open("data/processed/queries.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

# 2. 載入文檔庫 (600 篇)
with open("data/processed/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# 建立檢索索引 (Doc ID -> Content)
corpus_map = {doc["doc_id"]: doc["content"] for doc in corpus}
```

## 2. 評測流程範例

一般的 RAG 評測流程如下：

1. **建立索引 (Indexing)**：將 `corpus` 中的 600 篇文章轉換為向量並存入向量資料庫 (Vector DB)。
   - **Embedding Model**: `text-embedding-3-small`
   - **Chunking Strategy**: 不進行切分 (No Chunking)，直接使用完整文章內容 (content)。
2. **檢索 (Retrieval)**：針對每個 `query`，檢索出 Top-5 篇相關文章。
   - 若涉及 LLM (如重排序、查詢擴展)，統一使用 `gpt-4o-mini`。
3. **生成 (Generation)**：將檢索到的文章作為 Context，輸入 LLM 產生這題的答案。
   - **Generation Model**: `gpt-4o-mini`
4. **評分 (Scoring)**：計算檢索命中率與答案準確度。

## 3. 計算評測指標

### 3.1 檢索指標

針對檢索結果，計算以下三項指標，並按資料來源分組統計：

| 指標 | 說明 |
|------|------|
| **Hit Rate (單一)** | 是否至少找到 1 篇黃金文檔 (binary) |
| **Partial Hit Rate** | 找到的黃金文檔比例 (例如 17/20) |
| **MRR (Mean Reciprocal Rank)** | 所有黃金文檔排名倒數的平均 |

> **MRR 計算範例**：假設 gold docs = {A, B, C}，檢索結果 = [X, A, Y, B, C]
> - A 的 RR = 1/2 = 0.5
> - B 的 RR = 1/4 = 0.25
> - C 的 RR = 1/5 = 0.2
> - 平均 RR = (0.5 + 0.25 + 0.2) / 3 = 0.317

```python
from collections import defaultdict

k = 5

# 按資料來源分組統計
stats = defaultdict(lambda: {
    "total": 0,
    "hit_count": 0,        # Hit Rate (單一)
    "found_sum": 0,        # Partial Hit Rate 分子
    "gold_sum": 0,         # Partial Hit Rate 分母
    "rr_sum": 0.0,         # MRR 累計
})

for q in queries:
    source = q["source_dataset"]
    gold_ids = set(q["gold_doc_ids"])
    
    # 您的系統檢索結果 (回傳 doc_ids)
    retrieved_ids = your_rag_system.retrieve(q["question"], top_k=k)
    
    # 計算找到幾篇 Gold Docs
    found_count = sum(1 for doc_id in retrieved_ids if doc_id in gold_ids)
    
    # Hit Rate (單一): 至少找到 1 篇即算命中
    hit = 1 if found_count > 0 else 0
    
    # 平均 RR: 所有黃金文檔排名倒數的平均
    rr_list = []
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            rr_list.append(1.0 / rank)
    avg_rr = sum(rr_list) / len(gold_ids) if len(gold_ids) > 0 else 0.0
    
    # 累計統計
    stats[source]["total"] += 1
    stats[source]["hit_count"] += hit
    stats[source]["found_sum"] += found_count
    stats[source]["gold_sum"] += len(gold_ids)
    stats[source]["rr_sum"] += avg_rr

# 輸出結果
print("📊 按資料來源分組")
print()

for source in ["drcd", "hotpotqa", "2wiki"]:
    s = stats[source]
    if s["total"] == 0:
        continue
    
    hit_rate = s["hit_count"] / s["total"]
    partial_hit_rate = s["found_sum"] / s["gold_sum"] if s["gold_sum"] > 0 else 0
    mrr = s["rr_sum"] / s["total"]
    
    print(f"【{source}】")
    print(f"問題數:           {s['total']}")
    print(f"Hit Rate (單一):  {hit_rate:.2%} ({s['total']} 題)")
    print(f"Partial Hit Rate: {partial_hit_rate:.2%} ({s['found_sum']}/{s['gold_sum']})")
    print(f"MRR:              {mrr:.4f}")
    print()
```

### 3.2 生成指標：LLM-as-a-Judge

由於翻譯後的答案可能有用詞差異，不使用字串完全比對 (Exact Match)。使用 LLM ( GPT-4o-mini) 來判斷語意正確性。

**Prompt 範例：**

> 請判斷「模型回答」是否與「標準答案」語意一致。
>
> 問題：{question}
> 標準答案：{gold_answer}
> 模型回答：{model_answer}
>
> 如果語意一致請回答 "Pass"，否則回答 "Fail"。

計算通過率。

## 4. 資料欄位說明

### Query (`queries.json`)

| 欄位              | 說明                                                 |
| ----------------- | ---------------------------------------------------- |
| `question_id`   | 唯一識別碼                                           |
| `question`      | 繁體中文問題                                         |
| `gold_answer`   | 標準答案 (供評測比對)                                |
| `gold_doc_ids`  | 正解文檔 ID 列表 (供檢索評測)                        |
| `question_type` | `single-hop` 或 `multi-hop` (可依此分類分析效能) |

### Corpus (`corpus.json`)

| 欄位                | 說明                                         |
| ------------------- | -------------------------------------------- |
| `doc_id`          | 唯一識別碼 (與 queries 對應)                 |
| `content`         | 文章內文                                     |
| `original_source` | 原始資料來源 (如 `squad`, `hotpotqa`...) |
| `is_gold`         | 標示該文檔是否為某個問題的正解 (True/False)  |
