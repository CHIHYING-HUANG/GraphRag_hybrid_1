"""
評估指標計算模組

提供檢索和生成指標的計算功能
"""

import logging
from typing import List, Set, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)




def calculate_reciprocal_rank(gold_ids: Set[str], retrieved_ids: List[str]) -> float:
    """計算 Reciprocal Rank（標準 MRR 定義）
    
    取第一個命中的 gold doc 的 RR。
    """
    if not gold_ids:
        return 0.0
    
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    
    return 0.0


def calculate_average_precision(gold_ids: Set[str], retrieved_ids: List[str]) -> float:
    """計算 Average Precision (AP)
    
    AP = (1 / |Gold|) * sum(P@k * rel(k))
    """
    if not gold_ids:
        return 0.0
    
    relevant_count = 0
    precision_sum = 0.0
    
    for k, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            relevant_count += 1
            precision_at_k = relevant_count / k
            precision_sum += precision_at_k
            
    return precision_sum / len(gold_ids)


def calculate_retrieval_metrics(
    retrieved_ids: List[str],
    gold_ids: Set[str]
) -> Dict[str, Any]:
    """
    計算檢索指標
    
    參數:
        retrieved_ids: 檢索到的文檔 ID 列表（有序）
        gold_ids: 黃金文檔 ID 集合
    
    返回:
        包含以下指標的字典：
        - hit: 是否至少找到 1 篇黃金文檔（1 或 0）
        - found_count: 找到的黃金文檔數量
        - avg_rr: 平均 Reciprocal Rank（所有黃金文檔排名倒數的平均）
        - ap: Average Precision
    """
    # 計算找到的黃金文檔數量
    found_count = sum(1 for doc_id in retrieved_ids if doc_id in gold_ids)
    
    # Hit Rate (單一)：至少找到 1 篇即算命中
    hit = 1 if found_count > 0 else 0
    
    # 計算 MRR (標準定義)：取第一個命中的 gold doc 的 RR
    avg_rr = calculate_reciprocal_rank(gold_ids, retrieved_ids)

    # 計算 Average Precision
    ap = calculate_average_precision(gold_ids, retrieved_ids)
    
    return {
        "hit": hit,
        "found_count": found_count,
        "avg_rr": avg_rr,
        "ap": ap
    }


def compute_final_metrics(
    stats_by_source: Dict[str, Dict],
    stats_by_type: Dict[str, Dict],
    total_queries: int
) -> Dict[str, Any]:
    """
    計算最終彙總指標
    
    參數:
        stats_by_source: 按資料來源分組的統計資料
        stats_by_type: 按問題類型分組的統計資料
        total_queries: 總問題數量
    
    返回:
        包含整體、分組源、分問題類型指標的字典
    """
    # 計算整體指標 (合併所有來源)
    overall_stats = {
        "total": 0,
        "hit_count": 0,
        "found_sum": 0,
        "gold_sum": 0,
        "rr_sum": 0.0,
        "ap_sum": 0.0,
        "generation_pass": 0,
        "retrieval_time_sum": 0.0,
        "generation_time_sum": 0.0,
        "total_time_sum": 0.0
    }
    
    for source_stats in stats_by_source.values():
        overall_stats["total"] += source_stats["total"]
        overall_stats["hit_count"] += source_stats["hit_count"]
        overall_stats["found_sum"] += source_stats["found_sum"]
        overall_stats["gold_sum"] += source_stats["gold_sum"]
        overall_stats["rr_sum"] += source_stats["rr_sum"]
        overall_stats["ap_sum"] += source_stats.get("ap_sum", 0.0)
        overall_stats["generation_pass"] += source_stats["generation_pass"]
        overall_stats["retrieval_time_sum"] += source_stats.get("retrieval_time_sum", 0.0)
        overall_stats["generation_time_sum"] += source_stats.get("generation_time_sum", 0.0)
        overall_stats["total_time_sum"] += source_stats.get("total_time_sum", 0.0)
    
    # 計算整體指標
    overall_metrics = _compute_metrics(overall_stats)
    
    # 計算分組指標 (按來源)
    by_source = {}
    for source in ["drcd", "hotpotqa", "2wiki"]:
        if source in stats_by_source and stats_by_source[source]["total"] > 0:
            by_source[source] = _compute_metrics(stats_by_source[source])
    
    # 計算分組指標 (按問題類型)
    by_question_type = {}
    for qtype in ["single-hop", "multi-hop"]:
        if qtype in stats_by_type and stats_by_type[qtype]["total"] > 0:
            by_question_type[qtype] = _compute_metrics(stats_by_type[qtype])
    
    return {
        "overall": overall_metrics,
        "by_source": by_source,
        "by_question_type": by_question_type
    }


def _compute_metrics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    計算單個組的指標
    
    參數:
        stats: 統計資料
    
    返回:
        指標字典
    """
    total = stats["total"]
    
    if total == 0:
        return {
            "total_questions": 0,
            "hit_rate": 0.0,
            "partial_hit_rate": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "generation_pass_rate": 0.0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_total_time": 0.0
        }
    
    hit_rate = stats["hit_count"] / total
    partial_hit_rate = stats["found_sum"] / stats["gold_sum"] if stats["gold_sum"] > 0 else 0
    mrr = stats["rr_sum"] / total
    map_score = stats.get("ap_sum", 0.0) / total
    generation_pass_rate = stats["generation_pass"] / total
    
    # 計算平均延遲
    avg_retrieval_time = stats.get("retrieval_time_sum", 0.0) / total
    avg_generation_time = stats.get("generation_time_sum", 0.0) / total
    avg_total_time = stats.get("total_time_sum", 0.0) / total
    
    return {
        "total_questions": total,
        "hit_rate": hit_rate,
        "partial_hit_rate": partial_hit_rate,
        "mrr": mrr,
        "map": map_score,
        "generation_pass_rate": generation_pass_rate,
        "avg_retrieval_time": avg_retrieval_time,
        "avg_generation_time": avg_generation_time,
        "avg_total_time": avg_total_time
    }
