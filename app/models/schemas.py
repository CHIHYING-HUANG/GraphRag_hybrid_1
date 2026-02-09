"""
API 資料模型定義

使用 Pydantic 定義所有 API 端點的請求和回應格式。
提供自動驗證、序列化和 API 文件生成。
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any


# =============================================================================
# Chat 相關 Schema
# =============================================================================

class Message(BaseModel):
    """
    聊天訊息模型
    
    欄位：
        role: 訊息發送者的角色（user 或 assistant）
        content: 訊息內容
    """
    role: str = Field(
        ...,
        description="訊息發送者的角色", 
        examples=["user", "assistant"]
    )
    content: str = Field(
        ..., 
        description="訊息內容", 
        examples=["台灣於何年開始實施九年國民義務教育?"]
    )


class ChatRequest(BaseModel):
    """
    問答請求模型
    
    欄位：
        messages: 對話訊息列表，每則訊息包含 role 和 content
    """
    messages: List[Message] = Field(
        ...,
        description="對話訊息列表"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": "台灣於何年開始實施九年國民義務教育?"}
                    ]
                }
            ]
        }
    )


class ChatResponse(BaseModel):
    """
    問答回應模型
    
    欄位：
        answer: LLM 生成的答案
        context: 使用的上下文（向量檢索 + 圖譜檢索）
    """
    answer: str
    context: List[str]


# =============================================================================
# Corpus 資料匯入 Schema
# =============================================================================

class CorpusIngestRequest(BaseModel):
    """
    Corpus 匯入請求模型
    
    欄位：
        limit: 處理的文檔數量限制
    """
    limit: Optional[int] = Field(
        None,
        description="處理的文檔數量限制（None = 全部）", 
        examples=[10, 50, 100]
    )


class CorpusIngestResponse(BaseModel):
    """
    Corpus 資料匯入回應模型
    
    欄位：
        message: 處理結果訊息
        docs_processed: 已處理的文檔數量
        total_docs: 總文檔數量
    """
    message: str
    docs_processed: int
    total_docs: int


# =============================================================================
# Corpus 評估 Schema
# =============================================================================


class SourceMetrics(BaseModel):
    """
    評估指標模型
    
    欄位：
        total_questions: 問題數量
        hit_rate: Hit Rate (單一) - 至少找到 1 篇黃金文檔的比例
        partial_hit_rate: Partial Hit Rate - 找到的黃金文檔比例
        mrr: Mean Reciprocal Rank - 黃金文檔排名倒數的平均
        map: Mean Average Precision - 平均精度均值
        generation_pass_rate: LLM-as-a-Judge 通過率
        avg_retrieval_time: 平均檢索時間 (秒)
        avg_generation_time: 平均生成時間 (秒)
        avg_total_time: 平均總時間 (秒)
    """
    total_questions: int
    hit_rate: float
    partial_hit_rate: float
    mrr: float
    map: float
    generation_pass_rate: float
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_total_time: float = 0.0


class CorpusEvaluateResponse(BaseModel):
    """
    Corpus 評估回應模型
    
    欄位：
        overall: 整體評估指標
        by_source: 按資料來源分組的指標（drcd, hotpotqa, 2wiki）
        by_question_type: 按問題類型分組的指標
    """
    overall: SourceMetrics
    by_source: Dict[str, SourceMetrics]
    by_question_type: Dict[str, SourceMetrics] = {}


class QuestionDetail(BaseModel):
    """
    單一問題的詳細評估結果
    
    欄位：
        question_id: 問題 ID
        question: 問題內容
        gold_answer: 標準答案
        model_answer: 模型生成的答案
        gold_doc_ids: 黃金文檔 ID 列表
        retrieved_doc_ids: 檢索到的文檔 ID 列表
        hit: 是否命中（至少找到 1 篇黃金文檔）
        found_count: 找到的黃金文檔數量
        mrr: 該問題的 MRR 分數
        ap: 該問題的 Average Precision 分數
        judge_pass: LLM-as-a-Judge 是否通過
        source_dataset: 資料來源
        retrieval_time: 檢索時間 (秒)
        generation_time: 生成時間 (秒)
        total_time: 總時間 (秒)
    """
    question_id: str
    question: str
    gold_answer: str
    model_answer: str
    gold_doc_ids: List[str]
    retrieved_doc_ids: List[str]
    hit: bool
    found_count: int
    mrr: float
    ap: float
    judge_pass: bool
    source_dataset: str
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0


class DetailedEvaluateResponse(BaseModel):
    """
    詳細評估回應模型（包含每個問題的詳細信息）
    
    欄位：
        overall: 整體評估指標
        by_source: 按資料來源分組的指標
        by_question_type: 按問題類型分組的指標 (single-hop, multi-hop)
        questions: 每個問題的詳細評估結果
    """
    overall: SourceMetrics
    by_source: Dict[str, SourceMetrics]
    by_question_type: Dict[str, SourceMetrics] = {}
    questions: List[QuestionDetail]
