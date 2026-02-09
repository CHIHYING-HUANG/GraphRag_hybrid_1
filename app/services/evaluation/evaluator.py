"""
Corpus 評估器

負責執行完整的 RAG 系統評估流程
"""

import json
import logging
from typing import Dict, Any, List
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.services.retrieval.vector_retriever import VectorRetriever
from app.services.rag.graph_rag import run_graph_rag
from app.services.evaluation.metrics import calculate_retrieval_metrics, compute_final_metrics
from app.core.config import settings
from app.models import schemas

logger = logging.getLogger(__name__)


class CorpusEvaluator:
    """
    Corpus RAG 系統評估器
    
    職責：
    - 執行完整的評估流程
    - 使用 RAG 工作流程生成答案
    - 計算檢索和生成指標
    - 使用 LLM-as-a-Judge 評估答案品質
    """
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0
    ):
        """
        初始化評估器
        
        參數:
            vector_retriever: 向量檢索器（用於評估檢索）
            llm_model: LLM 模型名稱
            temperature: LLM 溫度參數
        """
        self.vector_retriever = vector_retriever
        
        # 初始化 LLM（用於 Judge）
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        logger.info("CorpusEvaluator 初始化完成")
    
    async def evaluate(
        self,
        queries_path: str = "data/queries.json",
        k: int = 5,
        limit: int = None
    ) -> Dict[str, Any]:
        """
        執行評估
        
        參數:
            queries_path: queries.json 的檔案路徑
            k: Top-K 檢索數量
            limit: 要評估的問題數量（None = 全部）
        
        返回:
            評估結果字典
        """
        logger.info(f"開始評估，queries_path={queries_path}, k={k}, limit={limit}")
        
        # 1. 載入問題
        queries = self._load_queries(queries_path)
        queries_to_eval = queries[:limit] if limit else queries
        total_queries = len(queries_to_eval)
        
        logger.info(f"準備評估 {total_queries} 個問題")
        
        # 2. 初始化統計 (按來源 + 按問題類型)
        stats_by_source = defaultdict(lambda: {
            "total": 0,
            "hit_count": 0,
            "found_sum": 0,
            "gold_sum": 0,
            "rr_sum": 0.0,
            "ap_sum": 0.0,
            "generation_pass": 0,
            "retrieval_time_sum": 0.0,
            "generation_time_sum": 0.0,
            "total_time_sum": 0.0,
        })
        
        stats_by_type = defaultdict(lambda: {
            "total": 0,
            "hit_count": 0,
            "found_sum": 0,
            "gold_sum": 0,
            "rr_sum": 0.0,
            "ap_sum": 0.0,
            "generation_pass": 0,
            "retrieval_time_sum": 0.0,
            "generation_time_sum": 0.0,
            "total_time_sum": 0.0,
        })
        
        # 3. 評估每個問題
        for idx, query in enumerate(queries_to_eval, 1):
            question = query.get("question")
            gold_answer = query.get("gold_answer")
            gold_doc_ids = set(query.get("gold_doc_ids", []))
            source_dataset = query.get("source_dataset", "unknown")
            question_type = query.get("question_type", "unknown")  # ✅ 新增
            
            logger.info(f"[{idx}/{total_queries}] 評估問題：{question[:50]}...")
            
            try:
                # 生成答案（使用 RAG 工作流程）
                rag_result = await run_graph_rag(question)
                model_answer = rag_result["answer"]
                retrieved_doc_ids = rag_result.get("retrieved_doc_ids", [])
                
                # 如果 RAG 沒有返回 doc_ids（例如出錯），則降級使用 vector_retriever
                if not retrieved_doc_ids:
                     _, retrieved_doc_ids = self.vector_retriever.retrieve_with_ids(question, k=k)

                # 檢索評估
                retrieval_metrics = calculate_retrieval_metrics(retrieved_doc_ids, gold_doc_ids)
                
                # LLM-as-a-Judge
                is_pass = await self._judge_answer(question, gold_answer, model_answer)
                
                # 累計統計 (按來源)
                stats_by_source[source_dataset]["total"] += 1
                stats_by_source[source_dataset]["hit_count"] += retrieval_metrics["hit"]
                stats_by_source[source_dataset]["found_sum"] += retrieval_metrics["found_count"]
                stats_by_source[source_dataset]["gold_sum"] += len(gold_doc_ids)
                stats_by_source[source_dataset]["rr_sum"] += retrieval_metrics["avg_rr"]
                stats_by_source[source_dataset]["ap_sum"] += retrieval_metrics["ap"]
                stats_by_source[source_dataset]["generation_pass"] += (1 if is_pass else 0)
                stats_by_source[source_dataset]["retrieval_time_sum"] += rag_result.get("retrieval_time", 0.0)
                stats_by_source[source_dataset]["generation_time_sum"] += rag_result.get("generation_time", 0.0)
                stats_by_source[source_dataset]["total_time_sum"] += rag_result.get("total_time", 0.0)
                
                # 累計統計 (按問題類型) ✅ 新增
                stats_by_type[question_type]["total"] += 1
                stats_by_type[question_type]["hit_count"] += retrieval_metrics["hit"]
                stats_by_type[question_type]["found_sum"] += retrieval_metrics["found_count"]
                stats_by_type[question_type]["gold_sum"] += len(gold_doc_ids)
                stats_by_type[question_type]["rr_sum"] += retrieval_metrics["avg_rr"]
                stats_by_type[question_type]["ap_sum"] += retrieval_metrics["ap"]
                stats_by_type[question_type]["generation_pass"] += (1 if is_pass else 0)
                stats_by_type[question_type]["retrieval_time_sum"] += rag_result.get("retrieval_time", 0.0)
                stats_by_type[question_type]["generation_time_sum"] += rag_result.get("generation_time", 0.0)
                stats_by_type[question_type]["total_time_sum"] += rag_result.get("total_time", 0.0)
                
                logger.debug(
                    f"  - Hit: {retrieval_metrics['hit']}, "
                    f"Found: {retrieval_metrics['found_count']}/{len(gold_doc_ids)}, "
                    f"MRR: {retrieval_metrics['avg_rr']:.4f}, "
                    f"AP: {retrieval_metrics['ap']:.4f}, "
                    f"Judge: {'Pass' if is_pass else 'Fail'}"
                )
                
            except Exception as e:
                logger.error(f"評估問題時發生錯誤：{e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 4. 計算最終指標
        results = compute_final_metrics(
            stats_by_source=stats_by_source,
            stats_by_type=stats_by_type,
            total_queries=total_queries
        )
        
        logger.info("評估完成！")
        return results

    async def evaluate_detailed(
        self,
        queries_path: str = "data/queries.json",
        k: int = 5,
        limit: int = None
    ) -> schemas.DetailedEvaluateResponse:
        """
        執行詳細評估
        
        參數:
            queries_path: queries.json 的檔案路徑
            k: Top-K 檢索數量
            limit: 要評估的問題數量（None = 全部）
        
        返回:
            詳細評估結果 (DetailedEvaluateResponse)
        """
        logger.info(f"開始詳細評估，queries_path={queries_path}, k={k}, limit={limit}")
        
        # 1. 載入問題
        queries = self._load_queries(queries_path)
        queries_to_eval = queries[:limit] if limit else queries
        total_queries = len(queries_to_eval)
        
        logger.info(f"準備評估 {total_queries} 個問題")
        
        # 2. 初始化統計
        stats_by_source = defaultdict(lambda: {
            "total": 0, "hit_count": 0, "found_sum": 0, "gold_sum": 0,
            "rr_sum": 0.0, "ap_sum": 0.0, "generation_pass": 0,
            "retrieval_time_sum": 0.0, "generation_time_sum": 0.0, "total_time_sum": 0.0,
        })
        
        stats_by_type = defaultdict(lambda: {
            "total": 0, "hit_count": 0, "found_sum": 0, "gold_sum": 0,
            "rr_sum": 0.0, "ap_sum": 0.0, "generation_pass": 0,
            "retrieval_time_sum": 0.0, "generation_time_sum": 0.0, "total_time_sum": 0.0,
        })
        
        question_details = []
        
        # 3. 評估每個問題
        for idx, query in enumerate(queries_to_eval, 1):
            question = query.get("question")
            gold_answer = query.get("gold_answer")
            gold_doc_ids = query.get("gold_doc_ids", [])  # list for details
            gold_doc_ids_set = set(gold_doc_ids)          # set for metrics
            source_dataset = query.get("source_dataset", "unknown")
            question_type = query.get("question_type", "unknown")
            
            logger.info(f"[{idx}/{total_queries}] 評估問題：{question[:50]}...")
            
            try:
                # 執行 RAG
                rag_result = await run_graph_rag(question)
                model_answer = rag_result.get("answer", "")
                retrieved_doc_ids = rag_result.get("retrieved_doc_ids", [])
                
                # Fallback check
                if not retrieved_doc_ids:
                     _, retrieved_doc_ids = self.vector_retriever.retrieve_with_ids(question, k=k)
                
                # 檢索評估
                retrieval_metrics = calculate_retrieval_metrics(retrieved_doc_ids, gold_doc_ids_set)
                
                # LLM Judge
                is_pass = await self._judge_answer(question, gold_answer, model_answer)
                
                # 累計統計 (按來源)
                stats_by_source[source_dataset]["total"] += 1
                stats_by_source[source_dataset]["hit_count"] += retrieval_metrics["hit"]
                stats_by_source[source_dataset]["found_sum"] += retrieval_metrics["found_count"]
                stats_by_source[source_dataset]["gold_sum"] += len(gold_doc_ids_set)
                stats_by_source[source_dataset]["rr_sum"] += retrieval_metrics["avg_rr"]
                stats_by_source[source_dataset]["ap_sum"] += retrieval_metrics["ap"]
                stats_by_source[source_dataset]["generation_pass"] += (1 if is_pass else 0)
                stats_by_source[source_dataset]["retrieval_time_sum"] += rag_result.get("retrieval_time", 0.0)
                stats_by_source[source_dataset]["generation_time_sum"] += rag_result.get("generation_time", 0.0)
                stats_by_source[source_dataset]["total_time_sum"] += rag_result.get("total_time", 0.0)
                
                # 累計統計 (按問題類型)
                stats_by_type[question_type]["total"] += 1
                stats_by_type[question_type]["hit_count"] += retrieval_metrics["hit"]
                stats_by_type[question_type]["found_sum"] += retrieval_metrics["found_count"]
                stats_by_type[question_type]["gold_sum"] += len(gold_doc_ids_set)
                stats_by_type[question_type]["rr_sum"] += retrieval_metrics["avg_rr"]
                stats_by_type[question_type]["ap_sum"] += retrieval_metrics["ap"]
                stats_by_type[question_type]["generation_pass"] += (1 if is_pass else 0)
                stats_by_type[question_type]["retrieval_time_sum"] += rag_result.get("retrieval_time", 0.0)
                stats_by_type[question_type]["generation_time_sum"] += rag_result.get("generation_time", 0.0)
                stats_by_type[question_type]["total_time_sum"] += rag_result.get("total_time", 0.0)
                
                # 收集詳細信息
                question_details.append({
                    "question_id": query.get("question_id"),
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer,
                    "gold_doc_ids": gold_doc_ids,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "hit": bool(retrieval_metrics["hit"]),
                    "found_count": retrieval_metrics["found_count"],
                    "mrr": retrieval_metrics["avg_rr"],
                    "ap": retrieval_metrics["ap"],
                    "judge_pass": is_pass,
                    "source_dataset": source_dataset,
                    "retrieval_time": rag_result.get("retrieval_time", 0.0),
                    "generation_time": rag_result.get("generation_time", 0.0),
                    "total_time": rag_result.get("total_time", 0.0)
                })
                
            except Exception as e:
                logger.error(f"評估問題時發生錯誤：{e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 4. 計算最終指標
        final_results = compute_final_metrics(
            stats_by_source=stats_by_source,
            stats_by_type=stats_by_type,
            total_queries=total_queries
        )
        
        logger.info("詳細評估完成！")
        
        return schemas.DetailedEvaluateResponse(
            overall=schemas.SourceMetrics(**final_results["overall"]),
            by_source={k: schemas.SourceMetrics(**v) for k, v in final_results["by_source"].items()},
            by_question_type={k: schemas.SourceMetrics(**v) for k, v in final_results.get("by_question_type", {}).items()},
            questions=[schemas.QuestionDetail(**q) for q in question_details]
        )
    
    def _load_queries(self, queries_path: str) -> List[Dict[str, Any]]:
        """
        載入 queries.json
        
        參數:
            queries_path: queries.json 的路徑
        
        返回:
            問題列表
        """
        try:
            with open(queries_path, "r", encoding="utf-8") as f:
                queries = json.load(f)
            logger.info(f"成功載入 queries.json，共 {len(queries)} 個問題")
            return queries
        except FileNotFoundError:
            logger.error(f"找不到檔案：{queries_path}")
            raise FileNotFoundError(f"queries.json 不存在於路徑：{queries_path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析錯誤：{e}")
            raise ValueError(f"queries.json 格式錯誤：{e}")
    
    async def _judge_answer(
        self,
        question: str,
        gold_answer: str,
        model_answer: str
    ) -> bool:
        """
        使用 LLM-as-a-Judge 評估答案
        
        參數:
            question: 問題
            gold_answer: 標準答案
            model_answer: 模型答案
        
        返回:
            是否通過（Pass/Fail）
        """
        try:
            # LLM-as-a-Judge (公正評分,使用通用原則)
            judge_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一個嚴謹但公正的評分者。請判斷「模型回答」是否與「標準答案」語意一致。

                    評分原則:
                    ✅ 核心事實一致即可判定為 Pass
                    ✅ 允許不同表達方式 (如「位於」vs「在」)
                    ✅ 允許翻譯差異 (如「紐約」vs「New York」)
                    ✅ 允許提供額外支持細節 (如附帶年份、完整名稱)
                    ✅ 允許數字格式差異 (如「92%」vs「百分之九十二」)

                    ❌ 只有核心事實明顯錯誤或完全無關時才判定為 Fail

                    如果語意一致請回答 \"Pass\"，否則回答 \"Fail\"。"""),
                ("human", "問題：{question}\n標準答案：{gold_answer}\n模型回答：{model_answer}\n\n判斷結果（Pass/Fail）：")
            ])
            
            chain = judge_prompt | self.llm
            
            # 取得 Langfuse callback（統一管理）
            from app.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()
            
            result = chain.invoke(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer
                },
                config={"callbacks": callbacks} if callbacks else {}
            )
            
            # 判斷是否包含 "Pass"
            is_pass = "pass" in result.content.lower()
            return is_pass
            
        except Exception as e:
            logger.error(f"LLM-as-a-Judge 錯誤：{e}")
            return False
