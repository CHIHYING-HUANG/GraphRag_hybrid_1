"""
統一服務介面層

提供向後兼容的服務介面，整合所有新的模組化元件
包含資料庫重複檢查以節省成本
"""

import logging
from typing import Dict, Any

from app.database.vector_store import VectorStoreManager
from app.database.graph_store import GraphStoreManager
from app.services.ingestion.data_loader import CorpusDataLoader
from app.services.ingestion.graph_builder import GraphBuilder
from app.services.retrieval.vector_retriever import VectorRetriever
from app.services.evaluation.evaluator import CorpusEvaluator

logger = logging.getLogger(__name__)


class CorpusIngestionService:
    """
    Corpus 資料匯入服務（統一介面）
    
    職責：
    - 提供向後兼容的介面
    - 整合資料載入、向量化、圖譜構建
    - 檢查資料庫是否已有資料，避免重複處理
    """
    
    def __init__(self):
        """初始化服務"""
        # 初始化資料庫管理器
        self.vector_store_manager = VectorStoreManager()
        self.graph_store_manager = GraphStoreManager()
        
        # 初始化資料處理元件
        self.data_loader = CorpusDataLoader()
        self.graph_builder = GraphBuilder(self.graph_store_manager)
        
        logger.info("CorpusIngestionService 初始化完成")
    

    
    async def ingest(
        self,
        corpus_path: str = "data/corpus.json",
        limit: int = None
    ) -> Dict[str, Any]:
        """
        執行資料匯入（向量化 + 圖譜構建）
        
        參數:
            corpus_path: corpus.json 路徑
            limit: 處理的文檔數量限制
        
        返回:
            包含處理結果的字典
        """
        try:
            # 1. 載入並準備文檔
            documents, ids = self.data_loader.load_and_prepare(corpus_path, limit)
            total_docs = len(documents)
            
            logger.info(f"開始處理 {total_docs} 篇文檔")
            
            # 3. 批次向量化
            logger.info("步驟 1/2: 向量化文檔...")
            self.vector_store_manager.add_documents(documents, ids=ids)
            
            # 4. 批次構建圖譜
            logger.info("步驟 2/2: 構建知識圖譜...")
            content_list = [(doc.page_content, doc_id) for doc, doc_id in zip(documents, ids)]
            await self.graph_builder.build_batch(content_list)
            
            logger.info(f"✅ 成功處理 {total_docs} 篇文檔")
            
            return {
                "message": f"成功處理 {total_docs} 篇文檔",
                "docs_processed": total_docs,
                "total_docs": total_docs,
                "skipped": False
            }
            
        except Exception as e:
            logger.error(f"資料匯入失敗: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        取得統計資訊
        
        返回:
            包含向量庫和圖譜統計的字典
        """
        vector_stats = self.vector_store_manager.get_stats()
        graph_stats = self.graph_store_manager.get_stats()
        
        return {
            "vector_store": vector_stats,
            "graph_store": graph_stats
        }


class CorpusEvaluationService:
    """
    Corpus 評估服務（統一介面）
    
    職責：
    - 提供向後兼容的介面
    - 整合評估流程
    """
    
    def __init__(self):
        """初始化服務"""
        # 初始化向量儲存和檢索器
        self.vector_store_manager = VectorStoreManager()
        self.vector_retriever = VectorRetriever(self.vector_store_manager)
        
        # 初始化評估器
        self.evaluator = CorpusEvaluator(self.vector_retriever)
        
        logger.info("CorpusEvaluationService 初始化完成")
    
    async def run_evaluation(
        self,
        queries_path: str = "data/queries.json",
        k: int = 5,
        limit: int = None
    ) -> Dict[str, Any]:
        """
        執行評估
        
        參數:
            queries_path: queries.json 路徑
            k: Top-K 檢索數量
            limit: 評估的問題數量限制
        
        返回:
            評估結果字典
        """
        return await self.evaluator.evaluate(
            queries_path=queries_path,
            k=k,
            limit=limit
        )

    async def evaluate_detailed(
        self,
        queries_path: str = "data/queries.json",
        k: int = 5,
        limit: int = None
    ) -> Dict[str, Any]:
        """
        執行詳細評估
        
        參數:
            queries_path: queries.json 路徑
            k: Top-K 檢索數量
            limit: 評估的問題數量限制
        
        返回:
            詳細評估結果 (DetailedEvaluateResponse)
        """
        return await self.evaluator.evaluate_detailed(
            queries_path=queries_path,
            k=k,
            limit=limit
        )


# 全域服務實例（向後兼容）
corpus_ingestion_service = CorpusIngestionService()
corpus_evaluation_service = CorpusEvaluationService()
