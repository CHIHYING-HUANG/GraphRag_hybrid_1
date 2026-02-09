"""
向量檢索器

負責從向量資料庫檢索相關文檔
"""

import logging
from typing import List, Tuple
from app.database.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    向量檢索器
    
    職責：
    - 執行向量相似度檢索
    - 提供不同格式的檢索結果
    """
    
    def __init__(self, vector_store: VectorStoreManager):
        """
        初始化向量檢索器
        
        參數:
            vector_store: 向量儲存管理器
        """
        self.vector_store = vector_store
        logger.info("VectorRetriever 初始化完成")
    
    def retrieve(self, question: str, k: int = 5) -> List[str]:
        """
        執行向量檢索，返回文檔內容
        
        參數:
            question: 查詢問題
            k: 返回的文檔數量
        
        返回:
            文檔內容列表
        """
        try:
            docs = self.vector_store.search(question, k=k)
            contents = [doc.page_content for doc in docs]
            logger.debug(f"向量檢索到 {len(contents)} 篇文檔")
            return contents
        except Exception as e:
            logger.error(f"向量檢索失敗: {e}")
            return []
    
    def retrieve_with_ids(
        self,
        question: str,
        k: int = 5
    ) -> Tuple[List[str], List[str]]:
        """
        執行向量檢索，返回文檔內容和 ID
        
        參數:
            question: 查詢問題
            k: 返回的文檔數量
        
        返回:
            (contents, doc_ids) 元組
        """
        try:
            docs = self.vector_store.search(question, k=k)
            contents = [doc.page_content for doc in docs]
            doc_ids = [doc.metadata.get("doc_id", "") for doc in docs]
            logger.debug(f"向量檢索到 {len(contents)} 篇文檔（含 ID）")
            return contents, doc_ids
        except Exception as e:
            logger.error(f"向量檢索失敗: {e}")
            return [], []
    
    def retrieve_with_metadata(
        self,
        question: str,
        k: int = 5
    ) -> List[dict]:
        """
        執行向量檢索，返回文檔內容和完整 metadata
        
        參數:
            question: 查詢問題
            k: 返回的文檔數量
        
        返回:
            包含 content 和 metadata 的字典列表
        """
        try:
            docs = self.vector_store.search(question, k=k)
            results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
            logger.debug(f"向量檢索到 {len(results)} 篇文檔（含 metadata）")
            return results
        except Exception as e:
            logger.error(f"向量檢索失敗: {e}")
            return []
