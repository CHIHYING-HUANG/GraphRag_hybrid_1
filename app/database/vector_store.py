"""
向量資料庫管理器

負責所有 ChromaDB 向量資料庫的操作
"""

import logging
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    ChromaDB 向量儲存管理器
    
    職責：
    - 初始化和管理 ChromaDB 連線
    - 提供文檔新增、檢索介面
    - 提供統計資訊查詢
    """
    
    def __init__(
        self,
        collection_name: str = "corpus_docs",
        persist_directory: str = "./chroma_db_corpus",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        初始化向量資料庫管理器
        
        參數:
            collection_name: Collection 名稱
            persist_directory: 持久化目錄
            embedding_model: Embedding 模型名稱
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # 初始化 Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # 初始化 ChromaDB
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        logger.info(f"VectorStoreManager 初始化完成: {collection_name}")
    
    def add_documents(self, documents: List[Document], ids: List[str] = None):
        """
        批次新增文檔到向量資料庫
        
        參數:
            documents: LangChain Document 列表
            ids: 文檔 ID 列表（可選）
        """
        try:
            if ids:
                self.vector_store.add_documents(documents, ids=ids)
            else:
                self.vector_store.add_documents(documents)
            logger.debug(f"成功新增 {len(documents)} 篇文檔")
        except Exception as e:
            logger.error(f"新增文檔失敗: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        向量檢索
        
        參數:
            query: 查詢文字
            k: 返回的文檔數量
        
        返回:
            Document 列表（包含 metadata）
        """
        try:
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": k}
            )
            docs = retriever.invoke(query)
            return docs
        except Exception as e:
            logger.error(f"向量檢索失敗: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        向量檢索（含相似度分數）
        
        參數:
            query: 查詢文字
            k: 返回的文檔數量
        
        返回:
            (Document, score) 元組列表
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"向量檢索失敗: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        取得向量資料庫統計資訊
        
        返回:
            包含文檔數量、collection 名稱等統計資訊
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"取得統計資訊失敗: {e}")
            return {"error": str(e)}
    
    def clear(self):
        """
        清空向量資料庫（危險操作）
        """
        try:
            # ChromaDB 沒有直接的 clear 方法，需要刪除並重建
            logger.warning(f"準備清空 collection: {self.collection_name}")
            # 實作清空邏輯（如果需要）
        except Exception as e:
            logger.error(f"清空資料庫失敗: {e}")
            raise
