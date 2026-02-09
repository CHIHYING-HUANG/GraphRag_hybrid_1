"""
Corpus 資料載入器

負責載入和處理 corpus.json 資料
"""

import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CorpusDataLoader:
    """
    Corpus 資料載入器
    
    職責：
    - 載入 corpus.json 檔案
    - 轉換為 LangChain Document 格式
    - 提供批次處理功能
    """
    
    def load_corpus(
        self,
        corpus_path: str = "data/corpus.json",
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        載入 corpus.json
        
        參數:
            corpus_path: corpus.json 的路徑
            limit: 載入的文檔數量限制（None = 全部）
        
        返回:
            原始 corpus 資料列表
        """
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            
            logger.info(f"成功載入 corpus.json，共 {len(corpus)} 篇文檔")
            
            # 應用 limit
            if limit:
                corpus = corpus[:limit]
                logger.info(f"限制載入前 {limit} 篇文檔")
            
            return corpus
            
        except FileNotFoundError:
            logger.error(f"找不到檔案: {corpus_path}")
            raise FileNotFoundError(f"corpus.json 不存在於路徑：{corpus_path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析錯誤: {e}")
            raise ValueError(f"corpus.json 格式錯誤：{e}")
    
    def create_documents(
        self,
        corpus_data: List[Dict[str, Any]]
    ) -> tuple[List[Document], List[str]]:
        """
        將 corpus 資料轉換為 LangChain Document
        
        參數:
            corpus_data: 原始 corpus 資料列表
        
        返回:
            (documents, ids) 元組
            - documents: LangChain Document 列表
            - ids: 文檔 ID 列表
        """
        documents = []
        ids = []
        
        for doc_data in corpus_data:
            doc_id = doc_data.get("doc_id")
            content = doc_data.get("content", "")
            original_source = doc_data.get("original_source", "")
            is_gold = doc_data.get("is_gold", False)
            
            # 跳過無效文檔
            if not content or not doc_id:
                logger.warning(f"跳過無效文檔: doc_id={doc_id}")
                continue
            
            # 建立 Document 物件（No Chunking - 使用完整內容）
            doc = Document(
                page_content=content,
                metadata={
                    "doc_id": doc_id,
                    "original_source": original_source,
                    "is_gold": is_gold
                }
            )
            
            documents.append(doc)
            ids.append(doc_id)
        
        logger.info(f"成功轉換 {len(documents)} 篇文檔為 LangChain Document")
        return documents, ids
    
    def load_and_prepare(
        self,
        corpus_path: str = "data/corpus.json",
        limit: int = None
    ) -> tuple[List[Document], List[str]]:
        """
        載入並準備文檔（一次完成）
        
        參數:
            corpus_path: corpus.json 的路徑
            limit: 載入的文檔數量限制
        
        返回:
            (documents, ids) 元組
        """
        corpus_data = self.load_corpus(corpus_path, limit)
        return self.create_documents(corpus_data)
