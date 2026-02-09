"""
快速重建 ChromaDB (含 entities from Neo4j)

不需要重新提取實體，直接從 Neo4j 讀取現有 entities
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
from langchain_core.documents import Document

from app.database.vector_store import VectorStoreManager
from app.database.graph_store import GraphStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rebuild_chromadb_with_entities(limit: int = None):
    """
    重建 ChromaDB，並從 Neo4j 取得 entities 加入 metadata
    
    步驟:
    1. 讀取 corpus.json
    2. 從 Neo4j 查詢每個文檔的 entities
    3. 建立 Document (含 entities metadata)
    4. 存入 ChromaDB
    """
    
    # 初始化
    vector_store = VectorStoreManager()
    graph_store = GraphStoreManager()
    
    # 1. 讀取 corpus
    logger.info("📖 讀取 corpus.json...")
    with open("data/corpus.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    if limit:
        corpus = corpus[:limit]
    
    logger.info(f"總共 {len(corpus)} 篇文檔")
    
    # 2. 建立 Documents（含 entities）
    documents = []
    ids = []
    
    for i, doc_data in enumerate(corpus, 1):
        doc_id = doc_data.get("doc_id")
        content = doc_data.get("content", "")
        original_source = doc_data.get("original_source", "")
        
        if not content or not doc_id:
            continue
        
        # 🔍 從 Neo4j 查詢此文檔的 entities
        try:
            query = """
            MATCH (e:Entity)
            WHERE e.doc_id = $doc_id
            RETURN e.name AS entity
            """
            results = graph_store.graph.query(query, {"doc_id": doc_id})
            entities = [record["entity"] for record in results]
        except Exception as e:
            logger.warning(f"查詢 entities 失敗 (doc {doc_id[:20]}...): {e}")
            entities = []
        
        # 建立 Document
        doc = Document(
            page_content=content,
            metadata={
                "doc_id": doc_id,
                "original_source": original_source,
                "entities": ",".join(entities) if entities else ""  # ✅ 轉成字串
            }
        )
        
        documents.append(doc)
        ids.append(doc_id)
        
        if i % 10 == 0:
            logger.info(f"處理進度: {i}/{len(corpus)} - Doc {doc_id[:20]}... 有 {len(entities)} 個 entities")
    
    # 3. 存入 ChromaDB
    logger.info(f"💾 存入 ChromaDB ({len(documents)} 篇文檔)...")
    vector_store.add_documents(documents, ids=ids)
    
    logger.info("✅ 完成！ChromaDB 已重建，metadata 含 entities")
    
    # 4. 驗證
    logger.info("\n🔍 驗證結果:")
    stats = vector_store.get_stats()
    logger.info(f"  - 總文檔數: {stats['total_documents']}")
    
    # 隨機檢查一篇
    sample_result = vector_store.collection.get(limit=1, include=['metadatas'])
    if sample_result['metadatas']:
        sample_meta = sample_result['metadatas'][0]
        logger.info(f"\n📋 範例文檔:")
        logger.info(f"  - doc_id: {sample_meta.get('doc_id', 'N/A')[:30]}...")
        logger.info(f"  - entities: {sample_meta.get('entities', [])[:5]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="重建 ChromaDB (含 Neo4j entities)")
    parser.add_argument("--limit", type=int, help="處理的文檔數量")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("🔧 ChromaDB 重建工具 (從 Neo4j 取得 entities)")
    logger.info("=" * 70)
    logger.info("⚠️  注意: 此腳本會清空並重建 ChromaDB")
    logger.info("ℹ️  Neo4j 不會被修改")
    logger.info("=" * 70)
    
    input("\n按 Enter 繼續，或 Ctrl+C 取消...")
    
    rebuild_chromadb_with_entities(args.limit)
