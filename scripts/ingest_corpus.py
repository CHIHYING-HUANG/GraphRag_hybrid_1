#!/usr/bin/env python3
"""
獨立 Corpus 資料匯入腳本

使用方式：
    python ingest_corpus.py --limit 10   # 匯入前 10 篇文檔
    python ingest_corpus.py              # 匯入全部文檔
"""

import asyncio
import argparse
import sys
from pathlib import Path

# 將專案根目錄加入 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.corpus_ingestion import corpus_ingestion_service


async def main():
    """
    主程式
    """
    # 解析命令列參數
    parser = argparse.ArgumentParser(description="匯入 Corpus 資料")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="要匯入的文件數量（預設全部）"
    )
    parser.add_argument(
        "--corpus", 
        type=str, 
        default="data/corpus.json",
        help="corpus.json 的路徑（預設 data/corpus.json）"
    )
    
    args = parser.parse_args()
    
    print(f"\n🚀 開始匯入 Corpus 資料...")
    print(f"   資料來源: {args.corpus}")
    print(f"   匯入數量: {'全部' if args.limit is None else args.limit}")
    
    try:
        # 執行匯入
        result = await corpus_ingestion_service.ingest_corpus(
            corpus_path=args.corpus,
            limit=args.limit
        )
        
        # 輸出結果
        print("\n" + "="*80)
        print("📊 匯入結果")
        print("="*80)
        print(f"  訊息:           {result['message']}")
        print(f"  已處理文檔:     {result['docs_processed']}")
        print(f"  總文檔數:       {result['total_docs']}")
        print("="*80)
        
        # 取得統計資訊
        stats = corpus_ingestion_service.get_stats()
        print("\n📈 向量資料庫統計")
        print("="*80)
        print(f"  Collection:     {stats.get('collection_name', 'N/A')}")
        print(f"  總文檔數:       {stats.get('total_documents', 'N/A')}")
        print(f"  Embedding 模型: {stats.get('embedding_model', 'N/A')}")
        print("="*80)
        
        print("\n✅ 匯入完成！")
        
    except FileNotFoundError as e:
        print(f"\n❌ 錯誤：{e}")
        print(f"\n請確保 corpus.json 存在於路徑：{args.corpus}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 匯入失敗：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
