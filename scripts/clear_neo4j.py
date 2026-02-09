#!/usr/bin/env python3
"""
清理 Neo4j 圖譜資料庫

此腳本會刪除 Neo4j 中所有的節點和關係，為新的 corpus 圖譜騰出空間。

使用方式：
    python clear_neo4j.py
"""

import sys
from pathlib import Path

# 將專案根目錄加入 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_neo4j import Neo4jGraph
from app.core.config import settings


def clear_neo4j():
    """
    清空 Neo4j 資料庫中的所有節點和關係
    """
    print("\n⚠️  警告：此操作將刪除 Neo4j 中的所有資料！")
    print("=" * 80)
    
    # 連接到 Neo4j
    try:
        graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
        print("✅ 成功連接到 Neo4j")
    except Exception as e:
        print(f"❌ 連接失敗：{e}")
        return
    
    # 確認操作
    response = input("\n是否確定要刪除所有資料？(yes/no): ")
    if response.lower() != "yes":
        print("❌ 操作已取消")
        return
    
    print("\n🗑️  開始清理...")
    
    # 刪除所有節點和關係
    try:
        # 查詢當前節點數量
        result = graph.query("MATCH (n) RETURN count(n) as count")
        count_before = result[0]["count"] if result else 0
        print(f"   原有節點數: {count_before}")
        
        # 刪除所有節點（會自動刪除關係）
        graph.query("MATCH (n) DETACH DELETE n")
        
        # 驗證清理結果
        result = graph.query("MATCH (n) RETURN count(n) as count")
        count_after = result[0]["count"] if result else 0
        
        print(f"   剩餘節點數: {count_after}")
        
        if count_after == 0:
            print("\n✅ Neo4j 資料庫已清空！")
            print("   現在可以匯入 corpus 資料並建立新的知識圖譜。")
        else:
            print(f"\n⚠️  警告：仍有 {count_after} 個節點")
            
    except Exception as e:
        print(f"\n❌ 清理失敗：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    clear_neo4j()
