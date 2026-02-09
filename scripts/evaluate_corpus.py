#!/usr/bin/env python3
"""
獨立 Corpus 評估腳本

使用方式：
    python evaluate_corpus.py --limit 5  # 評估 5 題
    python evaluate_corpus.py --limit 50  # 評估全部 50 題
    python evaluate_corpus.py --limit 5 --k 3  # 評估 5 題，Top-3 檢索
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# 將專案根目錄加入 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.service_layer import corpus_evaluation_service


def print_results(results: dict):
    """
    格式化輸出評估結果
    
    參數：
        results (dict): 評估結果字典
    """
    print("\n" + "="*80)
    print("📊 Corpus 評估結果")
    print("="*80)
    
    # 整體指標
    overall = results["overall"]
    print("\n【整體指標】")
    print(f"  總問題數:           {overall['total_questions']}")
    print(f"  Hit Rate (單一):    {overall['hit_rate']:.2%}")
    print(f"  Partial Hit Rate:   {overall['partial_hit_rate']:.2%}")
    print(f"  MRR:                {overall['mrr']:.4f}")
    print(f"  生成通過率:         {overall['generation_pass_rate']:.2%}")
    
    # 按資料來源分組
    print("\n【按資料來源分組】")
    print()
    
    by_source = results["by_source"]
    for source_name in ["drcd", "hotpotqa", "2wiki"]:
        if source_name not in by_source:
            continue
        
        source = by_source[source_name]
        print(f"【{source_name.upper()}】")
        print(f"  問題數:             {source['total_questions']}")
        print(f"  Hit Rate (單一):    {source['hit_rate']:.2%}")
        print(f"  Partial Hit Rate:   {source['partial_hit_rate']:.2%}")
        print(f"  MRR:                {source['mrr']:.4f}")
        print(f"  生成通過率:         {source['generation_pass_rate']:.2%}")
        print()
    
    print("="*80)


async def main():
    """
    主程式
    """
    # 解析命令列參數
    parser = argparse.ArgumentParser(description="執行 Corpus 評估")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=5,
        help="要評估的問題數量（預設 5）"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5,
        help="Top-K 檢索數量（預設 5）"
    )
    parser.add_argument(
        "--queries", 
        type=str, 
        default="data/queries.json",
        help="queries.json 的路徑（預設 queries.json）"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="將結果儲存到 JSON 檔案（選填）"
    )
    
    args = parser.parse_args()
    
    print(f"\n🚀 開始評估...")
    print(f"   問題數量: {args.limit}")
    print(f"   檢索數量: Top-{args.k}")
    print(f"   資料來源: {args.queries}")
    
    try:
        # 執行評估
        results = await corpus_evaluation_service.run_evaluation(
            queries_path=args.queries,
            k=args.k,
            limit=args.limit
        )
        
        # 輸出結果
        print_results(results)
        
        # 儲存到檔案（如果指定）
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 結果已儲存至：{args.output}")
        
        print("\n✅ 評估完成！")
        
    except FileNotFoundError as e:
        print(f"\n❌ 錯誤：{e}")
        print("\n請確保：")
        print("  1. corpus.json 存在於專案根目錄")
        print("  2. queries.json 存在於專案根目錄")
        print("  3. 已執行 corpus 資料匯入")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 評估失敗：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
