import sys
import os

# 將當前目錄加入 Python 路徑
sys.path.append(os.getcwd())

# 載入環境變數
from dotenv import load_dotenv
load_dotenv()

from app.database.vector_store import VectorStoreManager

# 讀取向量庫中的文檔
try:
    vs = VectorStoreManager()
    collection = vs.vector_store._collection
    data = collection.get()
    
    docs = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    
    print(f"Total docs in Chroma: {len(docs)}")
    
    if docs:
        print("\n=== Sample Document 1 ===")
        print(f"ID: {data['ids'][0]}")
        print(f"Content Preview: {docs[0][:300]}...")  # 只印前 300 字
        print(f"Metadata: {metadatas[0]}")
    else:
        print("Vector store is empty!")

except Exception as e:
    print(f"Error: {e}")
