"""
系統配置模組

使用 Pydantic Settings 管理環境變數，從 .env 檔案載入設定。
所有的 API 金鑰和資料庫連線資訊都在這裡定義。
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import sys

# 明確載入 .env 檔案中的環境變數
# 這確保即使在不同的執行環境中也能正確讀取設定
load_dotenv()

try:
    class Settings(BaseSettings):
        """
        系統設定類別
        
        所有欄位都會從環境變數自動載入：
        - OPENAI_API_KEY: OpenAI API 金鑰（用於 LLM 呼叫）
        - NEO4J_URI: Neo4j 資料庫連線 URI
        - NEO4J_USERNAME: Neo4j 使用者名稱
        - NEO4J_PASSWORD: Neo4j 密碼
        - LANGFUSE_SECRET_KEY: Langfuse 追蹤系統的密鑰
        - LANGFUSE_PUBLIC_KEY: Langfuse 公鑰
        - LANGFUSE_HOST: Langfuse 伺服器位址（美國區或歐洲區）
        """
        OPENAI_API_KEY: str
        NEO4J_URI: str
        NEO4J_USERNAME: str
        NEO4J_PASSWORD: str
        LANGFUSE_SECRET_KEY: str
        LANGFUSE_PUBLIC_KEY: str
        LANGFUSE_HOST: str
        LLM_MODEL: str = "gpt-4o-mini"

        # Pydantic 設定：從 .env 檔案讀取，忽略空值和額外欄位
        model_config = SettingsConfigDict(
            env_file=".env", 
            env_ignore_empty=True, 
            extra="ignore"
        )

    # 建立全域設定實例
    # 如果缺少必要的環境變數，這裡會拋出錯誤
    settings = Settings()
    
except Exception as e:
    # 配置載入失敗的錯誤處理
    print(f"嚴重錯誤：無法載入配置。請檢查 .env 檔案是否存在且包含所有必要的環境變數。")
    print(f"錯誤詳情：{e}")
    # 重新拋出錯誤以停止應用程式啟動
    # 這能避免在配置不完整的情況下執行系統
    raise e

