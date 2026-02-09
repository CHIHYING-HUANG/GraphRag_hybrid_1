"""
Langfuse 追蹤輔助模組

提供統一的 Langfuse callback 配置
"""

import logging
from typing import List, Optional
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from app.core.config import settings

logger = logging.getLogger(__name__)


def get_langfuse_callback() -> Optional[LangfuseCallbackHandler]:
    """
    取得 Langfuse callback handler
    
    返回:
        LangfuseCallbackHandler 或 None（如果配置不完整）
    """
    try:
        # 檢查是否有必要的配置
        if not all([
            settings.LANGFUSE_SECRET_KEY,
            settings.LANGFUSE_PUBLIC_KEY,
            settings.LANGFUSE_HOST
        ]):
            logger.info("Langfuse 配置不完整，追蹤功能將被停用")
            return None
        
        # 初始化 Langfuse Handler
        # 注意：Langfuse 會自動讀取環境變數 LANGFUSE_SECRET_KEY 等
        langfuse_handler = LangfuseCallbackHandler() # Assuming LangfuseCallbackHandler can be initialized without args and reads from env vars
        
        logger.info("Langfuse 追蹤已啟用")
        return langfuse_handler # Changed from 'handler' to 'langfuse_handler'
        
    except Exception as e:
        logger.warning(f"無法初始化 Langfuse：{e}")
        return None


def get_callbacks() -> List:
    """
    取得所有 callback handlers 列表
    
    返回:
        callback handlers 列表（可能為空）
    """
    callbacks = []
    
    langfuse_handler = get_langfuse_callback()
    if langfuse_handler:
        callbacks.append(langfuse_handler)
    
    return callbacks
