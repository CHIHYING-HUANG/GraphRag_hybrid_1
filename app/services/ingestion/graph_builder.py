"""
知識圖譜構建器

負責從文本中提取實體和關係，構建知識圖譜
"""

import logging
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.database.graph_store import GraphStoreManager, Entity, Relationship
from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 圖譜資料結構
# =============================================================================

class GraphData(BaseModel):
    """完整的圖譜資料結構"""
    entities: List[Entity]
    relationships: List[Relationship]





# =============================================================================
# 圖譜構建器
# =============================================================================

class GraphBuilder:
    """
    知識圖譜構建器
    
    職責：
    - 使用 LLM 從文本提取實體和關係
    - 將提取的資料儲存到 Neo4j
    - 支援批次處理
    """
    
    def __init__(
        self,
        graph_store: GraphStoreManager,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0
    ):
        """
        初始化圖譜構建器
        
        參數:
            graph_store: 圖譜儲存管理器
            llm_model: LLM 模型名稱
            temperature: LLM 溫度參數
        """
        self.graph_store = graph_store
        
        # 初始化 LLM（使用 structured_output 確保格式正確）
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY
        ).with_structured_output(GraphData)
        
        logger.info("GraphBuilder 初始化完成")
    
    async def extract_and_store(self, content: str, doc_id: str):
        """
        從文本提取實體和關係，並儲存到圖譜
        
        參數:
            content: 文檔內容
            doc_id: 文檔 ID
        """
        try:
            # 構建提示詞
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一個圖譜提取演算法。從文本中提取實體和關係。"),
                ("human", "{text}")
            ])
            
            chain = extraction_prompt | self.llm
            
            # 取得 Langfuse callback（統一管理）
            from app.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()
            
            # 執行圖譜提取（限制長度以節省成本）
            graph_data: GraphData = chain.invoke(
                {"text": content[:1000]},
                config={"callbacks": callbacks} if callbacks else {}
            )
            
            # 儲存實體
            for entity in graph_data.entities:
                self.graph_store.add_entity(entity, doc_id)
            
            # 儲存關係
            for rel in graph_data.relationships:
                self.graph_store.add_relationship(rel, doc_id)
            
            logger.debug(
                f"Doc {doc_id}: 提取 {len(graph_data.entities)} 個實體，"
                f"{len(graph_data.relationships)} 個關係"
            )
            
        except Exception as e:
            # 圖譜提取失敗不影響向量儲存
            logger.warning(f"圖譜提取失敗（Doc {doc_id}）：{e}")
    
    async def build_batch(
        self,
        content_list: List[tuple[str, str]]
    ):
        """
        批次構建圖譜
        
        參數:
            content_list: (content, doc_id) 元組列表
        """
        for content, doc_id in content_list:
            await self.extract_and_store(content, doc_id)
