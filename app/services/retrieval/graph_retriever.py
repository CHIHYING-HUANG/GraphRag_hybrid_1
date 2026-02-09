"""
圖譜檢索器

負責從知識圖譜檢索相關實體和關係
"""

import logging
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.database.graph_store import GraphStoreManager
from app.core.config import settings

logger = logging.getLogger(__name__)


class GraphRetriever:
    """
    圖譜檢索器
    
    職責：
    - 從問題中提取實體
    - 查詢 Neo4j 圖譜
    - 格式化圖譜檢索結果
    """
    
    def __init__(
        self,
        graph_store: GraphStoreManager,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0
    ):
        """
        初始化圖譜檢索器
        
        參數:
            graph_store: 圖譜儲存管理器
            llm_model: LLM 模型名稱（用於實體提取）
            temperature: LLM 溫度參數
        """
        self.graph_store = graph_store
        
        # 初始化 LLM（用於實體提取）
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        logger.info("GraphRetriever 初始化完成")
    
    async def retrieve(
        self,
        question: str,
        max_entities: int = 3,
        max_relations_per_entity: int = 5
    ) -> List[str]:
        """
        執行圖譜檢索，返回關係字串列表
        
        參數:
            question: 查詢問題
            max_entities: 最多提取的實體數量
            max_relations_per_entity: 每個實體最多返回的關係數量
        
        返回:
            關係字串列表（格式："實體 -[關係]-> 鄰居實體"）
        """
        try:
            # 1. 使用 LLM 提取問題中的實體
            entities = await self._extract_entities(question)
            
            if not entities:
                logger.debug("未提取到實體")
                return []
            
            # 限制實體數量
            entities = entities[:max_entities]
            
            # 2. 查詢每個實體的關係
            graph_context = []
            for entity in entities:
                results = self.graph_store.query_entity(
                    entity,
                    limit=max_relations_per_entity
                )
                
                for r in results:
                    rel_str = f"{r['entity']} -[{r['relationship']}]-> {r['neighbor']}"
                    graph_context.append(rel_str)
            
            logger.debug(f"圖譜檢索到 {len(graph_context)} 條關係")
            return graph_context
            
        except Exception as e:
            logger.warning(f"圖譜檢索失敗: {e}")
            return []
    
    async def get_related_entities(
        self,
        entities: List[str],
        max_neighbors: int = 5
    ) -> List[str]:
        """
        從圖譜中找出給定實體的相關實體 (用於擴展檢索)
        
        參數:
            entities: 起始實體列表
            max_neighbors: 每個實體最多返回的鄰居數量
        
        返回:
            相關實體名稱列表 (去重後)
        """
        try:
            related_entities = set()
            
            for entity in entities:
                results = self.graph_store.query_entity(
                    entity,
                    limit=max_neighbors
                )
                
                # 收集鄰居實體
                for r in results:
                    neighbor = r.get('neighbor')
                    if neighbor and neighbor not in entities:  # 避免重複
                        related_entities.add(neighbor)
            
            result = list(related_entities)
            logger.debug(f"從 {len(entities)} 個實體擴展到 {len(result)} 個相關實體")
            return result
            
        except Exception as e:
            logger.warning(f"圖譜擴展失敗: {e}")
            return []
    
    async def _extract_entities(self, question: str) -> List[str]:
        """
        從問題中提取實體
        
        參數:
            question: 查詢問題
        
        返回:
            實體名稱列表
        """
        try:
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", "從問題中提取主要實體，以逗號分隔的列表形式回答。只返回實體名稱，不需要其他說明。"),
                ("human", "{question}")
            ])
            
            chain = extraction_prompt | self.llm
            
            # 取得 Langfuse callback（統一管理）
            from app.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()
            
            result = chain.invoke(
                {"question": question},
                config={"callbacks": callbacks} if callbacks else {}
            )
            
            # 解析實體列表
            entities = [e.strip() for e in result.content.split(",") if e.strip()]
            logger.debug(f"提取到 {len(entities)} 個實體: {entities}")
            
            return entities
            
        except Exception as e:
            logger.error(f"實體提取失敗: {e}")
            return []
