"""
圖譜資料庫管理器

負責所有 Neo4j 圖譜資料庫的操作
"""

import logging
from typing import List, Dict, Any
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel, Field
from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 資料結構定義
# =============================================================================

class Entity(BaseModel):
    """實體模型"""
    name: str = Field(description="實體名稱")
    type: str = Field(description="實體類型（如 PERSON/ORGANIZATION/LOCATION）")


class Relationship(BaseModel):
    """關係模型"""
    source: str = Field(description="來源實體名稱")
    target: str = Field(description="目標實體名稱")
    type: str = Field(description="關係類型")
    description: str = Field(description="關係的簡短描述")


# =============================================================================
# 圖譜儲存管理器
# =============================================================================

class GraphStoreManager:
    """
    Neo4j 圖譜儲存管理器
    
    職責：
    - 初始化和管理 Neo4j 連線
    - 提供實體和關係的 CRUD 操作
    - 提供圖譜查詢介面
    """
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None
    ):
        """
        初始化圖譜資料庫管理器
        
        參數:
            uri: Neo4j URI（預設從 settings 讀取）
            username: 用戶名（預設從 settings 讀取）
            password: 密碼（預設從 settings 讀取）
        """
        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        
        # 初始化 Neo4j 連線
        self.graph = Neo4jGraph(
            url=self.uri,
            username=self.username,
            password=self.password
        )
        
        logger.info("GraphStoreManager 初始化完成")
    
    def add_entity(self, entity: Entity, doc_id: str = None):
        """
        新增實體到圖譜
        
        參數:
            entity: 實體物件
            doc_id: 關聯的文檔 ID（可選）
        """
        try:
            cypher_query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET e.type = $type, e.doc_id = $doc_id
            ON MATCH SET e.doc_id = $doc_id
            """
            self.graph.query(
                cypher_query,
                {
                    "name": entity.name,
                    "type": entity.type,
                    "doc_id": doc_id
                }
            )
            logger.debug(f"成功新增實體: {entity.name}")
        except Exception as e:
            logger.error(f"新增實體失敗: {e}")
            raise
    
    def add_relationship(self, relationship: Relationship, doc_id: str = None):
        """
        新增關係到圖譜
        
        參數:
            relationship: 關係物件
            doc_id: 關聯的文檔 ID（可選）
        """
        try:
            # 使用動態關係類型（需要字串插值，注意 Cypher 注入風險）
            # 這裡我們確保 relationship.type 是安全的
            rel_type = relationship.type.upper().replace(" ", "_")
            
            cypher_query = f"""
            MERGE (source:Entity {{name: $source}})
            MERGE (target:Entity {{name: $target}})
            MERGE (source)-[r:{rel_type}]->(target)
            ON CREATE SET r.description = $description, r.doc_id = $doc_id
            """
            
            self.graph.query(
                cypher_query,
                {
                    "source": relationship.source,
                    "target": relationship.target,
                    "description": relationship.description,
                    "doc_id": doc_id
                }
            )
            logger.debug(f"成功新增關係: {relationship.source} -[{rel_type}]-> {relationship.target}")
        except Exception as e:
            logger.error(f"新增關係失敗: {e}")
            raise
    
    def query_entity(self, entity_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        查詢實體的所有關係
        
        參數:
            entity_name: 實體名稱
            limit: 最多返回的關係數量
        
        返回:
            關係資訊列表
        """
        try:
            cypher_query = """
            MATCH (e:Entity {name: $name})-[r]-(neighbor)
            RETURN e.name as entity, type(r) as relationship, neighbor.name as neighbor
            LIMIT $limit
            """
            
            results = self.graph.query(
                cypher_query,
                {
                    "name": entity_name,
                    "limit": limit
                }
            )
            
            return results
        except Exception as e:
            logger.error(f"查詢實體失敗: {e}")
            return []
    
    def query_cypher(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        執行自定義 Cypher 查詢
        
        參數:
            cypher_query: Cypher 查詢語句
            params: 查詢參數
        
        返回:
            查詢結果列表
        """
        try:
            results = self.graph.query(cypher_query, params or {})
            return results
        except Exception as e:
            logger.error(f"Cypher 查詢失敗: {e}")
            return []
    
    def clear_all(self):
        """
        清空圖譜中的所有節點和關係（危險操作）
        """
        try:
            logger.warning("準備清空 Neo4j 圖譜資料庫")
            cypher_query = "MATCH (n) DETACH DELETE n"
            self.graph.query(cypher_query)
            logger.info("成功清空 Neo4j 圖譜資料庫")
        except Exception as e:
            logger.error(f"清空圖譜失敗: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        取得圖譜統計資訊
        
        返回:
            包含節點數、關係數等統計資訊
        """
        try:
            # 統計節點數
            node_count_query = "MATCH (n) RETURN count(n) as count"
            node_result = self.graph.query(node_count_query)
            node_count = node_result[0]["count"] if node_result else 0
            
            # 統計關係數
            rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
            rel_result = self.graph.query(rel_count_query)
            rel_count = rel_result[0]["count"] if rel_result else 0
            
            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "database": "Neo4j"
            }
        except Exception as e:
            logger.error(f"取得統計資訊失敗: {e}")
            return {"error": str(e)}
