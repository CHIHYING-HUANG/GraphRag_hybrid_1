"""
Graph RAG 工作流程（LangGraph 架構）

使用 LangGraph StateGraph 實作混合檢索（向量 + 圖譜）的 Graph RAG 系統
"""

import logging
from typing import List, Dict, Any, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from app.database.vector_store import VectorStoreManager
from app.database.graph_store import GraphStoreManager
from app.services.retrieval.vector_retriever import VectorRetriever
from app.services.retrieval.graph_retriever import GraphRetriever
from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Graph State 定義
# =============================================================================

class GraphState(TypedDict):
    """
    Graph RAG 工作流程的狀態
    
    屬性：
        question: 用戶問題
        expanded_queries: 查詢擴展後的多個問題變體
        candidates: 重排序前的候選文檔列表 (用於 reranking)
        vector_context: 向量檢索的結果
        graph_context: 圖譜檢索的結果
        final_answer: 最終生成的答案
    """
    question: str
    expanded_queries: List[str]
    candidates: List[Dict[str, Any]]
    vector_context: List[str]
    graph_context: List[str]
    final_answer: str
    retrieved_doc_ids: List[str]  # 最終檢索到的文檔 ID (用於評估)


# =============================================================================
# 全域實例（延遲初始化）
# =============================================================================

_vector_store = None
_graph_store = None
_vector_retriever = None
_graph_retriever = None
_llm = None


def _get_vector_retriever() -> VectorRetriever:
    """獲取向量檢索器（單例模式）"""
    global _vector_store, _vector_retriever
    if _vector_retriever is None:
        _vector_store = VectorStoreManager()
        _vector_retriever = VectorRetriever(_vector_store)
    return _vector_retriever


def _get_graph_retriever() -> GraphRetriever:
    """獲取圖譜檢索器（單例模式）"""
    global _graph_store, _graph_retriever
    if _graph_retriever is None:
        _graph_store = GraphStoreManager()
        _graph_retriever = GraphRetriever(_graph_store)
    return _graph_retriever


def _get_llm() -> ChatOpenAI:
    """獲取 LLM（單例模式）"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
    return _llm


# =============================================================================
# LangGraph 節點函數
# =============================================================================

def query_expansion_node(state: GraphState) -> GraphState:
    """
    節點 1: 查詢擴展
    
    使用 LLM 將原始問題改寫成 3 個語意相似的變體，增加檢索召回率
    """
    question = state["question"]
    
    try:
        expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是查詢優化專家,擅長分解多步驟問題。

任務:
1. 識別問題中的核心實體和關係
2. 產生 3 個有助於找到相關文檔的查詢變體
3. 如果是多步驟問題,考慮拆解中間步驟

要求:
- 變體應涵蓋不同角度或中間步驟
- 使用同義詞、實體別名
- 每個問題用換行分隔，不要編號

範例 (多跳題):
原始問題: A的父親是在哪一年出生的？
變體問題:
A的父親是誰
A的父親出生日期
A的家族背景

原始問題: B國的首都在哪？
變體問題:
B國的行政中心位於何處
B國首都名稱
B國政府所在地
"""),
            ("human", "原始問題：{question}")
        ])
        
        llm = _get_llm()
        chain = expansion_prompt | llm
        
        # 取得 Langfuse callback
        from app.core.langfuse_helper import get_callbacks
        callbacks = get_callbacks()
        
        response = chain.invoke(
            {"question": question},
            config={"callbacks": callbacks} if callbacks else {}
        )
        
        # 解析擴展後的查詢（包含原始問題）
        expanded = [question]  # 保留原始問題
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if line and line not in expanded:
                expanded.append(line)
        
        logger.debug(f"查詢擴展：{len(expanded)} 個變體")
        return {"expanded_queries": expanded[:3]}  # 最多 3 個
        
    except Exception as e:
        logger.error(f"查詢擴展錯誤：{e}")
        return {"expanded_queries": [question]}  # fallback to original


def retrieve_vector_node(state: GraphState) -> GraphState:
    """
    節點 2：多查詢向量檢索 + 圖譜擴展
    
    使用擴展後的查詢從 ChromaDB 檢索候選文檔,並利用圖譜關係發現額外相關文檔
    """
    expanded_queries = state.get("expanded_queries", [state["question"]])
    
    try:
        vector_retriever = _get_vector_retriever()
        graph_retriever = _get_graph_retriever()
        
        # Step 1: 對每個擴展查詢進行向量檢索
        all_candidates = []
        seen_ids = set()
        initial_entities = set()  # 收集初始檢索結果中的實體
        
        for query in expanded_queries:
            docs_with_meta = vector_retriever.retrieve_with_metadata(query, k=30)
            
            for doc_meta in docs_with_meta:
                doc_id = doc_meta.get("metadata", {}).get("doc_id")
                if doc_id and doc_id not in seen_ids:
                    all_candidates.append({
                        "doc_id": doc_id,
                        "content": doc_meta.get("content", ""),
                        "metadata": doc_meta.get("metadata", {})
                    })
                    seen_ids.add(doc_id)
                    
                    # 從文檔的 entities metadata 中收集實體
                    entities = doc_meta.get("metadata", {}).get("entities", [])
                    if isinstance(entities, list) and entities:
                        initial_entities.update(entities[:3])  # 每篇文檔取前3個實體
                    else:
                        # 🔧 Fallback: metadata 沒有 entities 時，從內容粗略提取
                        content = doc_meta.get("content", "")
                        if content:
                            import re
                            # 找中文人名/地名（常見姓氏開頭）和英文專有名詞
                            chinese_names = re.findall(r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|教授|博士|總統|主席)?', content[:800])
                            # 英文專有名詞（大寫開頭的連續詞）
                            english_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', content[:800])
                            
                            potential_entities = list(set(chinese_names[:5] + english_names[:5]))
                            if potential_entities:
                                initial_entities.update(potential_entities[:3])
        
        logger.debug(f"初始向量檢索: {len(all_candidates)} 篇文檔, {len(initial_entities)} 個實體")
        
        # Step 2: 圖譜擴展 - 找出相關實體
        if initial_entities:
            import asyncio
            related_entities = asyncio.run(
                graph_retriever.get_related_entities(
                    list(initial_entities)[:5],  # 限制起始實體數量
                    max_neighbors=3
                )
            )
            
            print(f"🚀 [Graph Expansion] 發現 {len(related_entities)} 個相關實體: {related_entities}")
            logger.debug(f"圖譜擴展發現 {len(related_entities)} 個相關實體")
            
            # Step 3: 用相關實體做額外檢索
            for entity in related_entities[:5]:  # 限制擴展實體數量
                entity_docs = vector_retriever.retrieve_with_metadata(entity, k=2)  # 每個實體取2篇
                
                for doc_meta in entity_docs:
                    doc_id = doc_meta.get("metadata", {}).get("doc_id")
                    if doc_id and doc_id not in seen_ids:
                        all_candidates.append({
                            "doc_id": doc_id,
                            "content": doc_meta.get("content", ""),
                            "metadata": doc_meta.get("metadata", {})
                        })
                        seen_ids.add(doc_id)
            
            logger.debug(f"圖譜擴展後總共 {len(all_candidates)} 篇候選文檔")
        
        # 限制候選數量
        return {"candidates": all_candidates[:30]}
        
    except Exception as e:
        logger.error(f"檢索錯誤：{e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"candidates": []}


def rerank_node(state: GraphState) -> GraphState:
    """
    節點 3：LLM 重排序
    
    節點 3：LLM 重排序
    
    使用 LLM 對候選文檔進行重排序，選出最相關的 Top-5
    """
    question = state["question"]
    candidates = state.get("candidates", [])
    
    if not candidates:
        return {"vector_context": []}
    
    try:
        # 構建重排序提示詞 (Listwise Reranking)
        candidate_texts = []
        for idx, cand in enumerate(candidates[:30], 1):  # 增加到30提升複雜問題覆蓋率  # 限制最多 20 個候選
            content_preview = cand["content"][:300]  # 只取前 300 字符以節省 token
            candidate_texts.append(f"[{idx}] {content_preview}...")
        
        candidates_str = "\n\n".join(candidate_texts)
        
        rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一個文檔排序專家。請根據問題，從候選文檔中選出最相關的 5 篇文檔。只需返回文檔編號（例如：1,3,5,8,12），用逗號分隔，不要其他說明。"),
            ("human", "問題：{question}\n\n候選文檔：\n{candidates}")
        ])
        
        llm = _get_llm()
        chain = rerank_prompt | llm
        
        # 取得 Langfuse callback
        from app.core.langfuse_helper import get_callbacks
        callbacks = get_callbacks()
        
        response = chain.invoke(
            {
                "question": question,
                "candidates": candidates_str
            },
            config={"callbacks": callbacks} if callbacks else {}
        )
        
        # 解析排序結果
        # 解析排序結果 (使用 Regex 增強魯棒性)
        import re
        selected_indices = []
        # 找出所有數字
        found_numbers = re.findall(r'\d+', response.content)
        
        for num_str in found_numbers:
            try:
                idx = int(num_str)
                if 1 <= idx <= len(candidates):
                    selected_indices.append(idx - 1)  # 轉為 0-index
            except ValueError:
                continue
        
        # 取得重排序後的文檔內容 和 ID
        reranked_contents = []
        reranked_ids = []
        for idx in selected_indices[:5]:  # Top-5
            if 0 <= idx < len(candidates):
                reranked_contents.append(candidates[idx]["content"])
                reranked_ids.append(candidates[idx].get("doc_id"))
        
        # Fallback: 如果 LLM 沒有返回合法索引，使用前 5 個
        if not reranked_contents and candidates:
            reranked_contents = [c["content"] for c in candidates[:5]]
            reranked_ids = [c.get("doc_id") for c in candidates[:5]]
        
        # 過濾 None ID
        reranked_ids = [uid for uid in reranked_ids if uid]
        
        logger.debug(f"重排序後選出 {len(reranked_contents)} 篇文檔")
        return {"vector_context": reranked_contents, "retrieved_doc_ids": reranked_ids}
        
    except Exception as e:
        logger.error(f"重排序錯誤：{e}")
        # Fallback: 直接使用前 5 個候選
        fallback_contents = [c["content"] for c in candidates[:5]]
        fallback_ids = [c.get("doc_id") for c in candidates[:5]]
        return {"vector_context": fallback_contents, "retrieved_doc_ids": [uid for uid in fallback_ids if uid]}


async def retrieve_graph_node(state: GraphState) -> GraphState:
    """
    節點 4：圖譜檢索
    
    從 Neo4j 知識圖譜檢索相關實體和關係
    """
    question = state["question"]
    
    try:
        graph_retriever = _get_graph_retriever()
        graph_context = await graph_retriever.retrieve(
            question,
            max_entities=3,
            max_relations_per_entity=10  # 5→10 提供更多關係路徑
        )
        
        logger.debug(f"圖譜檢索到 {len(graph_context)} 條關係")
        return {"graph_context": graph_context}
        
    except Exception as e:
        logger.warning(f"圖譜檢索錯誤：{e}")
        return {"graph_context": []}


def generate_answer_node(state: GraphState) -> GraphState:
    """
    節點 5：答案生成
    
    基於向量和圖譜檢索的上下文生成答案
    """
    question = state["question"]
    vector_context = state.get("vector_context", [])
    graph_context = state.get("graph_context", [])
    
    try:
        # 整合上下文
        context_parts = []
        
        if vector_context:
            context_parts.append("【向量檢索上下文】\n" + "\n\n".join(vector_context))
        
        if graph_context:
            context_parts.append("【圖譜檢索上下文】\n" + "\n".join(graph_context))
        
        context_str = "\n\n".join(context_parts) if context_parts else "無相關上下文"
        
        # 生成答案 (優化的多步推理 Prompt - 支援複雜多跳與數字推理 + Chain of Thought)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一個專業的問答助手,擅長整合多篇文檔資訊並進行多步驟邏輯推理。

                回答策略:
                1. 先分析問題結構,識別需要幾個步驟
                2. 對於多步驟問題,依序完成每個步驟:
                   - 第 1 步: 從文檔中找到起點資訊 (如人名、實體)。**注意：必須精確匹配實體名稱，避免混淆同名或相似實體。**
                   - 第 2 步: 用第 1 步的結果在文檔中找中間資訊 (如關係、屬性)
                   - 第 3 步: 用第 2 步的結果找最終答案
                3. 對於涉及數值或比例的問題:
                   - **優先尋找直接答案**：如果文檔中直接提供了數值或時間長度（如「歷時83年」、「佔比20%」），**必須直接引用**，禁止自行計算。
                   - 只有在文檔未直接提供答案時，才根據文檔中的數據進行計算。
                   - 若文檔提供的數據與問題詢問的角度相反，請進行簡單的數學轉換 (如 100% - X%) 以驗證答案。
                4. **邏輯一致性檢查**：
                   - 對於是非題（Yes/No），確保你的結論（是/否）與你的解釋完全一致。
                   - 例如：如果解釋是「A來自德國，B來自美國」，結論必須是「不是」（來自不同國家）。
                5. 整合所有資訊得出結論

                重要原則:
                - **嚴格以文檔為準**：如果文檔中有明確資訊，必須優先使用文檔內容，而非預訓練知識（例如不要使用外部知識補充年份）
                - 即使資訊分散在 2-4 篇不同文檔,也要努力整合
                - 對於關係類問題 (如"繼父"),明確推理關係鏈
                - 遇到數字問題,請精確核對文檔中的數據,不要憑印象回答
                - 只有在上下文完全沒有相關資訊時,才回答「根據提供的資料無法回答此問題」

                **輸出格式要求**：
                請務必按照以下 XML 格式輸出你的思考過程和最終答案：
                <reasoning>
                這裡寫下你的逐步推理過程...
                1. 根據文檔X...
                2. 發現...
                3. 因此...
                </reasoning>
                <answer>
                這裡寫下最終答案（精簡、直接）
                </answer>

                範例參考:
                
                範例1 - 優先使用文檔數值:
                Q: 某朝代持續了多久？
                Doc: "某朝代建立於100年，歷時300年，於400年滅亡。"
                <reasoning>
                1. 文檔明確提到「歷時300年」。
                2. 雖然400-100=300，但文檔已有直接答案。
                </reasoning>
                <answer>
                300年
                </answer>
                
                範例2 - 數值比較 (通用邏輯):
                Q: 蘋果和橘子哪一個比較重？
                Doc: "蘋果重200克，橘子重150克。"
                <reasoning>
                1. 文檔指出蘋果重200克。
                2. 文檔指出橘子重150克。
                3. 200克 > 150克，所以蘋果比較重。
                </reasoning>
                <answer>
                蘋果
                </answer>
                
                範例3 - 跨文檔推理 (地理/機構):
                Q: Alpha公司的總部所在的城市，其市長是誰？
                Doc1: "Alpha公司的總部位於貝克市。"
                Doc2: "貝克市的市長是詹姆斯·史密斯。"
                <reasoning>
                1. 從Doc1得知Alpha公司總部在貝克市。
                2. 從Doc2得知貝克市市長是詹姆斯·史密斯。
                3. 因此答案是詹姆斯·史密斯。
                </reasoning>
                <answer>
                詹姆斯·史密斯
                </answer> 
                
                範例4 - 實體區分 (科學/定義):
                Q: 什麼是「光合作用」的主要產物？
                Doc1: "呼吸作用產生二氧化碳和水。"
                Doc2: "光合作用將光能轉化為化學能，產生葡萄糖和氧氣。"
                <reasoning>
                1. 問題詢問光合作用的產物。
                2. Doc1描述呼吸作用，不相關。
                3. Doc2明確指出光合作用產生葡萄糖和氧氣。
                </reasoning>
                <answer>
                葡萄糖和氧氣
                </answer>"""),
            ("human", "上下文:\n{context}\n\n問題:{question}\n\n請依照指定格式輸出:")
        ])
        
        llm = _get_llm()
        chain = prompt | llm
        
        # 取得 Langfuse callback（統一管理）
        from app.core.langfuse_helper import get_callbacks
        callbacks = get_callbacks()
        
        response = chain.invoke(
            {
                "context": context_str,
                "question": question
            },
            config={"callbacks": callbacks} if callbacks else {}
        )
        
        # 解析輸出，提取 <answer> 標籤內容
        import re
        content = response.content
        final_answer = content
        
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            final_answer = match.group(1).strip()
        else:
            # Fallback: 如果沒有標籤，嘗試移除可能的 <reasoning> 部分
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
            if reasoning_match:
                final_answer = content.replace(reasoning_match.group(0), "").strip()
            
        return {"final_answer": final_answer}
        
    except Exception as e:
        logger.error(f"生成答案錯誤：{e}")
        return {"final_answer": "抱歉，生成答案時發生錯誤。"}


# =============================================================================
# 建立 LangGraph 工作流程
# =============================================================================

def create_graph_rag_workflow() -> StateGraph:
    """
    建立 Graph RAG 工作流程（優化版）
    
    工作流程：
    1. query_expansion：查詢擴展（生成 3 個問題變體）
    2. retrieve_vector：多查詢向量檢索（Top-20 候選）
    3. rerank：LLM 重排序（選出 Top-5）
    4. retrieve_graph：圖譜檢索
    5. generate_answer：生成答案
    
    回傳：
        StateGraph: 編譯後的工作流程圖
    """
    # 建立狀態圖
    workflow = StateGraph(GraphState)
    
    # 添加節點
    workflow.add_node("query_expansion", query_expansion_node)
    workflow.add_node("retrieve_vector", retrieve_vector_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("retrieve_graph", retrieve_graph_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # 定義邊（工作流程）
    workflow.set_entry_point("query_expansion")
    workflow.add_edge("query_expansion", "retrieve_vector")
    workflow.add_edge("retrieve_vector", "rerank")
    workflow.add_edge("rerank", "retrieve_graph")
    workflow.add_edge("retrieve_graph", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # 編譯圖
    return workflow.compile()


# 建立全域工作流程實例
_graph_rag_workflow = None


def _get_workflow():
    """獲取工作流程實例（單例模式）"""
    global _graph_rag_workflow
    if _graph_rag_workflow is None:
        _graph_rag_workflow = create_graph_rag_workflow()
    return _graph_rag_workflow


# =============================================================================
# 便捷函數
# =============================================================================

async def run_graph_rag(question: str) -> Dict[str, Any]:
    """
    執行完整的 Graph RAG 流程
    
   參數:
        question (str): 用戶問題
    
    回傳:
        Dict: 包含問題、答案、上下文和延遲指標的字典
    """
    import time
    
    # 記錄開始時間
    start_time = time.time()
    
    # 準備初始狀態
    initial_state = {
        "question": question,
        "expanded_queries": [],
        "candidates": [],
        "vector_context": [],
        "graph_context": [],
        "final_answer": ""
    }
    
    # 執行工作流程（檢索階段）
    retrieval_start = time.time()
    workflow = _get_workflow()
    result = await workflow.ainvoke(initial_state)
    retrieval_time = time.time() - retrieval_start
    
    # 生成階段的時間已包含在 workflow 中，這裡記錄總時間
    total_time = time.time() - start_time
    
    # 估算生成時間（總時間 - 檢索時間的合理部分）
    # 注意：這是近似值，因為工作流包含多個階段
    generation_time = max(0, total_time - retrieval_time * 0.5)
    
    return {
        "question": question,
        "answer": result["final_answer"],
        "vector_context": result.get("vector_context", []),
        "graph_context": result.get("graph_context", []),
        "retrieved_doc_ids": result.get("retrieved_doc_ids", []),
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time
    }
