"""
Corpus Graph RAG 系統主程式（重構版）

Fastapi 應用程式，提供：
1. Corpus 相關 API（/ingest_corpus, /evaluate_corpus, /corpus_stats）
2. Graph RAG 問答（/chat）
3. Web UI 介面

使用新的模組化架構：
- database/: 資料庫層
- services/ingestion/: 資料處理層
- services/retrieval/: 檢索層
- services/rag/: RAG 層（LangGraph）
- services/evaluation/: 評估層
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.models import schemas


# 建立 FastAPI 應用程式實例
app = FastAPI(
    title="Corpus Graph RAG System",
    description="Graph RAG 系統（混合檢索：向量 + 知識圖譜）使用 LangGraph 框架實作",
    version="4.0.0"
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載靜態檔案
app.mount("/static", StaticFiles(directory="static"), name="static")


# =============================================================================
# 首頁路由
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """根路徑：重導向到 Web UI"""
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    """健康檢查端點"""
    return {"status": "healthy", "version": "4.0.0"}


# =============================================================================
# Graph RAG 問答 API
# =============================================================================

@app.post("/chat", response_model=schemas.ChatResponse)
async def chat(request: schemas.ChatRequest):
    """
    Graph RAG 問答端點
    
    使用 LangGraph 工作流程執行混合檢索（向量 + 圖譜）並生成答案
    """
    from app.services.rag.graph_rag import run_graph_rag
    
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="訊息列表不能為空")
        
        user_message = request.messages[-1].content
        
        # 執行 RAG 工作流程
        result = await run_graph_rag(user_message)
        
        # 組合上下文
        context = result["vector_context"] + result["graph_context"]
        
        return schemas.ChatResponse(
            answer=result["answer"],
            context=context
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Corpus 資料匯入 API
# =============================================================================

@app.post("/ingest_corpus", response_model=schemas.CorpusIngestResponse)
async def ingest_corpus(request: schemas.CorpusIngestRequest):
    """
    Corpus 資料匯入端點
    
    執行完整的資料處理流程：
    1. 載入 corpus.json
    2. 向量化並存入 ChromaDB
    3. 提取實體和關係並存入 Neo4j
    """
    from app.services.service_layer import corpus_ingestion_service
    
    try:
        result = await corpus_ingestion_service.ingest(limit=request.limit)
        return schemas.CorpusIngestResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Corpus 評估 API
# =============================================================================

@app.post("/evaluate_corpus", response_model=schemas.CorpusEvaluateResponse)
async def evaluate_corpus(limit: int = 5, k: int = 5):
    """
    Corpus 評估端點（整體指標）
    
    執行評估並返回整體和分組指標
    """
    from app.services.service_layer import corpus_evaluation_service
    
    try:
        results = await corpus_evaluation_service.run_evaluation(
            limit=limit,
            k=k
        )
        return schemas.CorpusEvaluateResponse(**results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate_corpus_detailed", response_model=schemas.DetailedEvaluateResponse)
async def evaluate_corpus_detailed(limit: int = 5, k: int = 5):
    """
    Corpus 詳細評估端點（包含每個問題的詳細信息）
    
    執行評估並返回：
    1. 整體和分組指標
    2. 每個問題的詳細分析（問題、答案、判斷結果等）
    """
    from app.services.service_layer import corpus_evaluation_service
    
    try:
        results = await corpus_evaluation_service.evaluate_detailed(
            limit=limit,
            k=k
        )
        return results
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 系統統計 API
# =============================================================================

@app.get("/corpus_stats")
async def get_corpus_stats():
    """
    取得系統統計資訊
    
    返回向量資料庫和圖譜資料庫的統計信息
    """
    from app.services.service_layer import corpus_ingestion_service
    
    try:
        stats = corpus_ingestion_service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 啟動事件
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    應用程式啟動時執行
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("🚀 Corpus Graph RAG System 已啟動")
    logger.info("=" * 70)
    logger.info("📖 API 文檔: http://localhost:8000/docs")
    logger.info("🌐 Web UI:   http://localhost:8000/static/index.html")
    logger.info("💚 健康檢查: http://localhost:8000/health")
    logger.info("=" * 70)
