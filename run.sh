#!/bin/bash
echo "🚀 正在啟動 Graph RAG 系統..."
echo "📱 Web UI: http://localhost:8000/static/index.html"
echo "📄 API Docs: http://localhost:8000/docs"
echo "-----------------------------------"
./venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
