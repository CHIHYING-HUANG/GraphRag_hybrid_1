
import asyncio
import os
from app.services.rag.graph_rag import run_graph_rag

async def main():
    question = "高句麗在直到滅亡前多久時間其首都都在長安城？"
    print(f"Question: {question}")
    
    result = await run_graph_rag(question)
    
    print("\nFull Result:")
    print(result["answer"])
    
    print("\nVector Context:")
    for ctx in result["vector_context"]:
        print("-" * 20)
        print(ctx)

if __name__ == "__main__":
    asyncio.run(main())
