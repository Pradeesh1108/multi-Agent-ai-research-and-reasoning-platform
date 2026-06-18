import asyncio
from app.config.settings import get_settings
from app.rag.embeddings import EmbeddingManager
from app.rag.vector_store import VectorStoreManager

async def test_faiss():
    settings = get_settings()
    emb = EmbeddingManager(settings.embedding_model)
    store = VectorStoreManager(settings.faiss_index_path, emb)
    await store.initialise()
    
    query = "what is the username present in the resume"
    print(f"Searching for: {query}")
    
    # Bypass distance threshold to see raw scores
    results = store._vectorstore.similarity_search_with_score(query, k=10)
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source_file", "unknown")
        content = doc.page_content[:100].replace('\n', ' ')
        print(f"{i+1}. Score: {score:.4f} | Source: {source} | Content: {content}...")

if __name__ == "__main__":
    asyncio.run(test_faiss())
