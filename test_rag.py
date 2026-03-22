from src.rag import RAGPipeline

rag = RAGPipeline()
rag.build_vector_store()    # This will index all documents in data/
results = rag.retrieve("What is the main topic?")
for i, r in enumerate(results):
    print(f"Chunk {i+1}:\n{r[:200]}...\n")