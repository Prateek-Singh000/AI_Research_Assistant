# AI Research Assistant

An AI‑powered research assistant that uses **Retrieval‑Augmented Generation (RAG)** and a **LangGraph agent** to answer questions based on your own documents. The system is exposed as a REST API.

## Features
- **RAG pipeline** – Loads PDF/txt documents, splits them into chunks, and creates a vector index using local embeddings (sentence‑transformers).
- **Intelligent agent** – Uses DeepSeek’s language model to search the index and answer questions accurately.
- **FastAPI server** – Provides a `/query` endpoint with interactive documentation at `/docs`.
- **Ready to deploy** – Can be deployed to Sevalla, Render, or any cloud platform.

## How it works
1. Place your documents (PDF or TXT) in the `data/` folder.
2. The RAG pipeline builds a FAISS index of document chunks.
3. The agent uses the `search_documents` tool to retrieve relevant chunks.
4. The agent generates a final answer using DeepSeek’s language model.

## Setup (local)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Prateek-Singh000/AI_Research_Assistant.git
   cd AI_Research_Assistant
