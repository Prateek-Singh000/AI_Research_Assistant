import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .rag import RAGPipeline
from .logger import logger

load_dotenv()

# Initialize RAG (load existing index if present)
rag = RAGPipeline()
rag.load_vector_store()

@tool
def search_documents(query: str) -> str:
    """Search the internal knowledge base for relevant information."""
    docs = rag.retrieve(query, k=3)
    if not docs:
        return "No relevant information found in the documents."
    return "\n\n".join(docs)

def create_agent():
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0.3
    )

    tools = [search_documents]

    # Create the agent using langgraph's React agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="You are a helpful research assistant. Use the `search_documents` tool to answer questions based on the provided documents. If the tool doesn't return enough information, say so."
    )

    logger.info("Agent created with tools: %s", [t.name for t in tools])
    return agent