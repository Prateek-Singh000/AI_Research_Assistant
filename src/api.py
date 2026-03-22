from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .agent import create_agent
from .logger import logger
import uvicorn

app = FastAPI(title="AI Research Assistant", version="1.0")

# Instantiate agent once
agent = create_agent()

class QueryRequest(BaseModel):
    query: str
    session_id: str = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    logger.info("Received query: %s", request.query)
    try:
        result = agent.invoke({"input": request.query})
        logger.info("Query succeeded")
        return QueryResponse(answer=result["output"], session_id=request.session_id)
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)