from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG_chain import init_RAG_engine
from agent_chain import build_agent
from config import DOCUMENT_PATH


RAG_query = init_RAG_engine(DOCUMENT_PATH)
agent_query = build_agent()

app = FastAPI(
    title="RAG API",
    description="基于 ChromaDB + DeepSeek + LangChain 的问答服务",
)


class Question(BaseModel):
    question: str


@app.post("/chat")
def chat(q: Question):
    try:
        answer = RAG_query(q.question)
        return {"question": q.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agent")
def agent_endpoint(req: Question):
    try:
        answer = agent_query(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
