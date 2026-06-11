from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from RAG_engine import RAGEngine
from llm_client import LLM_client
from tools import calculate_tool, run_calculator
from config import DOCUMENT_PATH
import json

RAG = RAGEngine()
RAG.init_from_text_file(DOCUMENT_PATH)

app = FastAPI(
    title="RAG API",
    description="基于 ChromaDB + DeepSeek 的问答服务",
)


class Question(BaseModel):
    question: str


@app.post("/chat")
def chat(q: Question):
    try:
        answer = RAG.query(q.question)
        return {"question": q.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/agent")
def agent_endpoint(req: Question):
    messages = [{"role": "user", "content": req.question}]

    first_response = LLM_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=[calculate_tool],
        tool_choice="auto",
    )

    message = first_response.choices[0].message

    if not message.tool_calls:
        return {"question": req.question, "answer": message.content}

    for tool_call in message.tool_calls:
        if tool_call.function.name == "calculator":
            args = json.loads(tool_call.function.arguments)
            expression = args.get("expression", "")
            result = run_calculator(expression)
            messages.append(message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    final_reponse = LLM_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    final_answer = final_reponse.choices[0].message.content
    return {"question": req.question, "answer": final_answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
