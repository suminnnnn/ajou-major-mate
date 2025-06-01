from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.utils.auth import get_current_user
from app.domains.user.model import User
from app.agent.graph import graph
from app.agent.state import MessageState
from langchain_core.tracers import LangChainTracer
from langgraph.errors import GraphRecursionError
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

tracer = LangChainTracer()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

async def handle_graph_recursion_error(request: Request, exc: GraphRecursionError):
    return JSONResponse(
        status_code=200,
        content={"response": "관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요."},
    )

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, current_user: User = Depends(get_current_user)):
    session_id = current_user.bedrock_session_id
    user_id = current_user.id
    
    print("user_id: ",user_id,"thread_id: ",session_id)
    logger.info(f"[CHAT] New Chat\nuser_id: {user_id}\nthread_id: {session_id}\nquestion: {req.query}\n")

    inputs = MessageState(question=req.query)
    config = {
        "configurable": {
            "thread_id": session_id
        },
        "callbacks": [tracer],
        "recursion_limit": 10
    }

    try:
        result = graph.invoke(inputs, config)
    except GraphRecursionError as e:
        logger.warning(f"[GraphRecursionError] {e}")
        return ChatResponse(response="관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요.")

    return ChatResponse(response=result["generation"])