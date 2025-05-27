from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from app.utils.auth import get_current_user
from app.domains.user.model import User
from app.agent.graph import graph
from app.agent.state import MessageState
from langchain_core.tracers import LangChainTracer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

tracer = LangChainTracer()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

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
        "callbacks": [tracer]
    }

    result = graph.invoke(inputs, config)
    
    answer = result["generation"]

    return ChatResponse(response=answer)