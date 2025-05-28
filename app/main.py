from fastapi import FastAPI
from langgraph.errors import GraphRecursionError
from app.api import user_router, chat_router, data_router
from app.api.chat_router import handle_graph_recursion_error

app = FastAPI(title="Ajou Major Mate", version="1.0.0", debug=True)

app.add_exception_handler(GraphRecursionError, handle_graph_recursion_error)

app.include_router(user_router.router, prefix="/users", tags=["User"])
app.include_router(chat_router.router, prefix="/chat", tags=["Chat"])
app.include_router(data_router.router, prefix="/data", tags=["Data"])