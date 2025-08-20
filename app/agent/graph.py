from langgraph.graph import END, StateGraph
from langchain.schema.runnable import RunnableLambda
from app.agent.state import MessageState
from app.agent.node import *
from app.evaluation.node import evaluate_answer
from app.domains.course.graph import course_app
from app.domains.curriculum.graph import curriculum_app
from app.domains.department_intro.graph import department_intro_app
from app.domains.employment_status.graph import employment_status_app
from app.config.checkpointer import FixedBedrockSessionSaver

import os

workflow = StateGraph(MessageState)

workflow.add_node("query_filter", query_filter)
workflow.add_node("route_query", route_query)
workflow.add_node("decision", decision)
workflow.add_node("course", course_app)
workflow.add_node("curriculum", curriculum_app)
workflow.add_node("department_intro", department_intro_app)
workflow.add_node("employment_status", employment_status_app)
workflow.add_node("evaluate_answer", evaluate_answer)

workflow.set_entry_point("query_filter")

workflow.add_conditional_edges(
    "query_filter",
    lambda state: "end" if state.get("inappropriate") else "route_query",
    {"end": END, "route_query": "route_query"}
)

workflow.add_conditional_edges(
    "route_query",
    decision,
    {
        "course": "course",
        "curriculum": "curriculum",
        "department_intro": "department_intro",
        "employment_status": "employment_status",
        "other": END
    }
)

for node in ["course", "curriculum", "department_intro", "employment_status"]:
    workflow.add_edge(node, "evaluate_answer")
workflow.add_edge("evaluate_answer", END)

bedrock_checkpointer = FixedBedrockSessionSaver(
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

graph = workflow.compile(checkpointer=bedrock_checkpointer)