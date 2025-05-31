from langgraph.graph import END, StateGraph
from app.domains.department_intro.state import DepartmentIntroState
from app.domains.department_intro.node import *

workflow = StateGraph(DepartmentIntroState)
workflow.add_node("extract_department", extract_department)
workflow.add_node("not_supported_department", not_supported_department)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("extract_department")

workflow.add_conditional_edges(
    "extract_department",
    route_by_department_result,
    {
        "valid": "retrieve",
        "not_supported": "not_supported_department",
        "not_specific": "retrieve"
    }
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "transform_query": "transform_query"
    }
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "hallucination": "generate",
        "relevant": END,
        "not relevant": "transform_query"
    }
)
department_intro_app = workflow.compile()