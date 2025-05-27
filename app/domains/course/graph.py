from langgraph.graph import END, StateGraph
from app.domains.course.state import CourseState
from app.domains.course.node import *

workflow = StateGraph(CourseState)

workflow.add_node("extract_department", extract_department)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("extract_department")
workflow.add_edge("extract_department", "retrieve")
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

course_app = workflow.compile()