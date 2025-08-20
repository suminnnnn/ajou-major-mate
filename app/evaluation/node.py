from app.evaluation.evaluator import evaluate_response_with_llm
from app.agent.state import MessageState

def evaluate_answer(state: MessageState) -> tuple[MessageState, dict]:
    query = state.get("question")
    answer = state.get("generation")
    context_docs = state.get("documents", [])

    evaluation_result = evaluate_response_with_llm(query, answer, context_docs)
    
    print(evaluation_result)

    return state