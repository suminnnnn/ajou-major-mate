from app.domains.employment_status.state import EmploymentStatusState
from app.vectorstore.qdrant import similarity_search
from app.utils.document_formatter import format_documents
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

class DepartmentExtracted(BaseModel):
    result: str  # "valid", "not_supported", "not_specific"
    department: str = ""

def extract_department(state: EmploymentStatusState) -> EmploymentStatusState:
    logger.info("[NODE] extract_department 진입")
    logger.info(f"[INPUT] question: {state['question']}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "질문이 특정 학과에 대한 질문인지 판별하고, 학과 리스트에 존재하는지 확인하라.\n"
        "- 특정 학과에 대한 질문이고 학과 리스트에 있다면: result='valid', department='학과명'\n"
        "- 특정 학과에 대한 질문이지만 학과 리스트에 없다면: result='not_supported', department='질문에 포함된 학과명'\n"
        "- 특정 학과에 대한 질문이 아니면: result='not_specific', department=''\n"
        "학과 리스트: 소프트웨어학과, 디지털미디어학과, 국방디지털융합학과, 인공지능융합학과, 사이버보안학과"),
        ("human", "{question}")
    ])
    chain = prompt | llm.with_structured_output(DepartmentExtracted)
    result = chain.invoke({"question": state["question"]})

    logger.info(f"[OUTPUT] result: {result.result}, department: {result.department}")
    return {**state, "department": result.department, "department_result": result.result}

def route_by_department_result(state: EmploymentStatusState) -> str:
    return state["department_result"]

def not_supported_department(state: EmploymentStatusState) -> EmploymentStatusState:
    logger.info("[NODE] not_supported_department 진입")
    return {
        **state,
        "generation": "죄송합니다. 현재 아주대학교에는 해당 학과가 존재하지 않아 안내드릴 수 없습니다."
    }

def retrieve(state: EmploymentStatusState) -> EmploymentStatusState:
    logger.info("[NODE] retrieve 진입")
    
    filters = {"metadata.department": state["department"]} if state["department"] else None
    logger.info(f"[INPUT] filters: {filters}")
    
    hits = similarity_search(state["question"], domain="employment_status", k=2, metadata_filters=filters)
    
    formatted_docs = format_documents(hits)
    
    logger.info(f"[OUTPUT] {len(formatted_docs )} documents retrieved")
    return {**state, "documents": formatted_docs}

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_documents(state: EmploymentStatusState) -> EmploymentStatusState:
    logger.info("[NODE] grade_documents 진입")
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing whether a retrieved document is meaningfully relevant to a user question.\n
        Only respond 'yes' if the document contains concrete, informative content (not headings or placeholders) that can directly help answer the question.\n
        Do not mark as relevant if the document contains only general section titles or insufficient information.\n
        Your job is to filter out unhelpful or vague results, not to be lenient.\n
        Respond only with a binary score: 'yes' or 'no'."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}\n\n Retrieved document content:\n\n {document}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    question = state["question"]
    documents = state["documents"]

    filtered = []
    for i, doc in enumerate(documents):
        logger.info(f"[EVAL] Doc {i+1} 평가 중...")
        result = retrieval_grader.invoke({"question": question, "document": doc})
        logger.info(f"[RESULT] Doc {i+1}: {result.binary_score}")
        if result.binary_score == "yes":
            filtered.append(doc)

    logger.info(f"[OUTPUT] {len(filtered)} documents passed relevance filter")
    return {**state, "documents": filtered}

def decide_to_generate(state: EmploymentStatusState) -> str:
    logger.info("[NODE] decide_to_generate 진입")
    return "transform_query" if not state["documents"] else "generate"

def generate(state: EmploymentStatusState) -> EmploymentStatusState:
    logger.info("[NODE] generate 진입")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "문서를 참고하여 질문에 답하세요."),
        ("human", "문서: {documents}\n질문: {question}")
    ])
    response = (prompt | llm).invoke({"documents": "\n".join(state["documents"]), "question": state["question"]})
    logger.info(f"[OUTPUT] Generation: {response.content[:200]}")
    return {**state, "generation": response.content}

class GenEval(BaseModel):
    binary_score: str

def grade_generation_v_documents_and_question(state: EmploymentStatusState) -> str:
    logger.info("[NODE] grade_generation_v_documents_and_question 진입")
    gen = state["generation"]
    docs = state["documents"]
    question = state["question"]

    doc_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ])
    doc_chain = doc_prompt | llm.with_structured_output(GenEval)
    doc_check = doc_chain.invoke({"generation": gen, "documents": "\n\n".join(docs)})
    logger.info(f"[EVAL] groundedness → {doc_check.binary_score}")
    
    if doc_check.binary_score != "yes":
        logger.info("[DECISION] hallucination detected → regenerate")
        return "hallucination"
    
    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

    q_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    ])
    q_chain = q_prompt | llm.with_structured_output(GenEval)
    q_check = q_chain.invoke({"question": question, "generation": gen})
    logger.info(f"[EVAL] relevance to question → {q_check.binary_score}")

    return "relevant" if q_check.binary_score == "yes" else "not relevant"

class Rewritten(BaseModel):
    question: str

def transform_query(state: EmploymentStatusState) -> EmploymentStatusState:
    logger.info("[NODE] transform_query 진입")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "질문을 더 명확하게 한국어로 재작성해주세요."),
        ("human", "{question}")
    ])
    better = (prompt | llm.with_structured_output(Rewritten)).invoke({"question": state["question"]})
    logger.info(f"[OUTPUT] transformed question: {better.question}")
    return {**state, "question": better.question}
