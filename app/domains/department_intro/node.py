from app.domains.department_intro.state import DepartmentIntroState
from app.vectorstore.qdrant import similarity_search
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

class DepartmentExtracted(BaseModel):
    department: str

def extract_department(state: DepartmentIntroState) -> DepartmentIntroState:
    logger.info("[NODE] extract_department 진입")
    logger.info(f"[INPUT] question: {state['question']}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "질문에서 학과명을 추출하세요.\n학과 리스트: 소프트웨어학과, 디지털미디어학과, 국방디지털융합학과, 인공지능융합학과, 사이버보안학과"),
        ("human", "{question}")
    ])
    chain = prompt | llm.with_structured_output(DepartmentExtracted)
    result = chain.invoke({"question": state["question"]})
    logger.info(f"[OUTPUT] department: {result.department}")
    return {**state, "department": result.department}

def retrieve(state: DepartmentIntroState) -> DepartmentIntroState:
    logger.info("[NODE] retrieve 진입")
    filters = {"metadata.department": state["department"]} if state["department"] else None
    logger.info(f"[INPUT] filters: {filters}")
    hits = similarity_search(state["question"], domain="department_intro", k=2, metadata_filters=filters)
    docs = [doc["text"] for doc in hits]
    logger.info(f"[OUTPUT] {len(docs)} documents retrieved")
    return {**state, "documents": docs}

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_documents(state: DepartmentIntroState) -> DepartmentIntroState:
    logger.info("[NODE] grade_documents 진입")
    chain = ChatPromptTemplate.from_messages([
        ("system", "문서가 질문과 관련 있는지 평가하고 'yes' 또는 'no'로 답하세요."),
        ("human", "문서: {document}\n질문: {question}")
    ]) | llm.with_structured_output(GradeDocuments)
    filtered = [doc for doc in state["documents"] if chain.invoke({"question": state["question"], "document": doc}).binary_score == "yes"]
    logger.info(f"[OUTPUT] {len(filtered)} documents passed relevance filter")
    return {**state, "documents": filtered}

def decide_to_generate(state: DepartmentIntroState) -> str:
    logger.info("[NODE] decide_to_generate 진입")
    return "transform_query" if not state["documents"] else "generate"

def generate(state: DepartmentIntroState) -> DepartmentIntroState:
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

def grade_generation_v_documents_and_question(state: DepartmentIntroState) -> str:
    logger.info("[NODE] grade_generation_v_documents_and_question 진입")
    doc_eval = ChatPromptTemplate.from_messages([
        ("system", "응답이 문서를 기반으로 작성되었는지 평가하세요."),
        ("human", "응답: {generation}\n문서: {documents}")
    ]) | llm.with_structured_output(GenEval)
    if doc_eval.invoke({"generation": state["generation"], "documents": "\n".join(state["documents"])}).binary_score != "yes":
        return "hallucination"
    q_eval = ChatPromptTemplate.from_messages([
        ("system", "응답이 질문에 적절한지 평가하세요."),
        ("human", "질문: {question}\n응답: {generation}")
    ]) | llm.with_structured_output(GenEval)
    return "relevant" if q_eval.invoke({"question": state["question"], "generation": state["generation"]}).binary_score == "yes" else "not relevant"

class Rewritten(BaseModel):
    question: str

def transform_query(state: DepartmentIntroState) -> DepartmentIntroState:
    logger.info("[NODE] transform_query 진입")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "질문을 더 명확하게 한국어로 재작성해주세요."),
        ("human", "{question}")
    ])
    better = (prompt | llm.with_structured_output(Rewritten)).invoke({"question": state["question"]})
    logger.info(f"[OUTPUT] transformed question: {better.question}")
    return {**state, "question": better.question}