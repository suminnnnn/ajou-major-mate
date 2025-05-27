from app.domains.curriculum.state import CurriculumState
from app.vectorstore.qdrant import similarity_search
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

class DepartmentExtracted(BaseModel):
    department: str

def extract_department(state: CurriculumState) -> CurriculumState:
    logger.info("[NODE] extract_department 진입")
    logger.info(f"[INPUT] question: {state['question']}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "question을 분석했을 때, 학과 리스트에 있는 특정 학과에 대한 질문이라면 해당 학과명을 추출하고, 아니라면 빈 문자열을 반환하세요.\n"
                   "학과 리스트: 소프트웨어학과, 디지털미디어학과, 국방디지털융합학과, 인공지능융합학과, 사이버보안학과"),
        ("human", "{question}")
    ])
    chain = prompt | llm.with_structured_output(DepartmentExtracted)
    result = chain.invoke({"question": state["question"]})

    logger.info(f"[OUTPUT] department: {result.department}")
    return {**state, "department": result.department}

def retrieve(state: CurriculumState) -> CurriculumState:
    logger.info("[NODE] retrieve 진입")
    filters = {"metadata.department": state["department"]} if state["department"] else None
    logger.info(f"[INPUT] filters: {filters}")

    hits = similarity_search(state["question"], domain="curriculum", k=5, metadata_filters=filters)
    docs = [doc["text"] for doc in hits]

    logger.info(f"[OUTPUT] {len(docs)} documents retrieved")
    return {**state, "documents": docs}

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_documents(state: CurriculumState) -> CurriculumState:
    logger.info("[NODE] grade_documents 진입")
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question."
                   " If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant."),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])

    retrieval_grader = prompt | structured_llm_grader
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

def decide_to_generate(state: CurriculumState) -> str:
    logger.info("[NODE] decide_to_generate 진입")
    if not state["documents"]:
        logger.info("[DECISION] No relevant documents → transform_query")
        return "transform_query"
    logger.info("[DECISION] Relevant documents found → generate")
    return "generate"

def generate(state: CurriculumState) -> CurriculumState:
    logger.info("[NODE] generate 진입")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서를 참고하여 질문에 답변을 생성하세요."),
        ("human", "문서들: {documents}\n\n질문: {question}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "documents": "\n\n".join(state["documents"]),
        "question": state["question"]
    })
    logger.info(f"[OUTPUT] Generation (first 200 chars): {response.content[:200]}")
    return {**state, "generation": response.content}

class GenEval(BaseModel):
    binary_score: str

def grade_generation_v_documents_and_question(state: CurriculumState) -> str:
    logger.info("[NODE] grade_generation_v_documents_and_question 진입")
    gen = state["generation"]
    docs = state["documents"]
    question = state["question"]

    doc_prompt = ChatPromptTemplate.from_messages([
        ("system", "응답이 문서를 기반으로 작성되었는지 yes/no로 평가해주세요."),
        ("human", "응답: {generation}\n\n문서: {documents}")
    ])
    doc_chain = doc_prompt | llm.with_structured_output(GenEval)
    doc_check = doc_chain.invoke({"generation": gen, "documents": "\n\n".join(docs)})
    logger.info(f"[EVAL] groundedness → {doc_check.binary_score}")
    if doc_check.binary_score != "yes":
        logger.info("[DECISION] hallucination detected → regenerate")
        return "hallucination"

    q_prompt = ChatPromptTemplate.from_messages([
        ("system", "응답이 질문에 적절한 답변인지 평가해주세요. yes 또는 no로 답해주세요."),
        ("human", "질문: {question}\n응답: {generation}")
    ])
    q_chain = q_prompt | llm.with_structured_output(GenEval)
    q_check = q_chain.invoke({"question": question, "generation": gen})
    logger.info(f"[EVAL] relevance to question → {q_check.binary_score}")

    return "relevant" if q_check.binary_score == "yes" else "not relevant"

class Rewritten(BaseModel):
    question: str

def transform_query(state: CurriculumState) -> CurriculumState:
    logger.info("[NODE] transform_query 진입")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "질문을 더 명확하게 한국어로 재작성해주세요."),
        ("human", "{question}")
    ])
    chain = prompt | llm.with_structured_output(Rewritten)
    better_question = chain.invoke({"question": state["question"]})
    logger.info(f"[OUTPUT] transformed question: {better_question.question}")
    return {**state, "question": better_question.question}