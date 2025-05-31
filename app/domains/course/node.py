from app.domains.course.state import CourseState
from app.vectorstore.qdrant import similarity_search
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# ✅ 1. 학과 추출
class DepartmentExtracted(BaseModel):
    department: str

def extract_department(state: CourseState) -> CourseState:
    
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

# ✅ 2. 검색
def retrieve(state: CourseState) -> CourseState:
    logger.info("[NODE] retrieve 진입")
    filters = {"metadata.department": state["department"]} if state["department"] else None
    logger.info(f"[INPUT] filters: {filters}")
    
    hits = similarity_search(state["question"], domain="course", k=5, metadata_filters=filters)
    docs = [doc["text"] for doc in hits]
    
    logger.info(f"[OUTPUT] {len(docs)} documents retrieved")
    return {**state, "documents": docs}

# ✅ 3. 문서 평가
class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def grade_documents(state: CourseState) -> CourseState:
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

# ✅ 4. 결정
def decide_to_generate(state: CourseState) -> str:
    logger.info("[NODE] decide_to_generate 진입")
    if not state["documents"]:
        logger.info("[DECISION] No relevant documents → transform_query")
        return "transform_query"
    logger.info("[DECISION] Relevant documents found → generate")
    return "generate"

# ✅ 5. 생성
def generate(state: CourseState) -> CourseState:    
    logger.info("[NODE] generate 진입")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서를 참고하여 질문에 답변을 생성하세요.\n 만약, 문서 내에서 특정 과목에 대한 내용을 참고하여 답변을 생성한다면, 과목 코드를 참고하여 해당 과목이 몇 학년 때 수강하기를 권장하는 지에 대한 정보도 함께 제공하세요. 과목 코드는 영어 알파벳 3~4글자 + 숫자 3~4글자로 구성되며, 맨 처음 숫자가 해당 과목의 권장 수강 학년입니다. "),
        ("human", "문서들: {documents}\n\n질문: {question}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "documents": "\n\n".join(state["documents"]),
        "question": state["question"]
    })
    logger.info(f"[OUTPUT] Generation (first 200 chars): {response.content[:200]}")
    return {**state, "generation": response.content}

# ✅ 6. 환각/관련성 평가
class GenEval(BaseModel):
    binary_score: str

def grade_generation_v_documents_and_question(state: CourseState) -> str:
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

# ✅ 7. 쿼리 재작성
class Rewritten(BaseModel):
    question: str

def transform_query(state: CourseState) -> CourseState:
    logger.info("[NODE] transform_query 진입")
    prompt = ChatPromptTemplate.from_messages([
        ("system",  "질문을 더 명확하게 한국어로 재작성해주세요."),
        ("human", "{question}")
    ])
    chain = prompt | llm.with_structured_output(Rewritten)
    better_question = chain.invoke({"question": state["question"]})
    
    logger.info(f"[OUTPUT] transformed question: {better_question.question}")
    return {**state, "question": better_question.question}
