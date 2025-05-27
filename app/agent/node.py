from app.agent.state import MessageState
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

class QueryFilterOutput(BaseModel):
    inappropriate: bool = Field(..., description="질문이 부적절하거나 편향적인 경우 True, 아니면 False")


def query_filter(state: MessageState) -> MessageState:
    question = state["question"]
    
    logger.info("[NODE] query_filter 진입")
    logger.info(f"[INPUT] question: {question}")

    structured_llm = llm.with_structured_output(QueryFilterOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 질문이 부적절하거나 사회적으로 민감한 내용을 포함하는지 판단하는 필터입니다."),
        ("human", "다음 질문이 부적절하거나 편향적인가요? 판단해주세요.\n\n질문: {question}")
    ])
    chain = prompt | structured_llm

    result = chain.invoke({"question": question})
    logger.info(f"[OUTPUT] inappropriate: {result.inappropriate}")
    
    if(result.inappropriate):
        return {
            **state,
            "inappropriate": result.inappropriate,
            "generation": "죄송합니다. 해당 질문은 서비스 정책에 따라 답변드릴 수 없습니다. 다른 질문을 해주세요."
        }

    return {**state, "inappropriate": result.inappropriate}


class RouteQuery(BaseModel):
    """A domain to categorize the user question"""
    
    domain: str = Field(
        description="route the user question among 5 domains, 'course', 'curriculum', 'department_intro', 'employment_status', 'other' "
    )


def route_query(state: MessageState) -> MessageState:
    
    logger.info("[NODE] route_query 진입")
    logger.info(f"[INPUT] question: {state['question']}")
        
    structured_query_router = llm.with_structured_output(RouteQuery)
    
    system = """
    너는 유저의 질문을 아주대학교 관련 정보의 다섯 가지 도메인 중 하나로 분류하는 분류기 역할을 한다.
    아래 도메인 중 유저의 질문에 가장 적합한 하나를 골라야 한다:

    - course: 학과별 개설 과목, 과목명 등 수업 정보
    - curriculum: 학과별 졸업 요건, 학과별 학년별 커리큘럼, 학과별 권장이수
    - department_intro: 학과 소개, 교수진, 학과 사무실 전화 번호, 학과 사무실 위치, 학과 교육 목표
    - employment_status: 취업 현황, 진로, 진출 분야
    - other: 위 분류에 해당하지 않을 경우

    오직 하나의 도메인만 선택해서 응답하라.
    """
    
    query_router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
    )
    
    query_router = query_router_prompt | structured_query_router
    
    result = query_router.invoke({"question" : state["question"]})
    
    logger.info(f"[OUTPUT] domain: {result.domain}")
    
    if(result.domain=="other"):
        return {
            **state,
            "domain": result.domain,
            "generation": "해당 질문은 현재 제공 중인 학사 정보 범위에 포함되지 않습니다. 다른 질문을 해보세요."
        }

    return {**state, "domain": result.domain}
    
    
def decision(state: MessageState) -> str:
    domain = state.get("domain", "other")
    logger.info("[NODE] decision 진입")
    logger.info(f"[DECISION] selected domain: {domain}")
    return domain
