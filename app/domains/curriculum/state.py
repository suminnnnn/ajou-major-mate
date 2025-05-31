from typing import List
from typing_extensions import TypedDict, Annotated

class CurriculumState(TypedDict):
    question: Annotated[str, "User query"]
    generation: Annotated[str, "LLM-generated answer"]
    documents: Annotated[List[str], "Retrieved and filtered documents"]
    department: Annotated[str, "Department extracted from user query"]
    department_result: Annotated[str, "Result of department check"]