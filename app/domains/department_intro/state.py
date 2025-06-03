from typing import List
from typing_extensions import TypedDict, Annotated

class DepartmentIntroState(TypedDict):
    question: Annotated[str, "User query"]
    generation: Annotated[str, "LLM-generated answer"]
    documents: Annotated[List[str], "Retrieved and filtered documents"]
    department: Annotated[List[str], "List of departments extracted from user query"]
    department_result: Annotated[str, "Result of department check"]