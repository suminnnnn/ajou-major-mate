from typing import Annotated
from typing import TypedDict

class MessageState(TypedDict):
    question: Annotated[str, "User Question"]
    generation: Annotated[str, "LLM Generation"]
    inappropriate: Annotated[bool, "Result Of Filtering"]
    domain: Annotated[str, "Routed Domain"]