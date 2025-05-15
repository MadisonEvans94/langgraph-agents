from typing import List, Literal, Optional
from typing_extensions import Annotated, TypedDict
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class Task(TypedDict):
    """A single unit of work in the orchestrator."""
    id: str
    description: str
    assigned_to: str       # e.g. "math_agent" or "web_search_agent"
    status: str            # "pending" | "in_progress" | "done" | "error"
    result: Optional[str] 
    depends_on: Optional[List[str]] 

class SupervisorState(TypedDict):
    """
    Custom state schema for the orchestrator:
      - default message-loop fields
      - adds a `tasks` list of Task objects
    """
    messages: Annotated[List[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    tasks: List[Task]