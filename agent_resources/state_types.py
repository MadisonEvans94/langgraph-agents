# agent_resources/state_types.py
from typing import List, Literal, TypedDict, Optional
from typing_extensions import Annotated
from langgraph.graph.message import MessagesState
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class Task(TypedDict):
    """A single unit of work in the orchestrator."""
    id: str
    description: str
    assigned_to: str       # e.g. "math_agent" or "web_search_agent"
    status: str            # "pending" | "in_progress" | "done" | "error"
    result: Optional[str]  # filled in once the sub-agent returns

class OrchestratorState(TypedDict):
    """
    Custom state schema for the orchestrator:
      - inherits the default message‐loop fields
      - adds a `tasks` list of Task objects
    """
    messages: Annotated[List[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    tasks: List[Task]