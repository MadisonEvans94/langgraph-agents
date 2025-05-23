# agent_resources/state_types.py

from typing import Any, List, Dict, Optional
from typing_extensions import Annotated, TypedDict
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.messages import AnyMessage
from langchain.schema import Document


class Task(TypedDict):
    """A single unit of work in the orchestrator."""
    id: int
    description: str
    assigned_to: str        # "math_agent" or "web_search_agent"
    status: str             # "pending" | "done" | "error"
    result: Optional[str]
    depends_on: List[int]   # prerequisite task IDs

class SupervisorState(TypedDict):
    """
    State schema for in-degree + adjacency-list scheduling.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps

    tasks: List[Task]
    in_degree: Dict[str, int]
    dependents: Dict[str, List[str]]
    ready_queue: List[int]
    current_task_id: Optional[int]
    next: Optional[str]          # name of the next node to execute

class MarketingAgentState(TypedDict, total=False):
    """
    State schema for the AnalysisAgent's marketing workflow.

    Fields:
        messages: Annotated list of messages (AnyMessage) persisted through the agent for context.
        path: Filesystem path to the input PDF or text document being analyzed.
        chunks: List of Document objects representing text chunks extracted from the document.
        summary: String containing the executive summary generated from all chunks.
        key_points: List of concise key point phrases extracted from the summary.
        domain: One- or two-word label classifying the document's primary domain (e.g., "technology").
    """
    messages: Annotated[List[AnyMessage], add_messages]
    path: str
    chunks: List[Document]
    summary: str
    key_points: List[str]
    domain: str

class ImageSearchAgentState(TypedDict, total=False):
    """
    State schema for the ImageSearchAgent.

    Fields:
        messages: Annotated list of messages (AnyMessage) persisted through the agent for context.
        query: The search query string provided by the user.
        images: List of image URLs returned by the image search tool.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    images: List[str]
