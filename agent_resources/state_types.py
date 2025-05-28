from typing import List
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class MarketingAgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    analysis: str
    image_url: str