from typing import List, Optional
from pydantic import BaseModel

class QueryRequest(BaseModel):
    user_query: str
    thread_id: Optional[str] = None
    agent_type: Optional[str] = None  # e.g., "mcp_agent"
    use_mcp: Optional[bool] = None    # defaults to True if not provided

class QueryResponse(BaseModel):
    response: str
    thread_id: str
    agent_type: str