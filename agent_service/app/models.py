from typing import List, Optional
from pydantic import BaseModel

class QueryRequest(BaseModel):
    agent_type: str
    user_query: str

class QueryResponse(BaseModel):
    response: str
    model_used: Optional[str] = ""              # or just str if always provided
    tools_used: List[str] = []
