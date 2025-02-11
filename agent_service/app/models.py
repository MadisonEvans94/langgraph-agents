
from pydantic import BaseModel

class QueryRequest(BaseModel):
    agent_type: str
    user_query: str

class QueryResponse(BaseModel):
    response: str