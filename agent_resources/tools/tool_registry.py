from typing import Dict, Type, List
from langchain.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from .retrieve_documents import RetrieveDocuments  
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Import Chroma for vector store
from constants import COLLECTION_NAME, EMBEDDING_MODEL
from dotenv import load_dotenv
load_dotenv()
# TODO: Place embeddings and vector store in a separate config file
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Initialize Chroma vector store
chroma_vector_store = Chroma(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory="./chroma_db"  # Ensure this matches the directory used in your ingestion script
)

class ToolRegistry:
    """
    ToolRegistry manages the registration and retrieval of tools.
    """

    tool_registry: Dict[str, BaseTool] = {
        'tavily_search': TavilySearchResults(),
        'retrieve_documents': RetrieveDocuments(
            embeddings=embeddings,
            vector_store=chroma_vector_store
        )
    }

    @classmethod
    def get_tool(cls, tool_name: str, **kwargs) -> BaseTool:
        """
        Retrieve a single tool by name.
        """
        tool = cls.tool_registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        # If additional kwargs are provided, they are used to reinitialize the tool
        return tool if not kwargs else tool.__class__(**kwargs)

    @classmethod
    def get_tools(cls, tool_names: List[str], **kwargs) -> List[BaseTool]:
        """
        Retrieve multiple tools by their names.
        """
        return [cls.get_tool(name, **kwargs) for name in tool_names]