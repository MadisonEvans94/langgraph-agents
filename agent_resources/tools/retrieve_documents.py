from langchain.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieveDocuments(BaseTool):
    """
    A tool to retrieve documents to answer user questions.
    Uses a vector store to return relevant documents.
    """

    name: str = "retrieve_documents"
    description: str = (
        "Retrieves relevant documents from vector store to answer questions"
    )
    embeddings: Embeddings
    vector_store: VectorStore

    def _run(self, query: str) -> str:
        """
        Perform the document retrieval and serialize the result to JSON.
        """
        try:
            # Perform similarity search
            documents = self.vector_store.similarity_search(query, k=3)

            # Serialize to JSON format
            serialized_docs = json.dumps([
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in documents
            ])

            return serialized_docs

        except Exception as e:
            logger.error(
                "Error occurred during document retrieval", exc_info=True)
            # Return an empty JSON array to prevent callback errors
            return json.dumps([])

    async def _arun(self, query: str) -> str:
        """
        Asynchronous version of document retrieval.
        """
        return self._run(query)
