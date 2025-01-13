from langchain_community.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)
import os

MLV_HOST = os.getenv("MILVUS_HOST")
MLV_PORT = os.getenv("MILVUS_PORT")
MLV_URI = f"https://{MLV_HOST}:{MLV_PORT}"
SERVER_PEM_PATH = "/root/ca/milvus_certificate.pem"
CONNECTION_ARGS = {
    "uri": MLV_URI,
    "user": "root",
    "password": os.environ.get("MILVUS_PASSWORD"),
    "secure": True,
    "server_pem_path": SERVER_PEM_PATH,
    "server_name": "milvus",
}

# COLLECTION_NAME = "qna_collection"  # Replace with your desired collection name
COLLECTION_NAME = "test_collection"

VECTOR_DIM = 1536  # Example dimension size for embeddings
PARTITION_TAG = "default_partition"  # Optional: Specify a partition tag

TOP_K = 5
EXIT_COMMAND = 'exit'

SOURCE_DIR = "./ingestion/SOURCE_DOCUMENTS"

BATCH_SIZE = 100

EMBEDDING_MODEL="text-embedding-ada-002"

# Max number of workers for ingesting documents
INGEST_THREADS = int(os.getenv("INGEST_THREADS", 8))

# LangChain FileLoader for different file formats
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
