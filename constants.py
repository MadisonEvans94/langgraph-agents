

CONNECTION_ARGS = {
    "host": "localhost",  # Replace 'localhost' with your actual host if different
    "port": "19530"       # Default port for Milvus
}

# COLLECTION_NAME = "qna_collection"  # Replace with your desired collection name
COLLECTION_NAME = "clapnq_collection"

VECTOR_DIM = 1536  # Example dimension size for embeddings
PARTITION_TAG = "default_partition"  # Optional: Specify a partition tag

TOP_K = 5
EXIT_COMMAND = 'exit'

SOURCE_DIR = "./SOURCE_DOCUMENTS"

BATCH_SIZE = 100

EMBEDDING_MODEL="text-embedding-ada-002"