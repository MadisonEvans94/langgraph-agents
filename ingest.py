import os
import logging
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import Dataset

from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from pdfminer.high_level import extract_text

from constants import CONNECTION_ARGS, COLLECTION_NAME, BATCH_SIZE, SOURCE_DIR, EMBEDDING_MODEL

###############################################################################
# Config
###############################################################################



###############################################################################
# Setup Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment!")

###############################################################################
# Document Loading
###############################################################################

def load_documents(source_dir: str) -> List[Document]:
    documents = []
    for filename in tqdm(os.listdir(source_dir), desc="Loading documents"):
        file_path = os.path.join(source_dir, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(Document(page_content=text, metadata={"filename": filename}))
        elif filename.endswith(".pdf"):
            try:
                pdf_text = extract_text(file_path)
                documents.append(Document(page_content=pdf_text, metadata={"filename": filename}))
            except Exception as e:
                logging.error(f"Failed to extract text from {filename}: {e}")
        else:
            logging.warning(f"Unsupported file type: {filename}, skipping...")
    return documents

###############################################################################
# Ingestion Pipeline
###############################################################################

def ingest_documents():
    # Load documents from SOURCE_DIR
    logging.info("Loading documents from SOURCE_DIR...")
    raw_documents = load_documents(SOURCE_DIR)

    if not raw_documents:
        logging.warning("No documents loaded. Exiting.")
        return

    # Initialize embeddings and chunker
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL
    )

    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0
    )

    # Chunk documents
    logging.info("Splitting documents using SemanticChunker...")
    all_docs: List[Document] = []

    for doc in tqdm(raw_documents, desc="Chunking documents"):
        chunks = text_splitter.create_documents([doc.page_content])
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
        all_docs.extend(chunks)

    # Prepare Milvus vector store
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args=CONNECTION_ARGS,
        drop_old=True,
        auto_id=True
    )

    # Ingest documents into Milvus in batches
    logging.info(f"Total number of chunked Documents: {len(all_docs)}")
    for i in tqdm(range(0, len(all_docs), BATCH_SIZE), desc="Ingesting to Milvus"):
        batch = all_docs[i: i + BATCH_SIZE]
        vector_store.add_documents(batch)

    logging.info("Data ingestion complete!")

###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    ingest_documents()