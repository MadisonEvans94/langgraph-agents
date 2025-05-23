import json
import logging
from pathlib import Path
from typing import Dict

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chains.summarize import load_summarize_chain

from agent_resources.prompts import ANALYSIS_AGENT_PROMPT, SUMMARY_PROMPT, KEYPOINTS_PROMPT, DOMAIN_PROMPT

logger = logging.getLogger(__name__)

_LOADER_BY_EXT = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
}

_MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANALYSIS_AGENT_PROMPT),
    ("user", "{text}"),
])

async def extract_pdf_node(state: Dict) -> Dict:
    path = Path(state["path"])
    loader = _LOADER_BY_EXT.get(path.suffix.lower(), TextLoader)
    docs = loader(str(path)).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200
    ).split_documents(docs)
    logger.debug("Extracted %d chunks from %s", len(chunks), path)
    return {"chunks": chunks}

def summarise_node(state: Dict, llm) -> Dict:
    # Combine all chunk text into a single string
    text = "\n\n".join(doc.page_content for doc in state["chunks"])
    messages = [
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=text),
    ]
    response = llm.invoke(messages)
    summary = response.content.strip()
    logger.debug("Summary length: %d", len(summary))
    return {"summary": summary}

def extract_key_points_node(state: Dict, llm) -> Dict:
    messages = [
        SystemMessage(content=KEYPOINTS_PROMPT),
        HumanMessage(content=state.get("summary", "")),
    ]
    response = llm.invoke(messages)
    try:
        key_points = json.loads(response.content)
    except Exception:
        key_points = [
            line.strip(" -•") for line in response.content.splitlines()
            if line.strip()
        ]
    return {"key_points": key_points}

def detect_domain_node(state: Dict, llm) -> Dict:
    messages = [
        SystemMessage(content=DOMAIN_PROMPT),
        HumanMessage(content=state.get("summary", "")),
    ]
    response = llm.invoke(messages)
    domain = response.content.strip().strip('"')
    return {"domain": domain}