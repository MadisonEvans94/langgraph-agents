import logging
from typing import Dict, List, Optional
from pathlib import Path
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from agent_resources.base_agent import Agent
from agent_resources.state_types import MarketingAgentState
from langgraph.graph import StateGraph, START, END
from agent_resources.prompts import SUMMARY_PROMPT, KEYPOINTS_PROMPT, DOMAIN_PROMPT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

_LOADER_BY_EXT = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
}

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

class AnalysisAgent(Agent):
    def __init__(
        self,
        llm_configs: Dict[str, dict],
        memory=None,
        thread_id: Optional[str] = None,
        tools=None,
        name: str = "analysis_agent",
        **kwargs,
    ):
        self.use_llm_provider = kwargs.get("use_llm_provider", False)
        self.name = name
        self.tools = tools or []
        self._build_llm_dict(llm_configs)
        self.memory = memory
        self.thread_id = thread_id or "default"
        self.state_graph = self.build_graph()
        self.runner = self.state_graph

    def build_graph(self):
        """
        Graph flow:
            START → extract_pdf → summarise
                ↘→ extract_key_points → END
                ↘→ detect_domain → END
        """
        llm = self.llm_dict["default_llm"]

        sg = StateGraph(MarketingAgentState)
        # Nodes
        sg.add_node("extract_pdf", extract_pdf_node)
        sg.add_node("summarise", lambda state: summarise_node(state, llm))
        sg.add_node("extract_key_points", lambda state: extract_key_points_node(state, llm))
        sg.add_node("detect_domain", lambda state: detect_domain_node(state, llm))

        # Edges
        sg.add_edge(START, "extract_pdf")
        sg.add_edge("extract_pdf", "summarise")

        # fan-out after summarise
        sg.add_edge("summarise", "extract_key_points")
        sg.add_edge("summarise", "detect_domain")

        # join to END
        sg.add_edge("extract_key_points", END)
        sg.add_edge("detect_domain", END)

        return sg.compile()

    async def ainvoke(self, path: str, messages=None):
        """
        Run the graph and return the final state (which includes 'summary',
        'key_points', and 'domain').
        """
        init_state = {"path": path, "messages": messages or []}
        return await self.runner.ainvoke(init_state)