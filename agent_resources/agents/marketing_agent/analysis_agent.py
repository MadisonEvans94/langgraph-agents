import logging
from typing import Dict, List, Optional

from agent_resources.base_agent import Agent
from agent_resources.state_types import AnalysisState
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from agent_resources.prompts import ANALYSIS_AGENT_PROMPT
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain

logger = logging.getLogger(__name__)

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
            START → extract_pdf → summarise → END
        """

        # (1) node: extract & split PDF locally
        async def _extract_pdf(state: AnalysisState) -> dict:
            path = state["path"]
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            if path.lower().endswith(".pdf"):
                docs = PyPDFLoader(path).load()
            else:
                docs = TextLoader(path).load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            chunks: List[Document] = splitter.split_documents(docs)
            return {"chunks": chunks}

        # (2) node: summarise the chunks with an LLM
        llm = self.llm_dict["default_llm"]

        def _summarise(state: AnalysisState) -> dict:
            chunks: List[Document] = state["chunks"]

            # Build prompts that include the "{text}" placeholder
            map_prompt = ChatPromptTemplate.from_messages([
                ("system", ANALYSIS_AGENT_PROMPT),
                ("user", "{text}"),
            ])
            combine_prompt = ChatPromptTemplate.from_messages([
                ("system", ANALYSIS_AGENT_PROMPT),
                ("user", "{text}"),
            ])

            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
            )
            summary = chain.run(chunks).strip()
            return {"summary": summary}

        sg = StateGraph(AnalysisState)
        sg.add_node("extract_pdf", _extract_pdf)
        sg.add_node("summarise", _summarise)
        sg.add_edge(START, "extract_pdf")
        sg.add_edge("extract_pdf", "summarise")
        sg.add_edge("summarise", END)
        return sg.compile()

    async def ainvoke(self, path: str, messages=None):
        """
        Run the graph and return the final state (which includes 'summary').
        """
        init_state = {"path": path, "messages": messages or []}
        return await self.runner.ainvoke(init_state)