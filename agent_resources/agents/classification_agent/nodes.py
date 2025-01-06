# nodes.py
from typing import TypedDict, List
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Import the prompt constants from your prompts module
from agent_resources.prompts import (
    CLASSIFICATION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    SUMMARIZATION_PROMPT
)

load_dotenv()


class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def classification_node(state: State):
    """Classify the text into one of the categories: News, Blog, Research, or Other."""
    classification_template = PromptTemplate(
        input_variables=["text"],
        template=CLASSIFICATION_PROMPT
    )
    message = HumanMessage(
        content=classification_template.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}


def entity_extraction_node(state: State):
    """Extract all the entities (Person, Organization, Location) from the text."""
    entity_extraction_template = PromptTemplate(
        input_variables=["text"],
        template=ENTITY_EXTRACTION_PROMPT
    )
    message = HumanMessage(
        content=entity_extraction_template.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}


def summarization_node(state: State):
    """Summarize the text in one short sentence."""
    summarization_template = PromptTemplate(
        input_variables=["text"],
        template=SUMMARIZATION_PROMPT
    )
    message = HumanMessage(
        content=summarization_template.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}
