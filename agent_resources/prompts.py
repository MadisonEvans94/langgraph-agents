# prompts.py

RAG_AGENT_PROMPT = """
You have the following tool available:

1. retrieve_documents: Use this tool only for queries that directly ask a question.

When answering user queries:
If the query is a question, use the retrieve_documents tool to provide an answer. Be complete and thorough with your answers, including reasoning from the documents retrieved. The answer you provide should ONLY be derived from information provided in the retrieved document context, and nothing else.

If the retrieved documents do not provide a definitive answer, simply respond with an empty string.
"""

CLASSIFICATION_PROMPT = """
Classify the following text into one of the categories: News, Blog, Research, Documentation, Dialogue, or Other.

Text: {text}

Category:
"""

ENTITY_EXTRACTION_PROMPT = """
Extract all the entities (Person, Organization, Location) from the following text. 
Provide the result as a comma-separated list.

Text: {text}

Entities:
"""

SUMMARIZATION_PROMPT = """
Summarize the following text in one short sentence.

Text: {text}

Summary:
"""
