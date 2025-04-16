
REACT_AGENT_SYSTEM_PROMPT = """\
You have access to the following tool(s):

{tools_section}

Guidelines for Answering User Queries:
- If the query is a direct question, invoke the relevant tool(s) to gather information.

Key Reminders:
- Prioritize precision and completeness in your responses. 

"""

TASK_DECOMPOSITION_PROMPT = """
You are an intelligent assistant. Break down the following user question into 
a few high-level sub-questions that are necessary to provide a comprehensive and concise answer.

User question: "{question}"

Guidelines:
1. Limit the decomposition to **2 or 3** sub-questions.
2. Ensure each sub-question addresses a distinct aspect of the main question without overlapping.
3. Maintain a high-level overview without delving into excessive detail.

Provide the sub-queries in a JSON array format, for example:
{{
    "sub_queries": [
        "Sub-question 1",
        "Sub-question 2"
    ]
}}
"""
