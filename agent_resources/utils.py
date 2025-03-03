import logging
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.messages.utils import convert_to_openai_messages
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ChatVLLMWrapper:
    """
    Wraps a chat completion endpoint using OpenAI/vLLM.
    Accepts a list of BaseMessage objects and returns an AIMessage.
    Supports tool binding via the bind_tools() method.
    """

    def __init__(
        self,
        client,
        model: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ):
        self.client = client
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        # No tools bound by default.
        self.bound_tools = None

        logger.info(
            f"ChatVLLMWrapper initialized with model: {self.model}, "
            f"max_new_tokens: {self.max_new_tokens}, temperature: {self.temperature}, "
            f"top_p: {self.top_p}, repetition_penalty: {self.repetition_penalty}"
        )

    def bind_tools(self, tools: list) -> "ChatVLLMWrapper":
        """
        Bind a list of tools to the model.
        Each tool should have a 'name' and a 'description' attribute
        (or a default fallback). Tools can also optionally define
        a Pydantic 'args_schema' for function calling.
        Returns a new ChatVLLMWrapper instance with the tools bound.
        """
        new_instance = ChatVLLMWrapper(
            client=self.client,
            model=self.model,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )
        new_instance.bound_tools = tools
        logger.info(f"Bound {len(tools)} tools to the model.")
        return new_instance

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Accepts a list of messages, calls the chat completion endpoint,
        and returns an AIMessage containing the new assistant message.

        If tools are bound, includes their schemas under the 'functions' parameter
        in the OpenAI "function calling" format. 
        """
        try:
            openai_messages = convert_to_openai_messages(messages)
            
            params = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            
            # If tools have been bound, convert each tool to a JSON-serializable "function".
            # This is optional, but demonstrates how to do "function calling" style usage.
            if self.bound_tools:
                functions = []
                for tool in self.bound_tools:
                    # 1) Start with name + description
                    func_def = {
                        "name": tool.name,
                        "description": getattr(tool, "description", "No description provided"),
                        # Default minimal "parameters" block
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    }
                    
                    # 2) If the tool defines a Pydantic args_schema, build a real schema
                    #    so the model can do function calling properly.
                    if hasattr(tool, "args_schema") and isinstance(tool.args_schema, type) \
                       and issubclass(tool.args_schema, BaseModel):
                        pydantic_schema = tool.args_schema.schema()
                        # Make sure top-level says "type": "object"
                        pydantic_schema["type"] = "object"
                        func_def["parameters"] = pydantic_schema

                    # Append this JSON-safe function definition
                    functions.append(func_def)

                # Attach them to the params
                params["functions"] = functions

            logger.info(f"Invoking vLLM with parameters: {params}")

            # Make the request to your vLLM/OpenAI-compatible endpoint
            response = self.client.chat.completions.create(**params)
            logger.info(f"Received response from vLLM: {response}")

            # Extract assistant content
            content = response.choices[0].message.content.strip()
            
            return AIMessage(content=content)

        except Exception as e:
            logger.error(f"Error invoking vLLM client: {e}", exc_info=True)
            raise RuntimeError(f"vLLM invocation failed: {e}")
