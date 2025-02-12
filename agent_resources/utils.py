import logging
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
)
from langchain_core.messages.utils import convert_to_openai_messages

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ChatVLLMWrapper:
    """
    Wraps a chat completion endpoint using OpenAI/vLLM.
    Accepts a list of BaseMessage objects and returns an AIMessage.
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
        
        logger.info(f"ChatVLLMWrapper initialized with model: {self.model}, "
                    f"max_new_tokens: {self.max_new_tokens}, temperature: {self.temperature}, "
                    f"top_p: {self.top_p}, repetition_penalty: {self.repetition_penalty}")
        
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Accepts a list of messages. Calls chat completion endpoint.
        Returns an AIMessage containing the new assistant message.
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

            # Log request parameters for debugging
            logger.info(f"Invoking vLLM with parameters: {params}")
            
            response = self.client.chat.completions.create(**params)
            
            # Log full response for further debugging
            logger.info(f"Received response from vLLM: {response}")
            
            content = response.choices[0].message.content.strip()
            
            return AIMessage(content=content)

        except Exception as e:
            logger.error(f"Error invoking vLLM client: {e}", exc_info=True)
            raise RuntimeError(f"vLLM invocation failed: {e}")
