import logging
from typing import Iterator, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.messages.utils import convert_to_openai_messages
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ChatVLLMWrapper:
    """
    Wraps a chat completion endpoint using OpenAI/vLLM.

    - If `streaming=False`, invoke() returns a single AIMessage with the final text.
    - If `streaming=True`, invoke() returns an iterator that yields AIMessageChunk objects 
      as partial tokens arrive.
    """

    def __init__(
        self,
        client,
        model: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        streaming: bool = False,
    ):
        self.client = client
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.streaming = streaming
        self.bound_tools = None

        logger.info(
            f"🚀 ChatVLLMWrapper initialized | model={model}, streaming={streaming}, "
            f"max_new_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}"
        )

    def bind_tools(self, tools: list) -> "ChatVLLMWrapper":
        new_instance = ChatVLLMWrapper(
            client=self.client,
            model=self.model,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            streaming=self.streaming,  # carry over the streaming flag
        )
        new_instance.bound_tools = tools
        logger.info(f"🔧 Bound {len(tools)} tools to the model.")
        return new_instance

    def invoke(
        self, messages: list[BaseMessage]
    ) -> Union[AIMessage, Iterator[AIMessageChunk]]:
        """
        If streaming=False:
            returns a single final AIMessage (the entire response).
        If streaming=True:
            returns a generator that yields AIMessageChunk objects 
            (the partial tokens) as they arrive.
        """
        openai_messages = convert_to_openai_messages(messages)
        params = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.streaming,  # enables streaming mode if True
        }

        if self.bound_tools:
            params["functions"] = self._convert_tools_to_functions()

        if not self.streaming:
            logger.info(f"🟢 [ChatVLLMWrapper] Invoking (non-stream) with parameters: {params}")
            try:
                response = self.client.chat.completions.create(**params)
                logger.info(f"✅ Received non-stream response: {response}")
                content = response.choices[0].message.content
                logger.info(f"✅ Extracted content (first 50 chars): {content[:50]!r}")
                return AIMessage(content=content.strip() if content else "")
            except Exception as e:
                logger.error(f"❌ Error during non-stream invocation: {e}", exc_info=True)
                raise
        else:
            logger.info(f"🟡 [ChatVLLMWrapper] Invoking (stream) with parameters: {params}")
            return self._stream_generator(params)

    def _stream_generator(
        self, params: dict
    ) -> Iterator[AIMessageChunk]:
        """
        Internal helper: calls the vLLM-like endpoint with streaming,
        then yields `AIMessageChunk(content=...)` for each partial token received.
        """
        try:
            logger.info("🔄 Starting streaming call to vLLM endpoint...")
            stream_resp = self.client.chat.completions.create(**params)
            logger.info("🔄 Streaming response object received.")

            for i, chunk in enumerate(stream_resp):
                logger.debug(f"Chunk {i}: {chunk}")
                if not chunk.choices:
                    logger.warning(f"Chunk {i} has no choices; skipping.")
                    continue

                choice = chunk.choices[0]
                delta_obj = choice.get("delta", {})

                if "content" in delta_obj:
                    partial_text = delta_obj["content"]
                    logger.info(f"📌 Received streaming chunk {i}: {partial_text!r}")
                    yield AIMessageChunk(content=partial_text)

                finish_reason = choice.get("finish_reason", None)
                if finish_reason in ("stop", "finished"):
                    logger.info(f"✅ Finish reason received in chunk {i}: {finish_reason}. Ending stream.")
                    break

        except Exception as e:
            logger.error(f"❌ Error during streaming: {e}", exc_info=True)
            raise

    def _convert_tools_to_functions(self):
        """
        Utility to convert bound_tools -> JSON for function calling
        (if you are implementing the new function calling style).
        """
        functions = []
        for tool in self.bound_tools or []:
            func_def = {
                "name": tool.name,
                "description": getattr(tool, "description", ""),
                "parameters": {"type": "object", "properties": {}},
            }
            if hasattr(tool, "args_schema") and issubclass(tool.args_schema, BaseModel):
                schema = tool.args_schema.model_json_schema()
                schema["type"] = "object"
                func_def["parameters"] = schema
            functions.append(func_def)
        logger.info(f"Converted {len(functions)} tools to function definitions.")
        return functions

def make_llm(config: dict, use_llm_provider: bool) -> ChatOpenAI:
    model_id = config.get("model_id") or config.get("model")
    if model_id is None:
        raise ValueError("LLM config must include 'model_id' or 'model'.")
    temperature = config.get("temperature", 0.7)
    max_new_tokens = config.get("max_new_tokens") or config.get("max_tokens", 512)
    api_key = config.get("api_key", "")
    base_url = config.get("base_url")
    params = dict(
        model=model_id,
        temperature=temperature,
        max_tokens=max_new_tokens,
        timeout=None,
        max_retries=2,
        streaming=True,
        api_key=api_key or "EMPTY",
    )
    if not use_llm_provider:
        if not base_url:
            raise ValueError(f"When using vLLM, 'base_url' is required for model {model_id}.")
        params["api_base"] = base_url
    return ChatOpenAI(**params)
