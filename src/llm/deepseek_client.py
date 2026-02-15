"""
DeepSeek API client for ASEA-X
Handles LLM communication with retries and error handling
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider = LLMProvider.DEEPSEEK
    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 300
    max_retries: int = 3


@dataclass
class LLMResponse:
    """LLM response wrapper"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    raw_response: Any = None
    cached: bool = False


# -----------------------------
# Retry configuration (Option A)
# -----------------------------
# Decorators are evaluated at import/class-definition time, so this MUST be defined
# at module scope (not as an instance attribute). [1](https://tenacity.readthedocs.io/en/stable/api.html)[2](https://tenacity.readthedocs.io/en/latest/)
_DEFAULT_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

retry_config = {
    "stop": stop_after_attempt(_DEFAULT_MAX_RETRIES),
    "wait": wait_exponential(multiplier=1, min=4, max=10),
    "retry": retry_if_exception_type((openai.APITimeoutError, openai.APIError)),
    "reraise": True,
}


class DeepSeekClient:
    """Client for DeepSeek API with caching and retries"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or self._load_config()
        self.client = self._initialize_client()
        self.logger = logging.getLogger(__name__)
        self.cache: Dict[str, LLMResponse] = {}

    def _load_config(self) -> LLMConfig:
        """Load configuration from environment"""
        return LLMConfig(
            provider=LLMProvider(os.getenv("LLM_PROVIDER", "deepseek")),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "4096")),
            timeout=int(os.getenv("AGENT_TIMEOUT", "300")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        )

    def _initialize_client(self) -> OpenAI:
        """Initialize the OpenAI client for DeepSeek"""
        if not self.config.api_key:
            raise ValueError("API key is required")

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    @retry(**retry_config)
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Get chat completion from DeepSeek

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tools/functions
            tool_choice: Tool choice strategy
            stream: Whether to stream response

        Returns:
            LLMResponse object
        """
        cache_key = self._create_cache_key(messages, tools, tool_choice)

        if cache_key in self.cache:
            response = self.cache[cache_key]
            response.cached = True
            self.logger.debug("Cache hit for LLM request")
            return response

        request_data = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }

        if tools:
            request_data["tools"] = tools
            if tool_choice:
                request_data["tool_choice"] = tool_choice

        try:
            start_time = time.time()

            if stream:
                return self._handle_streaming(request_data)

            response = self.client.chat.completions.create(**request_data)

            elapsed = time.time() - start_time
            self.logger.debug(f"LLM call completed in {elapsed:.2f}s")

            llm_response = self._parse_response(response)
            self.cache[cache_key] = llm_response
            return llm_response

        except openai.AuthenticationError as e:
            self.logger.error(f"Authentication failed: {e}")
            raise
        except openai.RateLimitError as e:
            # Note: leaving your explicit sleep to preserve original behavior
            self.logger.warning(f"Rate limited: {e}")
            time.sleep(5)
            raise
        except openai.APITimeoutError as e:
            self.logger.warning(f"API timeout: {e}")
            raise
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

    def _handle_streaming(self, request_data: Dict) -> LLMResponse:
        """Handle streaming response"""
        collected_content: List[str] = []

        response = self.client.chat.completions.create(**request_data)

        # Keep track of the last chunk so we can access ids/metadata after iteration
        last_chunk = None
        for chunk in response:
            last_chunk = chunk
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                content = delta.content
                collected_content.append(content)
                if hasattr(self, "stream_callback"):
                    self.stream_callback(content)

        # If the stream produced no chunks for some reason, return empty
        if last_chunk is None:
            return LLMResponse(
                content="",
                usage={},
                model=self.config.model,
                finish_reason="stop",
                raw_response=None,
            )

        # Construct a minimal ChatCompletion-like object for _parse_response
        full_response = openai.types.chat.ChatCompletion(
            id=last_chunk.id,
            choices=[
                openai.types.chat.Choice(
                    index=0,
                    message=openai.types.chat.ChatCompletionMessage(
                        content="".join(collected_content),
                        role="assistant",
                    ),
                    finish_reason=last_chunk.choices[0].finish_reason,
                )
            ],
            created=last_chunk.created,
            model=last_chunk.model,
            object="chat.completion",
        )

        return self._parse_response(full_response)

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse API response into LLMResponse"""
        choice = response.choices[0]
        message = choice.message

        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content or "",
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            raw_response=response,
        )

    def _create_cache_key(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> str:
        """Create cache key from request parameters"""
        import hashlib

        key_data = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "model": self.config.model,
            "temperature": self.config.temperature,
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear response cache"""
        self.cache.clear()

    def get_models(self) -> List[str]:
        """Get available models from provider"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            self.logger.error(f"Failed to fetch models: {e}")
            return []

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.config.model)
            return len(encoding.encode(text))
        except Exception:
            return len(text.split())

    def truncate_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 8000,
    ) -> List[Dict[str, str]]:
        """
        Truncate messages to fit within token limit

        Args:
            messages: List of messages
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated messages list
        """
        total_tokens = 0
        truncated_messages: List[Dict[str, str]] = []

        system_message = None
        if messages and messages[0].get("role") == "system":
            system_message = messages[0]
            messages = messages[1:]

        for message in reversed(messages):
            message_tokens = self.count_tokens(message.get("content", ""))

            if total_tokens + message_tokens > max_tokens:
                if message.get("role") == "user":
                    truncated_content = self._truncate_text(
                        message.get("content", ""),
                        max_tokens - total_tokens,
                    )
                    if truncated_content:
                        msg_copy = message.copy()
                        msg_copy["content"] = truncated_content
                        truncated_messages.append(msg_copy)
                break

            total_tokens += message_tokens
            truncated_messages.append(message)

        truncated_messages.reverse()
        if system_message:
            truncated_messages.insert(0, system_message)

        return truncated_messages

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text

        keep_start = max_tokens // 2
        keep_end = max_tokens - keep_start
        truncated = tokens[:keep_start] + ["..."] + tokens[-keep_end:]
        return " ".join(truncated)
