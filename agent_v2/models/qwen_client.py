import os
from typing import Optional, Callable

from openai import OpenAI
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from agent_v2.models.llm_client import LLMClient

_client: Optional[OpenAI] = None


def _normalize_openai_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    suffix = "/chat/completions"
    if endpoint.endswith(suffix):
        return endpoint[:-len(suffix)]
    return endpoint


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        endpoint = os.environ.get(
            "OPENAI_API_ENDPOINT",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if not api_key:
            raise RuntimeError("请先设置环境变量 OPENAI_API_KEY 或 API_KEY。")
        _client = OpenAI(
            api_key=api_key,
            base_url=_normalize_openai_base_url(endpoint),
            timeout=600,
        )
    return _client


def _extra_body() -> dict:
    enable_thinking = os.environ.get("OPENAI_ENABLE_THINKING")
    if enable_thinking is None:
        return {}
    return {"enable_thinking": enable_thinking.lower() == "true"}


def _normalize_max_tokens(max_tokens: int) -> int:
    limit = os.environ.get("OPENAI_MAX_TOKENS_LIMIT")
    if not limit:
        return max_tokens
    return min(max_tokens, int(limit))

def chat(
        prompt: str,
        system_prompt: Optional[str] = None,
        *,
        model: str = "qwen3.6-35b",
        history: Optional[list[ChatCompletionMessageParam]] = None,
        max_tokens: int = 16384,
) -> str:
    if history is None:
        history = []
    logger.info(f"调用AI聊天接口，模型: {model}")
    logger.debug(f"用户提示词: {prompt[:100]}...")
    if system_prompt:
        logger.debug(f"系统提示词: {system_prompt[:100]}...")

    messages = [it for it in history if it["role"] != "system"]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    messages.append({"role": "user", "content": prompt})

    for i in range(3):
        try:
            logger.debug("发送请求到AI API")
            client = _get_client()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=600,
                max_tokens=_normalize_max_tokens(max_tokens),
                temperature=0.3,
                extra_body=_extra_body(),
            )
            result = response.choices[0].message.content or ""
            logger.info(f"AI响应成功，结果长度: {len(result)}字符")
            logger.debug(f"AI响应结果: {result[:100]}...")
            messages.append({"role": "assistant", "content": result})
            history.clear()
            history.extend(messages)
            return result
        except Exception as e:
            logger.error(f"AI API调用失败 (尝试{i+1}/3): {e}")
            if i >= 2:
                raise e


class QwenClient(LLMClient):
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        return chat(prompt, max_tokens=max_tokens or 16384)
