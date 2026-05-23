import json
import logging

from openai import OpenAI
from pydantic import BaseModel

from .config import get_settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        s = get_settings()
        _client = OpenAI(
            api_key=s.llm_api_key,
            base_url=s.llm_api_base,
        )
    return _client


def call_llm(
    system: str,
    user: str,
    response_model: type[BaseModel] | None = None,
) -> str | BaseModel:
    client = _get_client()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if response_model is not None:
        schema = response_model.model_json_schema()
        resp = client.chat.completions.create(
            model=get_settings().llm_model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        content = resp.choices[0].message.content
        logger.debug("LLM raw response: %s", content)
        return response_model.model_validate(json.loads(content))

    resp = client.chat.completions.create(
        model=get_settings().llm_model,
        messages=messages,
    )
    content = resp.choices[0].message.content
    logger.debug("LLM raw response: %s", content)
    return content


def call_llm_multi(
    messages: list[dict],
    response_model: type[BaseModel] | None = None,
) -> tuple[str | BaseModel, list[dict]]:
    client = _get_client()

    kwargs = {"model": get_settings().llm_model, "messages": messages}

    if response_model is not None:
        schema = response_model.model_json_schema()
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    logger.debug("LLM raw response: %s", content)

    updated_messages = messages + [{"role": "assistant", "content": content}]

    if response_model is not None:
        return response_model.model_validate(json.loads(content)), updated_messages

    return content, updated_messages
