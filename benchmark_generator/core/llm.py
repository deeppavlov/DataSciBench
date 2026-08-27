import json
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, cast, overload

from openai import Omit, OpenAI, omit
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel

from .config import get_settings

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


@lru_cache
def _get_client() -> OpenAI:
    s = get_settings()
    return OpenAI(
        api_key=s.llm_api_key,
        base_url=s.llm_api_base,
    )


def _build_response_format(response_model: type[BaseModel]) -> ResponseFormatJSONSchema:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": response_model.model_json_schema(),
            "strict": True,
        },
    }


@overload
def call_llm(system: str, user: str, response_model: None = None) -> str: ...


@overload
def call_llm[T: BaseModel](system: str, user: str, response_model: type[T]) -> T: ...


def call_llm[T: BaseModel](
    system: str,
    user: str,
    response_model: type[T] | None = None,
) -> str | T:
    client = _get_client()

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if response_model is not None:
        resp = client.chat.completions.create(
            model=get_settings().llm_model,
            messages=messages,
            response_format=_build_response_format(response_model),
        )
        content = cast("str", resp.choices[0].message.content)
        logger.debug("LLM raw response: %s", content)
        return response_model.model_validate(json.loads(content))

    resp = client.chat.completions.create(
        model=get_settings().llm_model,
        messages=messages,
    )
    content = cast("str", resp.choices[0].message.content)
    logger.debug("LLM raw response: %s", content)
    return content


@overload
def call_llm_multi(
    messages: list[dict[str, str]],
    response_model: None = None,
) -> tuple[str, list[dict[str, str]]]: ...


@overload
def call_llm_multi[T: BaseModel](
    messages: list[dict[str, str]],
    response_model: type[T],
) -> tuple[T, list[dict[str, str]]]: ...


def call_llm_multi[T: BaseModel](
    messages: list[dict[str, str]],
    response_model: type[T] | None = None,
) -> tuple[str | T, list[dict[str, str]]]:
    client = _get_client()

    response_format: ResponseFormatJSONSchema | Omit = omit
    if response_model is not None:
        response_format = _build_response_format(response_model)

    resp = client.chat.completions.create(
        model=get_settings().llm_model,
        messages=cast("list[ChatCompletionMessageParam]", messages),
        response_format=response_format,
    )
    content = cast("str", resp.choices[0].message.content)
    logger.debug("LLM raw response: %s", content)

    updated_messages = [*messages, {"role": "assistant", "content": content}]

    if response_model is not None:
        return response_model.model_validate(json.loads(content)), updated_messages

    return content, updated_messages
