"""Thin wrapper around an OpenAI-compatible chat endpoint.

Every component talks to the model through `LLM.chat` (free text) or `LLM.structured`
(JSON validated against a Pydantic model). Tests and offline demos swap in `ScriptedLLM`.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import ModelSettings, model_settings

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> str:
    """Return the first JSON object in `text`, tolerating code fences and surrounding prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise LLMError(f"no JSON object in model output: {text[:200]!r}")
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise LLMError(f"unterminated JSON object in model output: {text[:200]!r}")


class LLM:
    """Base interface. Subclasses implement `_complete`."""

    name = "llm"

    def _complete(self, messages: list[Message], json_output: bool) -> str:
        raise NotImplementedError

    def chat(self, messages: list[Message]) -> str:
        return self._complete(messages, json_output=False).strip()

    def structured(self, messages: list[Message], schema: type[T], attempts: int = 3) -> T:
        """Ask for JSON matching `schema`; on parse/validation errors, show the error and retry."""
        schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        convo = list(messages)
        convo[0] = {
            "role": convo[0]["role"],
            "content": convo[0]["content"]
            + "\n\nReply with a single JSON object that conforms to this JSON schema:\n"
            + schema_hint,
        }
        last_error: Exception | None = None
        for _ in range(attempts):
            raw = self._complete(convo, json_output=True)
            try:
                return schema.model_validate_json(extract_json(raw))
            except (LLMError, ValidationError) as err:
                last_error = err
                convo = convo + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That reply was not valid: {err}. Reply again with only the corrected JSON object.",
                    },
                ]
        raise LLMError(f"{self.name}: could not get valid {schema.__name__} JSON: {last_error}")


class OpenAICompatibleLLM(LLM):
    """Any endpoint that speaks the OpenAI chat-completions protocol (official API or a relay)."""

    def __init__(self, settings: ModelSettings):
        if not settings.api_key:
            raise LLMError("LLM_API_KEY is not set; copy .env.example to .env and fill it in")
        self.settings = settings
        self.name = settings.model
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout,
            max_retries=0,  # retries handled below so that we can back off on any error type
        )

    def _complete(self, messages: list[Message], json_output: bool) -> str:
        kwargs: dict = {"model": self.settings.model, "messages": messages}
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        use_json_mode = json_output and self.settings.json_mode == "json_object"
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        delay = 2.0
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as err:  # network errors, rate limits, unsupported params
                if use_json_mode and "response_format" in str(err):
                    # Some relays reject response_format; fall back to prompt-only JSON.
                    kwargs.pop("response_format", None)
                    use_json_mode = False
                    continue
                if attempt == self.settings.max_retries:
                    raise LLMError(f"{self.name}: request failed: {err}") from err
                time.sleep(delay)
                delay *= 2
        raise LLMError(f"{self.name}: request failed")


class ScriptedLLM(LLM):
    """Deterministic stand-in for tests: `responder(messages, json_output)` returns the reply."""

    def __init__(self, responder: Callable[[list[Message], bool], str], name: str = "scripted"):
        self.responder = responder
        self.name = name
        self.calls: list[list[Message]] = []

    def _complete(self, messages: list[Message], json_output: bool) -> str:
        self.calls.append(messages)
        return self.responder(messages, json_output)


def get_llm(role: str) -> LLM:
    """Build the model client for `role` from environment settings."""
    return OpenAICompatibleLLM(model_settings(role))
