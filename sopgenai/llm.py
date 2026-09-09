from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from huggingface_hub import InferenceClient


class LLMProvider(ABC):
    """
    Common interface used by the SOP application.

    All semantic components should depend on this interface
    rather than directly depending on Hugging Face.
    """

    @abstractmethod
    def structured_completion(
        self,
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class HuggingFaceLLM(LLMProvider):
    """
    Hugging Face Inference API implementation.

    The model should be an instruction/chat model available
    through the configured Hugging Face inference provider.
    """

    def __init__(
        self,
        model: str,
        api_token: str,
        provider: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        if not api_token:
            raise ValueError(
                "Hugging Face API token is required."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        client_args = {
            "api_key": api_token,
        }

        if provider:
            client_args["provider"] = provider

        self.client = InferenceClient(
            **client_args
        )

    def structured_completion(
        self,
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:

        user_content = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n\n"
                    + "Return valid JSON only. "
                    + "Do not include Markdown code fences."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        completion = self.client.chat_completion(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        content = (
            completion.choices[0]
            .message.content
        )

        return self._parse_json(content)

    @staticmethod
    def _parse_json(
        content: str,
    ) -> Dict[str, Any]:
        """
        Parse JSON defensively because some instruction models
        may wrap JSON in ```json ... ``` despite instructions.
        """

        if not content:
            raise ValueError(
                "Hugging Face model returned an empty response."
            )

        content = content.strip()

        # Remove Markdown code fences when returned by model.
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"\s*```$",
            "",
            content,
        )

        try:
            return json.loads(content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Hugging Face model did not return valid JSON.\n"
                f"Response:\n{content}"
            ) from exc
