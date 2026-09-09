from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict

from openai import OpenAI


class LLMProvider(ABC):

    @abstractmethod
    def structured_completion(
        self,
        system_prompt: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:

        raise NotImplementedError


class OpenAICompatibleLLM(
    LLMProvider
):

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ):

        kwargs = {
            "api_key": api_key
        }

        if base_url:
            kwargs["base_url"] = (
                base_url
            )

        self.client = OpenAI(
            **kwargs
        )

        self.model = model

    def structured_completion(
        self,
        system_prompt,
        payload,
    ):

        response = (
            self.client.chat.completions.create(
                model=self.model,

                temperature=0,

                response_format={
                    "type": "json_object"
                },

                messages=[
                    {
                        "role": "system",
                        "content":
                            system_prompt,
                    },
                    {
                        "role": "user",
                        "content":
                            json.dumps(
                                payload,
                                ensure_ascii=False,
                            ),
                    },
                ],
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        return json.loads(
            content
        )
