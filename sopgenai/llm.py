from __future__ import annotations

import json
import os
import re

from abc import (
    ABC,
    abstractmethod,
)

from typing import (
    Any,
    Dict,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)


# ==========================================================
# BASE LLM INTERFACE
# ==========================================================


class LLMProvider(ABC):

    @abstractmethod
    def json_completion(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
    ) -> Dict[str, Any]:

        raise NotImplementedError


# ==========================================================
# JSON RESPONSE PARSER
# ==========================================================


def parse_json_response(
    content: str,
) -> Dict[str, Any]:

    if not content:

        raise ValueError(
            "LLM returned an empty response."
        )

    content = content.strip()

    # Remove Markdown JSON fences if returned.

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

        return json.loads(
            content
        )

    except json.JSONDecodeError:

        # Some models may include additional
        # text around the JSON.

        start = content.find(
            "{"
        )

        end = content.rfind(
            "}"
        )

        if (
            start >= 0
            and end > start
        ):

            candidate = (
                content[
                    start:end + 1
                ]
            )

            return json.loads(
                candidate
            )

        raise


# ==========================================================
# HUGGING FACE
# ==========================================================


class HuggingFaceProvider(
    LLMProvider
):

    def __init__(
        self,
        config: dict,
    ):

        from huggingface_hub import (
            InferenceClient,
        )

        provider_config = (
            config[
                "huggingface"
            ]
        )

        self.model = (
            provider_config[
                "model"
            ]
        )

        token_env = (
            provider_config.get(
                "api_key_env",
                "HF_TOKEN",
            )
        )

        token = os.getenv(
            token_env
        )

        if not token:

            raise RuntimeError(
                f"Environment variable "
                f"{token_env} is not set."
            )

        self.temperature = (
            config.get(
                "temperature",
                0.0,
            )
        )

        self.max_tokens = (
            config.get(
                "max_tokens",
                8192,
            )
        )

        self.client = (
            InferenceClient(
                model=self.model,

                provider=(
                    provider_config.get(
                        "inference_provider",
                        "auto",
                    )
                ),

                token=token,

                timeout=(
                    config.get(
                        "timeout",
                        180,
                    )
                ),
            )
        )

    @retry(
        stop=stop_after_attempt(3),

        wait=wait_exponential(
            min=1,
            max=8,
        ),
    )
    def json_completion(
        self,
        system_prompt,
        user_payload,
    ):

        response = (
            self.client
            .chat_completion(
                messages=[
                    {
                        "role":
                            "system",

                        "content": (
                            system_prompt
                            + "\n"
                            + "Return valid JSON only. "
                            + "Do not use Markdown."
                        ),
                    },

                    {
                        "role":
                            "user",

                        "content":
                            json.dumps(
                                user_payload,
                                ensure_ascii=False,
                            ),
                    },
                ],

                max_tokens=(
                    self.max_tokens
                ),

                temperature=(
                    self.temperature
                ),
            )
        )

        return parse_json_response(
            response
            .choices[0]
            .message
            .content
        )


# ==========================================================
# OPENAI
# ==========================================================


class OpenAIProvider(
    LLMProvider
):

    def __init__(
        self,
        config: dict,
    ):

        from openai import OpenAI

        provider_config = (
            config[
                "openai"
            ]
        )

        token_env = (
            provider_config.get(
                "api_key_env",
                "OPENAI_API_KEY",
            )
        )

        token = os.getenv(
            token_env
        )

        if not token:

            raise RuntimeError(
                f"Environment variable "
                f"{token_env} is not set."
            )

        self.model = (
            provider_config[
                "model"
            ]
        )

        self.temperature = (
            config.get(
                "temperature",
                0.0,
            )
        )

        self.max_tokens = (
            config.get(
                "max_tokens",
                8192,
            )
        )

        kwargs = {
            "api_key":
                token
        }

        base_url = (
            provider_config.get(
                "base_url"
            )
        )

        if base_url:

            kwargs[
                "base_url"
            ] = base_url

        self.client = (
            OpenAI(
                **kwargs
            )
        )

    @retry(
        stop=stop_after_attempt(3),

        wait=wait_exponential(
            min=1,
            max=8,
        ),
    )
    def json_completion(
        self,
        system_prompt,
        user_payload,
    ):

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,

                messages=[
                    {
                        "role":
                            "system",

                        "content": (
                            system_prompt
                            + "\n"
                            + "Return valid JSON only."
                        ),
                    },

                    {
                        "role":
                            "user",

                        "content":
                            json.dumps(
                                user_payload,
                                ensure_ascii=False,
                            ),
                    },
                ],

                temperature=(
                    self.temperature
                ),

                max_tokens=(
                    self.max_tokens
                ),

                response_format={
                    "type":
                        "json_object"
                },
            )
        )

        return parse_json_response(
            response
            .choices[0]
            .message
            .content
        )


# ==========================================================
# ANTHROPIC
# ==========================================================


class AnthropicProvider(
    LLMProvider
):

    def __init__(
        self,
        config: dict,
    ):

        import anthropic

        provider_config = (
            config[
                "anthropic"
            ]
        )

        token_env = (
            provider_config.get(
                "api_key_env",
                "ANTHROPIC_API_KEY",
            )
        )

        token = os.getenv(
            token_env
        )

        if not token:

            raise RuntimeError(
                f"Environment variable "
                f"{token_env} is not set."
            )

        self.model = (
            provider_config[
                "model"
            ]
        )

        self.temperature = (
            config.get(
                "temperature",
                0.0,
            )
        )

        self.max_tokens = (
            config.get(
                "max_tokens",
                8192,
            )
        )

        self.client = (
            anthropic.Anthropic(
                api_key=token
            )
        )

    @retry(
        stop=stop_after_attempt(3),

        wait=wait_exponential(
            min=1,
            max=8,
        ),
    )
    def json_completion(
        self,
        system_prompt,
        user_payload,
    ):

        response = (
            self.client
            .messages
            .create(
                model=self.model,

                system=(
                    system_prompt
                    + "\n"
                    + "Return valid JSON only. "
                    + "Do not use Markdown."
                ),

                messages=[
                    {
                        "role":
                            "user",

                        "content":
                            json.dumps(
                                user_payload,
                                ensure_ascii=False,
                            ),
                    }
                ],

                max_tokens=(
                    self.max_tokens
                ),

                temperature=(
                    self.temperature
                ),
            )
        )

        content = "".join(
            block.text

            for block
            in response.content

            if hasattr(
                block,
                "text",
            )
        )

        return parse_json_response(
            content
        )


# ==========================================================
# OLLAMA
# ==========================================================


class OllamaProvider(
    LLMProvider
):

    """
    Local or self-hosted Ollama LLM provider.

    The configured model must already be available
    to the Ollama server.

    Example:

        ollama pull qwen3:8b

    Ollama API:

        http://localhost:11434/api/chat
    """

    def __init__(
        self,
        config: dict,
    ):

        import urllib.request

        provider_config = (
            config[
                "ollama"
            ]
        )

        self.model = (
            provider_config[
                "model"
            ]
        )

        self.base_url = (
            provider_config.get(
                "base_url",
                "http://localhost:11434",
            )
            .rstrip("/")
        )

        self.temperature = (
            config.get(
                "temperature",
                0.0,
            )
        )

        self.timeout = (
            config.get(
                "timeout",
                180,
            )
        )

        self._urlopen = (
            urllib.request.urlopen
        )

        self._Request = (
            urllib.request.Request
        )

    @retry(
        stop=stop_after_attempt(3),

        wait=wait_exponential(
            min=1,
            max=8,
        ),
    )
    def json_completion(
        self,
        system_prompt,
        user_payload,
    ):

        payload = {

            "model":
                self.model,

            # Do not stream because the pipeline
            # expects one complete JSON response.

            "stream":
                False,

            # Ask Ollama to enforce JSON output.

            "format":
                "json",

            "options": {

                "temperature":
                    self.temperature
            },

            "messages": [

                {
                    "role":
                        "system",

                    "content": (
                        system_prompt
                        + "\n"
                        + "Return valid JSON only. "
                        + "Do not use Markdown."
                    ),
                },

                {
                    "role":
                        "user",

                    "content":
                        json.dumps(
                            user_payload,
                            ensure_ascii=False,
                        ),
                },
            ],
        }

        request = (
            self._Request(

                self.base_url
                + "/api/chat",

                data=(
                    json.dumps(
                        payload
                    )
                    .encode(
                        "utf-8"
                    )
                ),

                headers={
                    "Content-Type":
                        "application/json"
                },

                method="POST",
            )
        )

        with self._urlopen(
            request,

            timeout=(
                self.timeout
            ),

        ) as response:

            body = json.loads(
                response
                .read()
                .decode(
                    "utf-8"
                )
            )

        content = (
            body
            .get(
                "message",
                {}
            )
            .get(
                "content",
                "",
            )
        )

        return parse_json_response(
            content
        )


# ==========================================================
# LLM FACTORY
# ==========================================================


class LLMFactory:

    @staticmethod
    def create(
        config: dict,
    ) -> LLMProvider:

        provider = (
            config.get(
                "provider",
                "",
            )
            .strip()
            .lower()
        )

        if provider == "huggingface":

            return (
                HuggingFaceProvider(
                    config
                )
            )

        if provider == "openai":

            return (
                OpenAIProvider(
                    config
                )
            )

        if provider == "anthropic":

            return (
                AnthropicProvider(
                    config
                )
            )

        if provider == "ollama":

            return (
                OllamaProvider(
                    config
                )
            )

        raise ValueError(
            "Unsupported LLM provider: "
            f"{provider}. "
            "Supported providers are: "
            "huggingface, openai, "
            "anthropic and ollama."
        )
