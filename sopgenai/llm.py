from __future__ import annotations

import json
import os
import re
import urllib.request

from abc import (
    ABC,
    abstractmethod,
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)


class LLMProvider(ABC):

    @abstractmethod
    def json_completion(
        self,
        system_prompt,
        payload,
    ):

        raise NotImplementedError


def parse_json_response(
    content,
):

    if not content:

        raise ValueError(
            "LLM returned empty content."
        )

    content = (
        content.strip()
    )

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

        start = (
            content.find("{")
        )

        end = (
            content.rfind("}")
        )

        if (
            start >= 0
            and end > start
        ):

            return json.loads(
                content[
                    start:
                    end + 1
                ]
            )

        raise


# =========================================================
# OLLAMA
# =========================================================


class OllamaProvider(
    LLMProvider
):

    def __init__(
        self,
        config,
    ):

        provider = (
            config["ollama"]
        )

        self.model = (
            provider["model"]
        )

        self.base_url = (
            provider.get(
                "base_url",
                "http://localhost:11434",
            )
            .rstrip("/")
        )

        self.timeout = (
            config.get(
                "timeout",
                600,
            )
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
                2048,
            )
        )

        self.max_input_chars = (
            config.get(
                "max_input_chars",
                24000,
            )
        )

        self.keep_alive = (
            provider.get(
                "keep_alive",
                "10m",
            )
        )

        self.num_ctx = (
            provider.get(
                "num_ctx",
                16384,
            )
        )

    @retry(
        stop=stop_after_attempt(2),

        wait=wait_exponential(
            min=1,
            max=4,
        ),
    )
    def json_completion(
        self,
        system_prompt,
        payload,
    ):

        user_text = (
            json.dumps(
                payload,
                ensure_ascii=False,
            )
        )

        # ---------------------------------------------
        # INPUT SIZE GUARD
        # ---------------------------------------------

        if (
            len(user_text)
            > self.max_input_chars
        ):

            raise ValueError(
                "LLM request exceeds configured "
                "input-size limit. "
                f"Current={len(user_text)} chars, "
                f"Limit={self.max_input_chars}. "
                "Reduce semantic batch size or "
                "retrieval evidence count."
            )

        request_body = {

            "model":
                self.model,

            "stream":
                False,

            "format":
                "json",

            # -----------------------------------------
            # Keep model loaded between section calls
            # -----------------------------------------

            "keep_alive":
                self.keep_alive,

            "options": {

                "temperature":
                    self.temperature,

                # Lower output limit.
                "num_predict":
                    self.max_tokens,

                "num_ctx":
                    self.num_ctx,
            },

            "messages": [

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
                        user_text,
                },
            ],
        }

        print(
            "[LLM] "
            f"Ollama model={self.model}, "
            f"input_chars={len(user_text)}"
        )

        request = (
            urllib.request.Request(

                self.base_url
                + "/api/chat",

                data=(
                    json.dumps(
                        request_body
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

        with urllib.request.urlopen(
            request,
            timeout=self.timeout,
        ) as response:

            body = json.loads(
                response
                .read()
                .decode(
                    "utf-8"
                )
            )

        content = (
            body.get(
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


# =========================================================
# HUGGING FACE
# =========================================================


class HuggingFaceProvider(
    LLMProvider
):

    def __init__(
        self,
        config,
    ):

        from huggingface_hub import (
            InferenceClient,
        )

        provider = (
            config[
                "huggingface"
            ]
        )

        token_env = (
            provider.get(
                "api_key_env",
                "HF_TOKEN",
            )
        )

        token = os.getenv(
            token_env
        )

        if not token:

            raise RuntimeError(
                f"{token_env} is not set."
            )

        self.model = (
            provider["model"]
        )

        self.max_tokens = (
            config.get(
                "max_tokens",
                2048,
            )
        )

        self.temperature = (
            config.get(
                "temperature",
                0.0,
            )
        )

        self.client = (
            InferenceClient(

                model=self.model,

                provider=(
                    provider.get(
                        "inference_provider",
                        "auto",
                    )
                ),

                token=token,

                timeout=(
                    config.get(
                        "timeout",
                        600,
                    )
                ),
            )
        )

    def json_completion(
        self,
        system_prompt,
        payload,
    ):

        response = (
            self.client
            .chat_completion(

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            system_prompt,
                    },

                    {
                        "role":
                            "user",

                        "content":
                            json.dumps(
                                payload,
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


# =========================================================
# OPENAI
# =========================================================


class OpenAIProvider(
    LLMProvider
):

    def __init__(
        self,
        config,
    ):

        from openai import OpenAI

        provider = (
            config["openai"]
        )

        token_env = (
            provider.get(
                "api_key_env",
                "OPENAI_API_KEY",
            )
        )

        token = os.getenv(
            token_env
        )

        if not token:

            raise RuntimeError(
                f"{token_env} is not set."
            )

        kwargs = {
            "api_key":
                token
        }

        if provider.get(
            "base_url"
        ):

            kwargs[
                "base_url"
            ] = (
                provider[
                    "base_url"
                ]
            )

        self.client = (
            OpenAI(
                **kwargs
            )
        )

        self.model = (
            provider["model"]
        )

        self.max_tokens = (
            config.get(
                "max_tokens",
                2048,
            )
        )

        self.temperature = (
            config.get(
                "temperature",
                0.0,
            )
        )

    def json_completion(
        self,
        system_prompt,
        payload,
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

                        "content":
                            system_prompt,
                    },

                    {
                        "role":
                            "user",

                        "content":
                            json.dumps(
                                payload,
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


# =========================================================
# ANTHROPIC
# =========================================================


class AnthropicProvider(
    LLMProvider
):

    def __init__(
        self,
        config,
    ):

        import anthropic

        provider = (
            config[
                "anthropic"
            ]
        )

        token_env = (
            provider.get(
                "api_key_env",
                "ANTHROPIC_API_KEY",
            )
        )

        token = os.getenv(
            token_env
        )

        if not token:

            raise RuntimeError(
                f"{token_env} is not set."
            )

        self.client = (
            anthropic.Anthropic(
                api_key=token
            )
        )

        self.model = (
            provider["model"]
        )

        self.max_tokens = (
            config.get(
                "max_tokens",
                2048,
            )
        )

        self.temperature = (
            config.get(
                "temperature",
                0.0,
            )
        )

    def json_completion(
        self,
        system_prompt,
        payload,
    ):

        response = (
            self.client
            .messages
            .create(

                model=self.model,

                system=(
                    system_prompt
                ),

                messages=[
                    {
                        "role":
                            "user",

                        "content":
                            json.dumps(
                                payload,
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


# =========================================================
# FACTORY
# =========================================================


class LLMFactory:

    @staticmethod
    def create(
        config,
    ):

        provider = (
            config.get(
                "provider",
                "",
            )
            .strip()
            .lower()
        )

        if provider == "ollama":

            return OllamaProvider(
                config
            )

        if provider == "huggingface":

            return HuggingFaceProvider(
                config
            )

        if provider == "openai":

            return OpenAIProvider(
                config
            )

        if provider == "anthropic":

            return AnthropicProvider(
                config
            )

        raise ValueError(
            "Unsupported provider: "
            f"{provider}"
        )
