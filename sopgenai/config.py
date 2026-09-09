from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Configuration:

    def __init__(
        self,
        app_config_path: Path,
        rules_config_path: Path,
    ):
        self.app = self._load(
            app_config_path
        )

        self.rules = self._load(
            rules_config_path
        )

    @staticmethod
    def _load(
        path: Path,
    ) -> Dict[str, Any]:

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return (
                yaml.safe_load(file)
                or {}
            )

    @property
    def llm(self):
        return self.app["llm"]

    @property
    def extraction(self):
        return self.app["extraction"]

    @property
    def embedding(self):
        return self.app["embedding"]

    @property
    def retrieval(self):
        return self.app["retrieval"]

    @property
    def authority(self):
        return self.app["authority"]

    @property
    def generation(self):
        return self.app["generation"]

    @property
    def output(self):
        return self.app["output"]
