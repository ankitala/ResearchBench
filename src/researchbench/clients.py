"""Model client abstractions."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


class ModelClient:
    """Minimal text-generation client interface."""

    is_mock = False

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        raise NotImplementedError


class MockClient(ModelClient):
    """Deterministic local client used for tests and smoke runs."""

    is_mock = True

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        if "**Selection of research hypothesis candidate**" in prompt:
            return "**Selection of research hypothesis candidate**: candidate 1"
        if "Matched score starts" in prompt or "Matched score:" in prompt:
            return "Reason: mock score based on deterministic smoke testing.\n**Matched score starts:** 3 **Matched score ends**"
        if "Validness Score starts" in prompt:
            return (
                "**Validness Reason starts:** mock validness **Validness Reason ends**\n"
                "**Validness Score starts:** 3 **Validness Score ends**\n"
                "**Novelty Reason starts:** mock novelty **Novelty Reason ends**\n"
                "**Novelty Score starts:** 3 **Novelty Score ends**\n"
                "**Significance Reason starts:** mock significance **Significance Reason ends**\n"
                "**Significance Score starts:** 3 **Significance Score ends**\n"
                "**Specificity Reason starts:** mock specificity **Specificity Reason ends**\n"
                "**Specificity Score starts:** 3 **Specificity Score ends**"
            )
        if "Title 1 starts" in prompt:
            titles = re.findall(r"Inspiration candidate \d+\. Title: (.*?); Abstract:", prompt)
            selected = titles[:3]
            return "\n".join(
                f"**Title {idx} starts:** {title} **Title {idx} ends**"
                for idx, title in enumerate(selected, 1)
            )
        if "Refined Hypothesis starts" in prompt:
            return (
                "**Refined Hypothesis starts:** A mock refined hypothesis combines the provided background "
                "with the supplied inspirations to propose a testable research direction. "
                "**Refined Hypothesis ends**\n"
                "**Key Reasoning starts:** This deterministic response is intended for smoke tests. "
                "**Key Reasoning ends**"
            )
        if "Hypothesis starts" in prompt:
            return (
                "**Hypothesis starts:** A mock hypothesis combines the provided background with the supplied "
                "inspirations to propose a testable research direction. **Hypothesis ends**\n"
                "**Key Reasoning starts:** This deterministic response is intended for smoke tests. "
                "**Key Reasoning ends**"
            )
        return "Mock response."


@dataclass
class OpenAICompatibleClient(ModelClient):
    model: str
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for non-mock model calls.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional dependency with `pip install -e .[openai]`.") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return completion.choices[0].message.content or ""


def make_client(model: str, *, api_key: str | None = None, base_url: str | None = None) -> ModelClient:
    if model == "mock":
        return MockClient()
    return OpenAICompatibleClient(model=model, api_key=api_key, base_url=base_url)
