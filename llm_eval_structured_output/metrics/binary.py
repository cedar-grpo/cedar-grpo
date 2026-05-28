"""
metrics/binary.py
-----------------
Binary metric: the LLM reasons about whether a phenomenon is present (yes/no)
and explains *why*.

No counts, no regex – just a structured yes/no + explanation.

Example metric: "Uncertainty Language"
  → Does the model use probabilistic language rather than absolute certainty?
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

import llm_client
from metrics.base import BaseMetric, MetricResult
from prompts import render_system_prompt

# ---------------------------------------------------------------------------
# Pydantic response schema for ALL binary metrics
# ---------------------------------------------------------------------------

class BinaryResponse(BaseModel):
    detected: bool = Field(
        ...,
        description="True if the phenomenon is present in the reasoning trace, False otherwise.",
    )
    reasoning: str = Field(
        ...,
        description="Step-by-step explanation of why the phenomenon is or is not present.",
    )
    evidence: str = Field(
        default="",
        description=(
            "Direct quote from the text that most strongly supports the decision. "
            "Leave empty if detected is False."
        ),
    )


# ---------------------------------------------------------------------------
# BinaryMetric class
# ---------------------------------------------------------------------------

class BinaryMetric(BaseMetric):
    """
    A metric that asks the LLM a yes/no question about the reasoning trace.

    Parameters
    ----------
    name : str
        Unique snake_case identifier (e.g. ``"uncertainty_language"``).
    description : str
        One-line description shown in reports.
    system_prompt : str
        Full system prompt sent to the judge model.
    user_prompt_template : str
        Template for the user message.  Use ``{text}`` and ``{dataset}`` as
        placeholders.  ``{dataset}`` is optional.
    """

    metric_type = "binary"

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self.name = name
        self.description = description
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template

    @property
    def schema(self) -> type[BinaryResponse]:
        return BinaryResponse

    def evaluate(
        self,
        text: str,
        *,
        dataset: str = "unknown",
        problem_id: str = "N/A",
        checkpoint: str = "N/A",
        run_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> MetricResult:
        if not isinstance(text, str) or not text.strip():
            return MetricResult(
                metric_name=self.name,
                error="Empty or invalid input text.",
            )

        full_input = ""
        if context and isinstance(context, dict):
            value = context.get("full_input")
            if value is not None:
                full_input = value if isinstance(value, str) else str(value)

        user_prompt = (
            self._user_prompt_template
            .replace("{text}", text)
            .replace("{dataset}", dataset)
            .replace("{full_input}", full_input)
        )

        system_prompt = render_system_prompt(self._system_prompt, self.name, dataset)

        payload = llm_client.ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            source_full_input=full_input,
            response_schema=BinaryResponse,
            dataset=dataset,
            problem_id=problem_id,
            metric_type=self.name,
            checkpoint=checkpoint,
            run_id=run_id,
        )

        detected: bool = bool(payload.get("detected", False))
        reasoning: str = payload.get("reasoning", "")
        evidence: str = payload.get("evidence", "")
        tokens: dict = payload.pop("__tokens__", {})

        examples = [{"excerpt": evidence, "explanation": reasoning}] if detected and evidence else []

        return MetricResult(
            metric_name=self.name,
            detected=detected,
            reasoning=reasoning,
            examples=examples,
            tokens=tokens,
            raw=payload,
        )
