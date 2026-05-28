"""
metrics/counting.py
-------------------
Counting metric: the LLM does NOT produce a numeric count.
Instead it outputs a list of concrete *examples* (excerpt + explanation) of the
phenomenon found in the reasoning trace.

The pipeline then uses len(examples) as the count when needed for plots, but the
primary output is the list of textual examples – which is much more interpretable.

Examples:
  - Branchiness        → list of branching moments
  - Backtracking       → list of self-correction moments
  - Self-verification  → list of verification moments
  - Neg. constraint    → list of ruling-out moments
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

import llm_client
from metrics.base import BaseMetric, MetricResult
from prompts import render_system_prompt

# ---------------------------------------------------------------------------
# Pydantic response schema for ALL counting metrics
# ---------------------------------------------------------------------------

class ExampleItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    excerpt: str = Field(
        ...,
        description="Exact short quote from the reasoning text.",
        validation_alias=AliasChoices("excerpt", "text"),
    )
    explanation: str = Field(..., description="Why this excerpt is an example of the phenomenon.")


class CountingResponse(BaseModel):
    overall_analysis: str = Field(
        ...,
        description="Brief overall analysis of the reasoning trace with respect to this metric.",
    )
    examples: list[ExampleItem] = Field(
        default_factory=list,
        description=(
            "List of concrete examples found in the text. "
            "Return an empty list if none are found."
        ),
    )


# ---------------------------------------------------------------------------
# CountingMetric class
# ---------------------------------------------------------------------------

class CountingMetric(BaseMetric):
    """
    A metric that asks the LLM to extract concrete *examples* of a phenomenon.

    Parameters
    ----------
    name : str
        Unique snake_case identifier (e.g. ``"branchiness"``).
    description : str
        One-line description shown in reports.
    system_prompt : str
        Full system prompt sent to the judge model.
    user_prompt_template : str
        Template for the user message.  Use ``{text}`` and ``{dataset}`` as
        placeholders.  ``{dataset}`` is optional.
    """

    metric_type = "counting"

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
    def schema(self) -> type[CountingResponse]:
        return CountingResponse

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

        user_prompt = (
            self._user_prompt_template
            .replace("{text}", text)
            .replace("{dataset}", dataset)
        )

        system_prompt = render_system_prompt(self._system_prompt, self.name, dataset)

        payload = llm_client.ask_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=CountingResponse,
            dataset=dataset,
            problem_id=problem_id,
            metric_type=self.name,
            checkpoint=checkpoint,
            run_id=run_id,
        )

        tokens: dict = payload.pop("__tokens__", {})
        analysis: str = payload.get("overall_analysis", "")
        raw_examples: list = payload.get("examples", [])

        examples = [
            {
                "excerpt": (
                    e.get("excerpt", e.get("text", "")) if isinstance(e, dict) else str(e)
                ),
                "explanation": e.get("explanation", "") if isinstance(e, dict) else "",
            }
            for e in raw_examples
        ]

        return MetricResult(
            metric_name=self.name,
            detected=len(examples) > 0,
            reasoning=analysis,
            examples=examples,
            tokens=tokens,
            raw=payload,
        )
