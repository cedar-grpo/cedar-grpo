"""
metrics/coverage.py
-------------------
Coverage metric: the LLM enumerates every specific detail mentioned in the
observation (or observable evidence) and decides whether the chosen hypothesis
explicitly addresses each one.

Two-step structured output
  Step 1  – per-detail annotation
      observation_details : list[ObservationDetail]
          detail    : str   – one specific fact from the observation
          addressed : bool  – did the hypothesis/reasoning explicitly account for it?
          evidence  : str   – supporting quote from the reasoning trace

  Step 2  – synthesis
      overall_analysis : str – brief holistic commentary

Final score  =  addressed_count / total_details   (0.0 – 1.0)
A score of 1.0 means the chosen hypothesis accounts for every observation detail.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

import llm_client
from metrics.base import BaseMetric, MetricResult
from prompts import render_system_prompt

# ---------------------------------------------------------------------------
# Pydantic response schema
# ---------------------------------------------------------------------------

class ObservationDetail(BaseModel):
    detail: str = Field(
        ...,
        description="A single specific fact or detail from the observation that the hypothesis should explain.",
    )
    addressed: bool = Field(
        ...,
        description="True if the reasoning trace explicitly connects this detail to the chosen hypothesis.",
    )
    evidence: str = Field(
        default="",
        description=(
            "Exact short quote from the reasoning trace that shows the detail is addressed. "
            "Leave empty if addressed is False."
        ),
    )


class ObservationCoverageResponse(BaseModel):
    observation_details: list[ObservationDetail] = Field(
        ...,
        description=(
            "Exhaustive list of every specific detail present in the observation. "
            "Each detail must be assessed independently."
        ),
    )
    overall_analysis: str = Field(
        ...,
        description=(
            "Brief synthesis: how completely does the chosen hypothesis account for "
            "the full set of observation details, and what (if anything) is left unexplained?"
        ),
    )


# ---------------------------------------------------------------------------
# CoverageMetric class
# ---------------------------------------------------------------------------

class CoverageMetric(BaseMetric):
    """
    A metric that measures the proportion of observation details explicitly
    accounted for by the chosen hypothesis.

    Parameters
    ----------
    name : str
        Unique snake_case identifier (e.g. ``"observation_coverage"``).
    description : str
        One-line description shown in reports.
    system_prompt : str
        Full system prompt sent to the judge model.
    user_prompt_template : str
        Template for the user message.  Use ``{text}`` and ``{dataset}`` as
        placeholders.  ``{dataset}`` is optional.
    """

    metric_type = "coverage"

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
    def schema(self) -> type[ObservationCoverageResponse]:
        return ObservationCoverageResponse

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

        payload = llm_client.ask_llm(
            system_prompt=render_system_prompt(self._system_prompt, self.name, dataset),
            user_prompt=user_prompt,
            source_full_input=full_input,
            response_schema=ObservationCoverageResponse,
            dataset=dataset,
            problem_id=problem_id,
            metric_type=self.name,
            checkpoint=checkpoint,
            run_id=run_id,
        )

        tokens: dict = payload.pop("__tokens__", {})
        analysis: str = payload.get("overall_analysis", "")
        raw_details: list = payload.get("observation_details", [])

        # Normalise each detail into a plain dict
        details: list[dict] = []
        for d in raw_details:
            if isinstance(d, dict):
                details.append(
                    {
                        "detail": d.get("detail", ""),
                        "addressed": bool(d.get("addressed", False)),
                        "evidence": d.get("evidence", ""),
                    }
                )

        # Core metric: proportion of addressed details
        n_total = len(details)
        n_addressed = sum(1 for d in details if d["addressed"])
        score = round(n_addressed / n_total, 4) if n_total > 0 else 0.0

        return MetricResult(
            metric_name=self.name,
            detected=(score == 1.0),   # True only when ALL details are covered
            reasoning=analysis,
            examples=details,          # each dict: detail / addressed / evidence
            score=score,
            tokens=tokens,
            raw=payload,
        )
