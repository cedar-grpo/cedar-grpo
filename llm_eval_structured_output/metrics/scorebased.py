"""
metrics/scorebased.py
---------------------
Score-based metric: the LLM assigns a graded numeric score for a phenomenon
in the reasoning trace rather than a binary yes/no.

The canonical use case is Evidence-Explanation Directionality Awareness:
  1.0 – correct directionality   (observation → explanation)
  0.5 – ambiguous directionality
  0.0 – backward / circular / missing directionality

Response schema (Pydantic):
  reasoning_analysis   : str   – 1–2 sentence justification
  directionality_score : float – snapped to nearest of {0.0, 0.5, 1.0}

MetricResult mapping:
  score    → directionality_score  (the primary output)
  reasoning → reasoning_analysis
  detected → True when score > 0.0  (i.e. not clearly backward/missing)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

import llm_client
from metrics.base import BaseMetric, MetricResult
from prompts import render_system_prompt

# ---------------------------------------------------------------------------
# Allowed score set
# ---------------------------------------------------------------------------

_ALLOWED_SCORES: frozenset[float] = frozenset({0.0, 0.5, 1.0})


def _snap_to_allowed(v: float) -> float:
    """Round an arbitrary float to the nearest member of {0.0, 0.5, 1.0}."""
    return min(_ALLOWED_SCORES, key=lambda s: abs(s - v))


# ---------------------------------------------------------------------------
# Pydantic response schema
# ---------------------------------------------------------------------------

class ScoreBasedResponse(BaseModel):
    reasoning_analysis: str = Field(
        ...,
        description=(
            "1–2 sentence explanation of the logical flow observed, "
            "specifically noting the presence or absence of directional logic."
        ),
    )
    directionality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Graded score: 1.0 = correct directionality, "
            "0.5 = ambiguous, 0.0 = backward / circular / missing."
        ),
    )

    @field_validator("directionality_score")
    @classmethod
    def snap_score(cls, v: float) -> float:
        return _snap_to_allowed(v)


# ---------------------------------------------------------------------------
# ScoreBasedMetric class
# ---------------------------------------------------------------------------

class ScoreBasedMetric(BaseMetric):
    """
    A metric that asks the LLM to assign a graded score for a phenomenon
    observed in the reasoning trace.

    Parameters
    ----------
    name : str
        Unique snake_case identifier
        (e.g. ``"evidence_explanation_directionality_scorebased"``).
    description : str
        One-line description shown in reports.
    system_prompt : str
        Full system prompt sent to the judge model.
    user_prompt_template : str
        Template for the user message.  Supported placeholders:
        ``{text}``       – the reasoning chain (always provided)
        ``{dataset}``    – dataset name
        ``{full_input}`` – original question / observations (from context dict)
    """

    metric_type = "scorebased"

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
    def schema(self) -> type[ScoreBasedResponse]:
        return ScoreBasedResponse

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
            response_schema=ScoreBasedResponse,
            dataset=dataset,
            problem_id=problem_id,
            metric_type=self.name,
            checkpoint=checkpoint,
            run_id=run_id,
        )

        tokens: dict = payload.pop("__tokens__", {})
        reasoning_analysis: str = payload.get("reasoning_analysis", "")
        raw_score: float = float(payload.get("directionality_score", 0.0))
        score: float = _snap_to_allowed(raw_score)

        return MetricResult(
            metric_name=self.name,
            detected=(score > 0.0),
            reasoning=reasoning_analysis,
            score=score,
            tokens=tokens,
            raw=payload,
        )
