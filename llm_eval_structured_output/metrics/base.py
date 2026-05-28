"""
metrics/base.py
---------------
Abstract base class and shared data structures for all metric types.

Adding a new metric is a two-step process:
  1. Create a prompt file under prompts/binary/ or prompts/counting/.
  2. Instantiate BinaryMetric or CountingMetric and register it in registry.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Shared result container
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """
    Unified result returned by every metric regardless of type.

    Binary metrics:
        detected  – True / False
        reasoning – model's explanation for its decision
        examples  – empty list (not used)
        score     – None (not applicable)

    Counting metrics:
        detected  – True when at least one example was found
        reasoning – model's overall analysis
        examples  – list of {"excerpt": str, "explanation": str} dicts
        score     – None (not applicable; use example_count instead)

    Coverage metrics:
        detected  – True when all observation details are addressed (score == 1.0)
        reasoning – model's overall analysis
        examples  – list of {"detail": str, "addressed": bool, "evidence": str} dicts
        score     – proportion of addressed details (0.0 – 1.0)
    """
    metric_name: str
    detected: bool = False
    reasoning: str = ""
    examples: list[dict[str, str]] = field(default_factory=list)
    error: str = ""
    score: float | None = None

    # Token usage reported by the LLM API for this metric call.
    # Keys present: "input", "output", and optionally "reasoning" / "cached_input".
    # Empty dict when response came from cache or the API did not report usage.
    tokens: dict[str, int] = field(default_factory=dict)

    # Raw payload returned by the LLM (for debugging / logging)
    raw: dict[str, Any] = field(default_factory=dict)

    # Optional scalar outputs for custom metric types.
    scalar_metrics: dict[str, float] = field(default_factory=dict)
    normalized_scalar_metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseMetric(ABC):
    """
    Every metric must implement ``evaluate``.

    Attributes
    ----------
    name : str
        Unique snake_case identifier used throughout the pipeline.
    metric_type : str
        Either ``"binary"`` or ``"counting"``.
    description : str
        Human-readable description shown in reports.
    """

    name: str
    metric_type: str  # "binary" | "counting"
    description: str = ""

    @abstractmethod
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
        """Run the metric on ``text`` and return a :class:`MetricResult`."""

    @property
    def schema(self) -> type[BaseModel]:
        """Pydantic model used as the JSON response schema for ask_llm."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} name={self.name!r} type={self.metric_type!r}>"
