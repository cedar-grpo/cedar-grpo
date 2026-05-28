"""
metrics/registry.py
-------------------
Central registry of all metrics.

To add a new metric:
  1. Create its prompt file in prompts/binary/ or prompts/counting/.
  2. Import the prompt strings here.
  3. Instantiate BinaryMetric or CountingMetric.
  4. Add it to METRICS.

The pipeline resolves active metrics at runtime by filtering METRICS against
config.ACTIVE_METRICS.
"""

from __future__ import annotations

from metrics.base import BaseMetric
from metrics.binary import BinaryMetric
from metrics.counting import CountingMetric
from metrics.coverage import CoverageMetric
from metrics.graph_structure import RationaleGraphMetric
from metrics.scorebased import ScoreBasedMetric

# ---------------------------------------------------------------------------
# Prompt imports – counting metrics
# ---------------------------------------------------------------------------
from prompts.counting.branchiness import (
    SYSTEM_PROMPT as BRANCH_SYS,
    USER_PROMPT_TEMPLATE as BRANCH_USR,
)
from prompts.counting.backtracking import (
    SYSTEM_PROMPT as BT_SYS,
    USER_PROMPT_TEMPLATE as BT_USR,
)
from prompts.counting.uncertainty_markers import (
    SYSTEM_PROMPT as UM_SYS,
    USER_PROMPT_TEMPLATE as UM_USR,
)
from prompts.counting.prior import (
    SYSTEM_PROMPT as PRIOR_SYS,
    USER_PROMPT_TEMPLATE as PRIOR_USR,
)
from prompts.counting.differential_elimination import (
    SYSTEM_PROMPT as DELIM_SYS,
    USER_PROMPT_TEMPLATE as DELIM_USR,
)

# ---------------------------------------------------------------------------
# Prompt imports – coverage metrics
# ---------------------------------------------------------------------------
from prompts.coverage.observation_coverage import (
    SYSTEM_PROMPT as OC_SYS,
    USER_PROMPT_TEMPLATE as OC_USR,
)

# ---------------------------------------------------------------------------
# Prompt imports – graph metrics
# ---------------------------------------------------------------------------

from prompts.graph_structure.rationale_graph import (
    SYSTEM_PROMPT as RG_SYS,
    USER_PROMPT_TEMPLATE as RG_USR,
)

# ---------------------------------------------------------------------------
# Prompt imports – binary metrics
# ---------------------------------------------------------------------------
from prompts.binary.evidence_explanation_directionality import (
    SYSTEM_PROMPT as EED_SYS,
    USER_PROMPT_TEMPLATE as EED_USR,
)

# ---------------------------------------------------------------------------
# Prompt imports – score-based metrics
# ---------------------------------------------------------------------------
from prompts.scorebased.evidence_explanation_directionality_scorebased import (
    SYSTEM_PROMPT as EEDS_SYS,
    USER_PROMPT_TEMPLATE as EEDS_USR,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRICS: dict[str, BaseMetric] = {
    # ── Counting metrics ────────────────────────────────────────────────────
    "branchiness": CountingMetric(
        name="branchiness",
        description="Exploring multiple distinct reasoning paths or hypotheses.",
        system_prompt=BRANCH_SYS,
        user_prompt_template=BRANCH_USR,
    ),
    "backtracking": CountingMetric(
        name="backtracking",
        description="Explicit self-correction or revision of a previous reasoning step.",
        system_prompt=BT_SYS,
        user_prompt_template=BT_USR,
    ),
    "uncertainty_markers": CountingMetric(
        name="uncertainty_markers",
        description="Count of individual probabilistic/hedging words and phrases in the reasoning trace.",
        system_prompt=UM_SYS,
        user_prompt_template=UM_USR,
    ),
    "prior": CountingMetric(
        name="prior",
        description="Instances where the model considers prior probabilities or base rates in its reasoning.",
        system_prompt=PRIOR_SYS,
        user_prompt_template=PRIOR_USR,
    ),
    "differential_elimination": CountingMetric(
        name="differential_elimination",
        description="Count of explicit eliminations/refutations of alternative hypotheses.",
        system_prompt=DELIM_SYS,
        user_prompt_template=DELIM_USR,
    ),
    # ── Binary metrics ──────────────────────────────────────────────────────
    "evidence_explanation_directionality": BinaryMetric(
        name="evidence_explanation_directionality",
        description="Awareness that abduction runs from evidence to explanation, not reverse.",
        system_prompt=EED_SYS,
        user_prompt_template=EED_USR,
    ),
    # ── Score-based metrics ─────────────────────────────────────────────────
    "evidence_explanation_directionality_scorebased": ScoreBasedMetric(
        name="evidence_explanation_directionality_scorebased",
        description=(
            "Graded directionality score (0.0 / 0.5 / 1.0): how well the reasoning chain "
            "respects the abductive direction from observations to explanation."
        ),
        system_prompt=EEDS_SYS,
        user_prompt_template=EEDS_USR,
    ),
    # ── Coverage metrics ────────────────────────────────────────────────────
    "observation_coverage": CoverageMetric(
        name="observation_coverage",
        description="Proportion of specific observation details explicitly accounted for by the chosen hypothesis.",
        system_prompt=OC_SYS,
        user_prompt_template=OC_USR,
    ),
    # ── Graph metrics ─────────────────────────────────────────────────────
    "rationale_graph": RationaleGraphMetric(
        name="rationale_graph",
        description="Text-grounded directed rationale graph and graph-structure statistics.",
        system_prompt=RG_SYS,
        user_prompt_template=RG_USR,
    ),
}


def get_active_metrics(active: list[str] | None = None) -> dict[str, BaseMetric]:
    """Return only the metrics whose names appear in ``active``.

    If ``active`` is empty or None, all registered metrics are returned.
    """
    if not active:
        return dict(METRICS)
    return {name: m for name, m in METRICS.items() if name in active}
