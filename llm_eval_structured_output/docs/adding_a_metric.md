# How to Add a New Metric

There are two paths. Follow **Path A** if your metric fits the existing binary (yes/no), counting (example-extraction), coverage (per-detail coverage + score), graph (rationale-graph extraction + scalar computation), or score-based (graded scalar judgment) patterns. Follow **Path B** if you need fundamentally different LLM output structure or evaluation logic.

---

## Path A — Adding a Binary, Counting, Coverage, Graph, or Score-based Metric

This is the normal path. You only touch two things: a new prompt file and one registration entry.

### What is the difference?

| Type | What the LLM outputs | `detected` field | `examples` field |
|---|---|---|---|
| **Binary** | `detected` bool + `reasoning` + `evidence` quote | Direct from LLM | Empty (or one item if evidence exists) |
| **Counting** | `overall_analysis` string + list of `{excerpt, explanation}` pairs | `true` when at least one example found | All extracted examples |
| **Coverage** | `observation_details` list + `overall_analysis` | Derived in code (true only when score = 1.0) | List of per-detail `{detail, addressed, evidence}` dicts |
| **Graph** | `general_reasoning` + `vertices` + `edges` | Derived in code (`true` when at least one vertex exists) | Graph entities (vertices and edges) |
| **Score-based** | score field + brief reasoning analysis | Derived in code from the score | Usually empty |

---

### Step 1 — Create the prompt file

**Binary** → create `prompts/binary/your_metric_name.py`  
**Counting** → create `prompts/counting/your_metric_name.py`  
**Coverage** → create `prompts/coverage/your_metric_name.py`  
**Graph** → create `prompts/graph_structure/your_metric_name.py`  
**Score-based** → create `prompts/scorebased/your_metric_name.py`

Both files must export exactly these two names:

```python
SYSTEM_PROMPT: str         # full instructions for the judge LLM
USER_PROMPT_TEMPLATE: str  # uses {text} and optionally {dataset}
```

#### Binary prompt template

```python
"""
prompts/binary/your_metric_name.py
"""

SYSTEM_PROMPT = """\
You are an expert reasoning analyst.

## What is <Your Metric>?
<Define the phenomenon clearly.>

## What COUNTS (detected = true)
<Bullet list of clear positive cases.>

## What does NOT count (detected = false)
<Bullet list of common false-positive traps.>

## Dataset-specific notes (optional)
If the dataset is MedQA: ...
If the dataset is ART: ...
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for <Your Metric>.

<reasoning_trace>
{text}
</reasoning_trace>
"""
```

The LLM response is validated against `BinaryResponse` (defined in `metrics/binary.py`), which has three fields: `detected` (bool), `reasoning` (str), and `evidence` (str).  Your prompt should instruct the model to produce all three.  The OpenAI Structured Outputs API enforces the schema automatically — no additional format instructions are needed.

#### Counting prompt template

```python
"""
prompts/counting/your_metric_name.py
"""

SYSTEM_PROMPT = """\
You are an expert reasoning analyst.

## What is <Your Metric>?
<Define the phenomenon.>

## What COUNTS as an example
<Numbered list of concrete extraction criteria.>

## What does NOT count
<Common false positives.>

## Dataset-specific notes (optional)
If the dataset is MedQA: ...
If the dataset is ART: ...
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for <Your Metric> and extract concrete examples.

<reasoning_trace>
{text}
</reasoning_trace>
"""
```

The LLM response is validated against `CountingResponse` (defined in `metrics/counting.py`), which has two fields: `overall_analysis` (str) and `examples` (list of `{excerpt, explanation}` items). The schema also accepts legacy `text` on input for backward compatibility, but new prompts should emit `excerpt`. The OpenAI Structured Outputs API enforces the schema automatically — no additional format instructions are needed.

#### Coverage prompt template

```python
"""
prompts/coverage/your_metric_name.py
"""

SYSTEM_PROMPT = """\
You are an expert evaluator of abductive reasoning traces.

Extract every specific detail in the observation and, for each detail, decide
whether the reasoning trace explicitly connects it to the chosen hypothesis.

Return:
- observation_details: list of {detail, addressed, evidence}
- overall_analysis: brief synthesis
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace.

<reasoning_trace>
{text}
</reasoning_trace>
"""
```

The LLM response is validated against `ObservationCoverageResponse` (defined in `metrics/coverage.py`), which has two fields: `observation_details` (list of per-detail objects) and `overall_analysis` (str).

#### Score-based prompt template

```python
"""
prompts/scorebased/your_metric_name.py
"""

SYSTEM_PROMPT = """\
You are an expert reasoning analyst.

## What is <Your Metric>?
<Define the graded phenomenon clearly.>

## Scoring rubric
- 1.0: <high score definition>
- 0.5: <mixed/partial definition>
- 0.0: <absent or failed definition>
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for <Your Metric> and assign a score.

<reasoning_trace>
{text}
</reasoning_trace>
"""
```

The LLM response is validated against a score-based response model such as `ScoreBasedResponse` in `metrics/scorebased.py`. For the existing directionality score metric, the schema fields are `reasoning_analysis` and `directionality_score`, and the output score is snapped to the allowed set in code.

---

### Step 2 — Register the metric in `metrics/registry.py`

Open [`metrics/registry.py`](../metrics/registry.py) and make **three edits**:

**1. Import the prompts** (add to the relevant import section at the top):

```python
# Binary:
from prompts.binary.your_metric_name import (
    SYSTEM_PROMPT as YM_SYS,
    USER_PROMPT_TEMPLATE as YM_USR,
)

# OR Counting:
from prompts.counting.your_metric_name import (
    SYSTEM_PROMPT as YM_SYS,
    USER_PROMPT_TEMPLATE as YM_USR,
)

# OR Coverage:
from prompts.coverage.your_metric_name import (
    SYSTEM_PROMPT as YM_SYS,
    USER_PROMPT_TEMPLATE as YM_USR,
)

# OR Graph:
from prompts.graph_structure.your_metric_name import (
    SYSTEM_PROMPT as YM_SYS,
    USER_PROMPT_TEMPLATE as YM_USR,
)

# OR Score-based:
from prompts.scorebased.your_metric_name import (
    SYSTEM_PROMPT as YM_SYS,
    USER_PROMPT_TEMPLATE as YM_USR,
)
```

**2. Add an entry to the `METRICS` dict**:

```python
# Binary:
"your_metric_name": BinaryMetric(
    name="your_metric_name",
    description="One-line description shown in reports.",
    system_prompt=YM_SYS,
    user_prompt_template=YM_USR,
),

# OR Counting:
"your_metric_name": CountingMetric(
    name="your_metric_name",
    description="One-line description shown in reports.",
    system_prompt=YM_SYS,
    user_prompt_template=YM_USR,
),

# OR Coverage:
"your_metric_name": CoverageMetric(
    name="your_metric_name",
    description="One-line description shown in reports.",
    system_prompt=YM_SYS,
    user_prompt_template=YM_USR,
),

# OR Graph:
"your_metric_name": RationaleGraphMetric(
    name="your_metric_name",
    description="One-line description shown in reports.",
    system_prompt=YM_SYS,
    user_prompt_template=YM_USR,
),

# OR Score-based:
"your_metric_name": ScoreBasedMetric(
    name="your_metric_name",
    description="One-line description shown in reports.",
    system_prompt=YM_SYS,
    user_prompt_template=YM_USR,
),
```

**3. Make sure it is active** in [`config.py`](../config.py):

```python
ACTIVE_METRICS = ["uncertainty_markers", "your_metric_name", ...]  # add your metric
# or leave the list empty to run ALL registered metrics
ACTIVE_METRICS = []
```

That's it. The pipeline picks it up automatically — no changes needed to `evaluator.py`, `main.py`, or the reporting layer.

---

## Path B — Adding a Completely New Metric Type

Use this when `BinaryMetric` and `CountingMetric` are not enough — for example, you need a numeric score, a multi-label output, a multi-turn LLM call, or non-LLM computation.

You must touch **four** things.

---

### Step 1 — Define the Pydantic response schema

This is the structured contract between your code and the LLM.  The OpenAI Structured Outputs API uses this schema to enforce the response shape — the model is guaranteed to return valid JSON that matches your schema.  Create the schema in a new file under `metrics/`:

```python
# metrics/your_type.py
from pydantic import BaseModel, Field

class YourResponse(BaseModel):
    score: float = Field(..., description="Numeric score between 0 and 1.")
    label: str   = Field(..., description="Dominant category detected.")
    reasoning: str = Field(..., description="Step-by-step justification.")
    # Add whatever fields your metric needs.
```

The `description` on each field is important — it becomes part of the JSON schema sent to the model, helping it understand what to produce for each field.

---

### Step 2 — Write the metric class

Your class must inherit from `BaseMetric` ([`metrics/base.py`](../metrics/base.py)) and implement everything listed below.

```python
# metrics/your_type.py (continued)
from metrics.base import BaseMetric, MetricResult
import llm_client

class YourTypeMetric(BaseMetric):

    # ── Required class-level attribute ──────────────────────────────────────
    metric_type = "your_type"   # new string identifier for your category

    # ── Constructor ─────────────────────────────────────────────────────────
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        user_prompt_template: str,
        # add any extra params your type needs, e.g. score_threshold: float = 0.5
    ) -> None:
        self.name = name
        self.description = description
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template
        # store extra params as self attributes

    # ── Required: schema property ────────────────────────────────────────────
    @property
    def schema(self) -> type[YourResponse]:
        """Returned schema is passed to llm_client.ask_llm as response_schema."""
        return YourResponse

    # ── Required: evaluate() ────────────────────────────────────────────────
    def evaluate(
        self,
        text: str,
        *,
        dataset: str = "unknown",
        problem_id: str = "N/A",
        checkpoint: str = "N/A",
        run_id: str | None = None,
    ) -> MetricResult:
        """
        Must always return a MetricResult, even on error.

        MetricResult fields:
            metric_name  – always set to self.name
            detected     – bool: was the phenomenon present? (required)
            reasoning    – str: model's justification (required)
            examples     – list[dict{"excerpt", "explanation"}]: evidence items
            score        – float|None: optional numeric score (e.g. coverage proportion)
            tokens       – dict: token usage for this LLM call (input/output, etc)
            error        – str: non-empty only on failure
            raw          – dict: the full LLM payload for logging/debugging
        """
        # 1. Guard against empty input — always return MetricResult, never raise
        if not isinstance(text, str) or not text.strip():
            return MetricResult(metric_name=self.name, error="Empty input.")

        # 2. Truncate to stay within context limits (same threshold as other metrics)
        trimmed = text[:15_000] + "\n…(truncated)" if len(text) > 15_000 else text

        # 3. Build prompts
        user_prompt = self._user_prompt_template.format(text=trimmed, dataset=dataset)

        # 4. Call the LLM — response is parsed and validated by the SDK
        payload = llm_client.ask_llm(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            response_schema=YourResponse,
            dataset=dataset,
            problem_id=problem_id,
            metric_type=self.name,
            checkpoint=checkpoint,
            run_id=run_id,
        )

        # 5. Extract fields from the returned dict
        score: float   = float(payload.get("score", 0.0))
        label: str     = payload.get("label", "")
        reasoning: str = payload.get("reasoning", "")

        # 6. Decide what "detected" means for your metric
        detected = score > 0.5  # replace with your own logic

        # 7. Build and return MetricResult
        return MetricResult(
            metric_name=self.name,
            detected=detected,
            reasoning=reasoning,
            examples=[],   # populate with {"excerpt": ..., "explanation": ...} dicts if relevant
            raw=payload,
        )
```

> **Why Pydantic + Structured Outputs?**  
> `llm_client.ask_llm` calls `client.chat.completions.parse` with `response_format=YourResponse`.  The OpenAI SDK deserializes the response directly into your Pydantic model — no regex, no XML, no post-processing.  If the model returns malformed output, the SDK raises an error and the pipeline's retry loop handles it.

---

### Step 3 — Create the prompt file

Same as Path A. Create a new subfolder for your type:

```
prompts/
└── your_type/
    ├── __init__.py              ← empty file, required for Python imports
    └── your_metric_name.py     ← SYSTEM_PROMPT + USER_PROMPT_TEMPLATE
```

Use `{text}` and optionally `{dataset}` as the only placeholders in `USER_PROMPT_TEMPLATE`.

---

### Step 4 — Register in `metrics/registry.py`

```python
from metrics.your_type import YourTypeMetric
from prompts.your_type.your_metric_name import (
    SYSTEM_PROMPT as YM_SYS,
    USER_PROMPT_TEMPLATE as YM_USR,
)

METRICS: dict[str, BaseMetric] = {
    # ... existing metrics ...
    "your_metric_name": YourTypeMetric(
        name="your_metric_name",
        description="One-line description.",
        system_prompt=YM_SYS,
        user_prompt_template=YM_USR,
    ),
}
```

Then make sure it is active in [`config.py`](../config.py) by adding it to `ACTIVE_METRICS` (or leave `ACTIVE_METRICS = []` to run all registered metrics).

---

## Summary Checklist

| | Path A — Binary | Path A — Counting | Path A — Coverage | Path A — Graph | Path A — Score-based | Path B — New Type |
|---|---|---|---|---|---|---|
| New prompt file | `prompts/binary/name.py` | `prompts/counting/name.py` | `prompts/coverage/name.py` | `prompts/graph_structure/name.py` | `prompts/scorebased/name.py` | `prompts/your_type/name.py` + `__init__.py` |
| New `__init__.py` in prompt folder | Not needed | Not needed | Not needed | Not needed | Not needed | **Required** |
| New metric class file | Not needed | Not needed | Not needed (`CoverageMetric` exists) | Not needed (`RationaleGraphMetric` exists) | Not needed (`ScoreBasedMetric` exists) | `metrics/your_type.py` |
| Pydantic response schema | `BinaryResponse` (already exists) | `CountingResponse` (already exists) | `ObservationCoverageResponse` (already exists) | `RationaleGraphResponse` (already exists) | `ScoreBasedResponse` (already exists) | **You define `YourResponse`** |
| Inherit `BaseMetric` | Done by `BinaryMetric` | Done by `CountingMetric` | Done by `CoverageMetric` | Done by `RationaleGraphMetric` | Done by `ScoreBasedMetric` | **You must inherit it** |
| Declare `metric_type` | Already `"binary"` | Already `"counting"` | Already `"coverage"` | Already `"graph"` | Already `"scorebased"` | **You must set a new string** |
| Implement `schema` property | Already done | Already done | Already done | Already done | Already done | **You must implement it** |
| Implement `evaluate()` | Already done | Already done | Already done | Already done | Already done | **You must implement it, always return `MetricResult`** |
| Format instructions in system prompt | Not needed (SDK enforces schema) | Not needed (SDK enforces schema) | Not needed (SDK enforces schema) | Not needed (SDK enforces schema) | Not needed (SDK enforces schema) | Not needed (SDK enforces schema) |
| Import in `registry.py` | Prompts only | Prompts only | Prompts only | Prompts only | Prompts only | Class + prompts |
| Add to `METRICS` dict | `BinaryMetric(...)` | `CountingMetric(...)` | `CoverageMetric(...)` | `RationaleGraphMetric(...)` | `ScoreBasedMetric(...)` | `YourTypeMetric(...)` |
| Enable in `config.py` | Add to `ACTIVE_METRICS` (or set `ACTIVE_METRICS = []` for all) | Add to `ACTIVE_METRICS` (or set `ACTIVE_METRICS = []` for all) | Add to `ACTIVE_METRICS` (or set `ACTIVE_METRICS = []` for all) | Add to `ACTIVE_METRICS` (or set `ACTIVE_METRICS = []` for all) | Add to `ACTIVE_METRICS` (or set `ACTIVE_METRICS = []` for all) | Add to `ACTIVE_METRICS` (or set `ACTIVE_METRICS = []` for all) |
