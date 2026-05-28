# Metric Definitions

This document defines each metric type and every metric currently implemented in the pipeline.  For instructions on adding new metrics see [adding_a_metric.md](adding_a_metric.md).

> **Controlling which metrics run:** edit `ACTIVE_METRICS` in `config.py`.  List the metric names you want; leave the list empty (`[]`) to run all registered metrics.

---

## Metric Types

### Binary Metric

A binary metric asks the judge LLM a single yes/no question about the reasoning trace.

- **Output:** `detected` (true/false), a `reasoning` explanation, and an `evidence` quote (empty when `detected` is false).
- **Count used in reports:** 1 if detected, 0 otherwise (one value per trace).
- **Use when:** you want to know *whether* a phenomenon exists in the trace, not *how many times*.

### Counting Metric

A counting metric asks the judge LLM to extract every individual occurrence of a phenomenon as a concrete example (an excerpt quote + explanation pair).

- **Output:** An `overall_analysis` string and a list of `{excerpt, explanation}` items.
- **Count used in reports:** `len(examples)` — the raw number of extracted occurrences.  This can be normalized per 100 words in the `normalized/` results.
- **Use when:** you want to measure the *density* or *frequency* of a phenomenon, not just its presence.

### Coverage Metric

A coverage metric asks the judge LLM to enumerate **all specific details** present in the observation and assess whether the reasoning trace explicitly connects each detail to the chosen hypothesis.

- **Output:** `observation_details` as a list of `{detail, addressed, evidence}` and an `overall_analysis` synthesis.
- **Score used in reports:** `addressed_count / total_details` (a float in 0.0–1.0). Aggregations typically use the mean score per dataset.
- **Use when:** you want to measure *how completely* a hypothesis accounts for the full set of observation details, not just whether it mentions them.

### Graph Metric

A graph metric asks the judge LLM to extract a directed, text-grounded rationale graph, then computes structural statistics from that graph.

- **Output:** `vertices`, `edges`, and `general_reasoning` from the LLM, plus computed scalar metrics (for example degree/depth/cycle/centrality statistics).
- **Score used in reports:** graph scalar metrics and normalized scalar metrics (per 100 words for selected keys).
- **Use when:** you want to analyze reasoning structure and connectivity patterns, not only presence/count/coverage of specific phenomena.

### Score-based Metric

A score-based metric asks the judge LLM to assign a graded scalar value to a phenomenon in the reasoning trace.

- **Output:** a score field plus a brief explanation of the judgment.
- **Score used in reports:** the scalar score itself, usually averaged per dataset or checkpoint.
- **Use when:** you want more granularity than a binary metric but less output structure than a full extraction metric.

---

## Implemented Metrics

### `uncertainty_markers` — Counting

> **How many individual probabilistic or hedging words and phrases appear in the reasoning trace?**

Where `uncertainty_language` gives a yes/no answer, `uncertainty_markers` extracts *every single occurrence* of a hedging word or phrase as its own entry.  The count gives a quantitative density measure of epistemic hedging.

Markers are grouped into five categories:
1. **Probability / likelihood qualifiers** — "probably", "likely", "possibly", "in all likelihood"
2. **Epistemic modal verbs** — "might", "may", "could", "seems to", "appears to"
3. **Hedging phrases** — "I believe", "I think", "this suggests", "this is consistent with"
4. **Degree / approximation qualifiers** — "approximately", "roughly", "to some extent", "somewhat"
5. **Epistemic uncertainty statements** — "we cannot be sure", "it is unclear", "the evidence is inconclusive"

**Does not count:** boiler-plate disclaimers, purely logical hypotheticals ("if we could assume…")

---

### `branchiness` — Counting

> **How many times does the reasoning genuinely explore multiple distinct candidate explanations for the same observation?**

Measures whether the model considers substantively different explanatory candidates rather than following a single linear chain or merely revising one candidate. Each distinct branching moment is extracted as one example.

**Positive examples:** "If diagnosis X we'd expect F… If diagnosis Y we'd expect G…", comparing two different causal interpretations of the same evidence  
**Does not count:** multiple phrasings or refinements of the same explanation, the final answer selection, brief mention of an alternative followed by immediate rejection, simple step-by-step narration, restating the given answer options

---

### `backtracking` — Counting

> **How many times does the reasoning explicitly identify an error or flaw and change direction?**

Captures deliberate self-correction: the model realises something it said or computed is wrong and reverses course.  This is distinct from `branchiness` (exploring valid alternatives).

**Positive examples:** "Wait, that's wrong", "On second thought…", "Let me re-read the problem", "I realise I forgot to account for…"  
**Does not count:** comparing two valid paths (Branchiness), a simple "however" contrast without admitting an error, the final answer selection

---

### `differential_elimination` — Counting

> **How many explicit elimination/refutation moves against alternatives appear in the reasoning trace?**

Extracts each distinct case where the model rules out an alternative hypothesis, answer option, or interpretation with an explicit reason grounded in the trace.

**Positive examples:** "We can rule out A because it contradicts symptom X", "If B were true we'd see Y, but we don't"  
**Does not count:** listing options without refuting them, pure support for the chosen option without alternative elimination

---

### `prior` — Counting

> **How many times does the reasoning explicitly invoke prior probability, typicality, or base-rate knowledge?**

Captures explicit references to what is common, rare, expected, or more probable before or alongside the case-specific evidence. This is useful when the trace relies on domain priors, typical scenarios, or general population tendencies.

**Positive examples:** "This disease is rare", "Usually this symptom indicates...", "X is more common than Y"  
**Does not count:** pure hedging language, generic facts without likelihood content, or conclusions drawn only from the specific case evidence

---

### `evidence_explanation_directionality` — Binary

> **Does the reasoning clearly move from given evidence or observations toward an explanatory conclusion, rather than assuming the conclusion and back-fitting support?**

This metric checks whether the trace respects the abductive direction from evidence to explanation. It is a presence/absence test of directional awareness, not a graded quality score.

**Positive cases:** explicit separation of observations from explanatory hypotheses, reasoning that starts from the given facts and asks what best explains them  
**Does not count:** assuming a hypothesis first and then merely verifying that it matches the evidence

---

### `evidence_explanation_directionality_scorebased` — Score-based

> **How strongly does the reasoning respect the abductive direction from evidence to explanation?**

This is the graded version of the directionality metric. The judge assigns exactly one of three scores:

- `1.0` — clear evidence → explanation reasoning
- `0.5` — mixed or ambiguous directionality
- `0.0` — backward, circular, or missing directionality

Use this when binary presence/absence is too coarse and you want a gradable measure of how well the trace follows abductive direction.

---

### `observation_coverage` — Coverage

> **What fraction of specific observation details are explicitly accounted for by the chosen hypothesis?**

Extracts an exhaustive set of observation details and marks each one as addressed or not addressed.

- **Per-detail fields:**
	- `detail` — one concrete observation fact
	- `addressed` — whether the trace explicitly links that fact to the hypothesis
	- `evidence` — a direct quote showing the link (empty if `addressed = false`)

The metric score is:

$$
	ext{score} = \frac{\#\text{addressed details}}{\#\text{total details}}
$$

`detected` is set to true only when the score is 1.0 (all details addressed).

---

### `rationale_graph` — Graph

> **What directed rationale graph structure is explicitly present in the reasoning trace?**

This metric extracts graph vertices and edges grounded in exact text spans, then computes structural statistics including out-degree, in-degree, depth, cycle count, weakly connected components, and betweenness centrality summaries.

- **Core extraction fields:**
	- `vertices` — list of `{vertex_id, label, description, text_correspondence}`
	- `edges` — list of `{source_vertex_label, target_vertex_label, edge_label, description, text_correspondence}`
	- `general_reasoning` — short explanation of the extracted graph
- **Computed scalar metrics:** degree/depth/cycle/component/centrality metrics in `scalar_metrics`
- **Computed normalized metrics:** selected metrics in `normalized_scalar_metrics` (per 100 words)

---

## Relationship Between Metrics

```
Reasoning trace phenomenon
│
├── What graph structure does reasoning express?   → rationale_graph       (graph)
│
├── What fraction of details are addressed?        → observation_coverage  (coverage)
│
├── How densely does it hedge?                     → uncertainty_markers   (counting)
│
├── Does it explore multiple candidate explanations? → branchiness         (counting)
│
├── Does it catch and fix its own mistakes?        → backtracking          (counting)
│
├── Does it invoke priors or base rates?           → prior                 (counting)
│
├── How many alternatives are explicitly ruled out? → differential_elimination (counting)
│
├── Is directional reasoning (evidence→explanation) shown? → evidence_explanation_directionality (binary)
│
└── How strongly is that directionality expressed? → evidence_explanation_directionality_scorebased (score-based)
```

---

## Dataset & Checkpoint Selection

Datasets and checkpoints to include in a run are controlled via `config.py`:

| Setting | Effect when empty (`[]`) | Effect when non-empty |
|---|---|---|
| `ACTIVE_DATASETS` | All dataset folders found in each checkpoint are evaluated | Only the listed dataset names are evaluated |
| `ACTIVE_METRICS` | All registered metrics are run | Only the listed metric names are run |

### `raw_model` as a checkpoint alias

A checkpoint directory named `raw_model` is automatically treated as **checkpoint-0** (the untrained baseline).  Both `checkpoint-0/` and `raw_model/` are discovered and reported identically — no manual renaming is needed.
