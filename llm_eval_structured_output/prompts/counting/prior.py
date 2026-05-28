"""
prompts/counting/prior.py
--------------------------
Prompt for the Prior counting metric.

This metric extracts instances where the model explicitly considers prior
probabilities or base rates in its reasoning. A prior is a pre-existing
probability or belief that is taken into account before considering new
observations or evidence.

For example, in medical reasoning: "Given that this type of cancer occurs
in only 2% of the population, even with these symptoms, the probability
remains relatively low." The 2% base rate is a prior being considered.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Extract priors only when they are clearly introduced as part of the model's inferential reasoning, rather than simply restating historical patient data."
    ),
    "art": (
        "Do not count priors that merely restate provided hypotheses; count inferential base-rate/plausibility reasoning."
    ),
    "strategyqa": (
        "Look for world-knowledge priors and typicality assumptions used to guide inference."
    ),
    "copa_guess_effect": (
        "Look for priors about typical everyday cause-effect relationships."
    ),
    "defeasible_nli": (
        "Look for priors about typical premise-hypothesis relations and default expectations."
    ),
    "goemotion": (
        "Look for priors about emotions typically associated with contexts, when explicitly used."
    ),
    "musr": (
        "Look for priors about typical narrative behavior, motives, or scenario patterns."
    ),
    "neulr_abductive": (
        "Look for priors used to justify why one hypothesis is more plausible than another."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}
INCLUDE_FEW_SHOT: bool = False
INCLUDE_DATASET_SPECIFIC_NOTES: bool = True

SYSTEM_PROMPT = """\
You are an expert analyst evaluating AI-generated reasoning traces.

## What is a Prior?

A prior (or prior probability / base rate) is a pre-existing probability,
frequency, or background knowledge about how common or likely something is
in general, *before* considering the specific observations at hand. The
reasoner uses this prior to adjust their final judgment.

Your task is to identify every instance where the model explicitly brings
in such prior information to inform its reasoning.

## Categories of priors to extract

### 1. Population base rates
Statistical information about how common a condition, event, or outcome is
in a relevant population.
Examples: "This disease affects 1 in 10,000 people", "The prevalence of
this condition is approximately 5%", "This is a rare disorder", "This is
a common occurrence in this age group".

### 2. Prior probabilities from domain knowledge
General knowledge about likelihoods that the model brings to bear on the
problem, not derived from the specific observations.
Examples: "Most patients with these symptoms have condition X", "Typically,
this type of failure is caused by Y", "In general, Z is more likely than W".

### 3. Comparative likelihoods
Explicit comparisons of how likely different possibilities are, based on
background knowledge rather than the specific evidence.
Examples: "X is far more common than Y", "This explanation is more probable
a priori", "Without specific evidence, we would expect Z".

### 4. Reference to general tendencies or patterns
References to what "usually" or "typically" happens, used as a prior to
guide reasoning.
Examples: "Usually, this symptom indicates...", "Typically, patients with
this profile...", "In most cases like this...".

### 5. Explicit Bayesian-style reasoning
Cases where the model explicitly weighs prior probability against new
evidence.
Examples: "Even though the test is positive, given the low base rate...",
"The prior probability is low, so we need strong evidence...", "Combining
the prior with these observations...".

## Extraction rules

- Extract **each distinct prior consideration** as a separate example.
- The `excerpt` must be a **short, direct quote** from the text that shows
  the model referencing prior information (≤ 25 words of context).
- The `explanation` must identify the type of prior (from the categories
  above) and briefly explain what prior probability or base rate is being
  referenced.
- If the model mentions the same prior multiple times in different parts
  of the reasoning, extract each occurrence separately.
- Do **not** paraphrase or alter the quoted text.

## What does NOT count as a prior

- Conclusions drawn *only* from the specific observations in the problem
  (these are posterior inferences, not priors).
- General knowledge that doesn't involve probability or frequency (e.g.,
  "The heart pumps blood" is a fact, not a prior).
- Hypotheses generated during reasoning without reference to their general
  likelihood.
- The model's own uncertainty expressions (e.g., "I think", "probably") —
  those are captured by the `uncertainty_markers` metric.
- Restatements of information given in the problem prompt.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## JSON output format

Return ONLY valid JSON with this structure:
{
  "overall_analysis": "Brief analysis of prior probability usage in this reasoning trace",
  "examples": [
    {
      "excerpt": "Quote of the prior probability consideration from the reasoning trace",
      "explanation": "Type of prior and what probability/frequency is being referenced"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Extract every instance where the model considers a prior probability or
base rate in the following reasoning trace. Return one entry per prior
consideration.

<reasoning_trace>
{text}
</reasoning_trace>
"""
