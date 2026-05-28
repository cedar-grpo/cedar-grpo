"""
prompts/counting/uncertainty_markers.py
----------------------------------------
Prompt for the Uncertainty Markers counting metric.

Unlike the binary `uncertainty_language` metric (which answers yes/no on whether
the reasoning hedges at all), this metric extracts *every individual probabilistic
word or phrase* as a separate example.  The count of examples is then used as a
quantitative measure of how densely hedged the reasoning trace is.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Extract markers only when the model expresses uncertainty in its own reasoning or conclusions, not when it is merely listing probabilistic symptoms."
    ),
    "art": (
        "Focus on uncertainty while reasoning about which hypothesis better explains observations."
    ),
    "strategyqa": (
        "Look for uncertainty in multi-step reasoning and evidence-to-conclusion transitions."
    ),
    "copa_guess_effect": (
        "Focus on uncertainty in causal reasoning and option comparison."
    ),
    "defeasible_nli": (
        "Look for uncertainty indicating potentially defeasible or non-certain inference relations."
    ),
    "goemotion": (
        "Look for uncertainty in label selection; avoid counting task-domain terms that are not hedging."
    ),
    "musr": (
        "Look for uncertainty in narrative interpretation and conclusion drawing."
    ),
    "neulr_abductive": (
        "Focus on uncertainty while weighing competing abductive explanations."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}
INCLUDE_FEW_SHOT: bool = False
INCLUDE_DATASET_SPECIFIC_NOTES: bool = True

SYSTEM_PROMPT = """\
You are an expert linguistic analyst evaluating AI-generated reasoning traces.

## What is an Uncertainty Marker?

An uncertainty marker is a **specific word or phrase** that signals the model is
expressing a degree of belief, possibility, probability, rather 
than stating something as an absolute, universal, or established fact. Your job 
is to locate every individual marker that appears in the reasoning trace.

## Categories of uncertainty markers to extract

### 1. Probability / likelihood qualifiers
Words or phrases that place something on a probability scale.
Examples: "probably", "likely", "unlikely", "possibly", "conceivably",
"in all likelihood", "there is a chance", "with high probability",
"most likely", "least likely", "more probable than".

### 2. Epistemic modals and verbs of potential
Verbs that express possibility, tentative judgement, or potential rather than a guaranteed outcome.
Examples: "might", "may", "could", "would", "can" (when used as 'has the potential to', e.g., "can help"), 
"seems to", "appears to", "tends to".

### 3. Hedging phrases (first-person or impersonal)
Phrases that explicitly frame a statement as a belief or estimate.
Examples: "I believe", "I think", "I suspect", "I'm not certain but",
"it is possible that", "it seems that", "it appears that",
"this suggests","this may indicate".

### 4. Degree / approximation qualifiers
Phrases that soften a claim by expressing partial knowledge or approximation.
Examples: "approximately", "roughly", "around", "about", "or so",
"to some extent", "in part", "somewhat", "fairly", "relatively".

### 5. Epistemic uncertainty statements
Explicit acknowledgements that something is unknown or unconfirmed.
Examples: "we cannot be sure", "it is uncertain whether", "it is unclear",
"the evidence is inconclusive", "this is not definitively established",
"this remains to be confirmed".

### 6. Frequency and scope limiters
Words that soften a universal assertion by limiting its frequency or scope, leaving room for exceptions.
Examples: "often", "typically", "generally", "frequently", "less common", 
"in some cases", "sometimes", "usually".

## What NOT to Extract (False Positives)

Do **NOT** extract the following linguistic constructs, as they do not represent epistemic uncertainty:

- **Objective risk or statistical metrics:** Mentions of "risk" describe an objective state or classification, not the speaker's doubt. (e.g., Do NOT extract "high risk", "reduces the risk").
- **Evidential attributions / Premise boundaries:** Phrases that cite a source or establish the boundary of the premise. (e.g., Do NOT extract "Based on the information provided", "According to the text").
- **Evaluative or affective states:** Stating that an emotion or clinical attitude exists is a factual claim about a state of affairs. (e.g., Do NOT extract "There is concern", "It is alarming").

## Extraction rules

- Extract **each individual marker occurrence** as a separate example, even if
  the same word appears multiple times. Every occurrence is its own entry.
- The `excerpt` must be a **short, direct quote** from the text — ideally the
  single word or short phrase itself, plus just enough surrounding context
  (≤ 15 words) to make it readable.
- The `explanation` must name the marker category (from the list above) and
  briefly state what belief, probability, or limitation the marker expresses in context.
- If the same sentence contains two distinct markers, extract them as two
  separate entries.
- Do **not** paraphrase or alter the quoted text.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## JSON output format

Return ONLY valid JSON with this structure:
{
  "overall_analysis": "Brief analysis of uncertainty markers density in this reasoning trace",
  "examples": [
    {
      "excerpt": "Quote of the uncertainty marker from the reasoning trace",
      "explanation": "Category and meaning of this uncertainty marker"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Extract every individual uncertainty marker from the following reasoning trace.
Return one entry per marker occurrence.

<reasoning_trace>
{text}
</reasoning_trace>
"""
