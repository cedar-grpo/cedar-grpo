"""
prompts/binary/evidence_explanation_directionality.py
---------------------------------------------------
Prompt for the Evidence-Explanation Directionality Awareness binary metric.

Evidence-Explanation Directionality: The model demonstrates awareness that
abduction runs from evidence to explanation (not explanation to evidence,
as in prediction/deduction). The model distinguishes what is observed from
what needs to be explained.

The LLM answers yes/no and explains why.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "The 'Problem' text provides the patient's symptoms (the evidence). "
        "Check that reasoning starts from this clinical evidence and seeks the condition (explanation). "
        "Declaring a diagnosis upfront and checking if symptoms match is backward."
    ),
    "art": (
        "Check that the reasoning starts from 'Observation 1' and 'Observation 2' "
        "and seeks the connecting event/hypothesis, rather than assuming a hypothesis first and back-fitting observations."
    ),
    "strategyqa": (
        "The 'Evidence' and 'Question' give the fixed facts. "
        "Check that reasoning starts from these facts and logically supports the final YES/NO conclusion."
    ),
    "copa_guess_effect": (
        "The 'Cause' is the premise observation. Check that the reasoning uses this Cause "
        "to evaluate which 'Option' is the best effect, rather than deducing from an Option."
    ),
    "defeasible_nli": (
        "The 'Premise', 'Hypothesis', and 'Update' are the evidence. Check that reasoning "
        "evaluates the logical impact of the Update starting from this given text."
    ),
    "goemotion": (
        "Look for reasoning that extracts cues from the 'Text' before selecting the emotion label."
    ),
    "musr": (
        "The 'Context' provides scenario details. Check that reasoning builds explanations "
        "from these details, rather than assuming a conclusion and back-fitting support."
    ),
    "neulr_abductive": (
        "Check that reasoning uses 'Logical Rules and Known Facts' to find the 'Missing Fact' (Target Conclusion), "
        "rather than working backwards from a chosen fact."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}
INCLUDE_FEW_SHOT: bool = False
INCLUDE_DATASET_SPECIFIC_NOTES: bool = True

SYSTEM_PROMPT = """\
You are an expert evaluator of abductive reasoning traces.

## What is Evidence-Explanation Directionality Awareness?

Abduction (inference to the best explanation) moves FROM the evidence/observations
TO the explanatory hypothesis. This is different from deduction (hypothesis → prediction)
or induction (examples → generalization).

A reasoning trace demonstrates this awareness when:
1. It explicitly separates what is OBSERVED from what needs to be EXPLAINED
2. It treats the observations as GIVEN and tries to find their CAUSE
3. It doesn't treat the hypothesis as a given and work backward to find "evidence"
4. It shows awareness that we're finding the best explanation for given facts

## Examples of correct directionality (detected = true)

- Explicitly separates known facts from explanatory goals:
  "Given these observations (X, Y), we need to explain why they occurred."
- Shows awareness of moving from observation to explanation:
  "The observation is that A happened. What could explain this?"
- Distinguishes between what's given vs what's inferred:
  "We know observation X is true. To explain it, hypothesis Y would need to be true."
- Doesn't confuse prediction with explanation:
  "If we assume Y caused X, then we would predict..." (vs assuming Y and saying "we have evidence")

## Examples of incorrect or missing directionality (detected = false)

- Confuses abduction with deduction:
  "We know hypothesis Y is true, therefore evidence X must be true."
- Treats the hypothesis as given and finds "supporting evidence":
  "Since Option A is correct, we can see evidence for it in the observation."
- Works from hypothesis to observation instead of observation to hypothesis:
  "If Option B were true, then X would be true - and X is true, so B is correct"
  (This is deduction, not abduction)
- Doesn't distinguish between what is observed vs what is inferred.

## Important

The key is whether the model treats observations as fixed and seeks explanations
for them, rather than treating hypotheses as fixed and finding "evidence" for them.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Set `detected` to true only when the trace clearly reasons from given evidence toward explanation.
- Set `detected` to false when it assumes a hypothesis first and then back-fits evidence.
- Keep `reasoning` focused on directionality cues in the trace.
- Set `evidence` to a direct quote that best demonstrates directionality (or its absence).
- Use empty `evidence` only when `detected` is false and no positive directionality quote applies.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "detected": true,
  "reasoning": "Step-by-step explanation of why evidence-explanation directionality is present/absent",
  "evidence": "Direct quote from the text supporting the decision (leave empty if detected is false)"
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Evidence-Explanation Directionality
Awareness (does the model move from evidence to explanation, not the reverse?).

**Observations / Evidence:**
<observations>
{full_input}
</observations>

**Model's Reasoning Chain:**
<reasoning_trace>
{text}
</reasoning_trace>
"""
