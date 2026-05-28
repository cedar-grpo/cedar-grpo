"""
prompts/coverage/observation_coverage.py
-----------------------------------------
Prompt for the Observation Coverage metric.

The LLM identifies every specific detail present in the observation and
annotates each one: was it explicitly accounted for by the chosen hypothesis?
The final score is the fraction of addressed details (0.0 – 1.0).
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Treat 'Problem' as observation and extract all clinical details."
    ),
    "art": (
        "Treat 'Observation 1' and 'Observation 2' together as the complete observation and Extract all details from both."
    ),
    "strategyqa": (
        "Treat both the 'Question' and 'Evidence' as the complete observation and extract all relevant facts that should be addressed."
    ),
    "copa_guess_effect": (
        "Treat premise (labeled 'Cause:') as observation and extract all relevant details for coverage checks."
    ),
    "defeasible_nli": (
        "Treat the 'Premise' (if present), 'Hypothesis', and 'Update' together as the complete observation. Extract all details across all three sections."
    ),
    "goemotion": (
        "Treat the 'Text' as the complete observation and extract salient spans that should inform emotion classification."
    ),
    "musr": (
        "Treat the 'Context' story and the 'Problem' question together as the complete observation and extract narrative details (actors, actions, timing, locations, relations)."
    ),
    "neulr_abductive": (
        "Treat the 'Logical Rules and Known Facts' and the 'Target Conclusion' together as the complete observation and extract all details across all three sections."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {}
INCLUDE_FEW_SHOT: bool = False
INCLUDE_DATASET_SPECIFIC_NOTES: bool = True

SYSTEM_PROMPT = """\
You are an expert evaluator of abductive reasoning traces.

## Your Task

Given a reasoning_trace in which a model selects one hypothesis to explain an
observation, you must:

1. **Extract every specific detail** that appears in the observation (or the
   model's description of the observation) — not just the main event, but also
   peripheral facts, contextual clues, timing details, quantities, locations,
   named entities, and any other particulars mentioned.

2. **For each detail**, decide whether the reasoning_trace *explicitly* connects
   that detail to the chosen hypothesis.  A detail is "addressed" only if the
   reasoning_trace makes a clear logical link between that detail and the hypothesis — not
   merely restating it or acknowledging it exists.

3. **Provide evidence** for every addressed detail: quote the exact short
   passage from the reasoning_trace that demonstrates the connection.

4. **Write a brief overall analysis** summarising how fully the hypothesis
   accounts for the complete observation.

## Grading criteria

- **Addressed (True)**: The trace contains a direct explanation of *why* or
  *how* the chosen hypothesis accounts for this specific detail.
- **Not addressed (False)**: The detail is present in the observation but the
  trace either ignores it, only restates it, or treats it as irrelevant without
  justification.

## Important rules

- Be exhaustive: do not skip minor or background details.
- Do not reward vague gestures.
- A hypothesis that explains the main event but ignores supporting details
  should receive a low coverage score.
- Base every judgement solely on what is written in the reasoning_trace —
  do not infer or assume anything that is not stated.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- **Enforce Atomicity:** Break down compound sentences and lists into atomic (single, indivisible) facts. 
  - *Example:* "headache and vomiting" must be split into two separate details: "headache" and "vomiting".
  - *Example:* "Kernig and Brudzinski signs are present" must be split into "Kernig sign present" and "Brudzinski sign present".
  - *Example:* Separate every single medication, vital sign, and lab value into its own item.
- Extract each atomic observation fact as one item in `observation_details`.
- Use `detail` for the observation fact text, `addressed` for explicit linkage status, and `evidence` for a supporting quote.
- Set `addressed` to true only when the reasoning_trace explicitly links the detail to the chosen hypothesis.
- If `addressed` is false, leave `evidence` as an empty string.
- Be exhaustive across main and peripheral details.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "overall_analysis": "Brief analysis of observation coverage in this reasoning_trace",
  "observation_details": [
    {
      "detail": "Specific atomic detail from the observation",
      "addressed": true,
      "evidence": "Quote from reasoning_trace showing how this detail is explained"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyse the following observation and reasoning_trace and produce the structured
observation-coverage evaluation.

<observation>
{full_input}
</observation>

<reasoning_trace>
{text}
</reasoning_trace>
"""
