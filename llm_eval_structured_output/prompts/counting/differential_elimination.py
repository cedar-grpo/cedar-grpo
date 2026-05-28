"""
prompts/counting/differential_elimination.py
--------------------------------------------
Prompt for the Differential Elimination counting metric.

This metric counts explicit elimination/refutation moves against alternative
hypotheses in the reasoning trace.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": "Count each answer choice explicitly ruled out with concrete clinical rationale.",
    "art": (
        "Count explicit eliminations of the non-chosen hypothesis or intermediate alternatives."
    ),
    "strategyqa": (
        "Count explicit rejection of competing YES/NO reasoning lines or assumptions."
    ),
    "copa_guess_effect": (
        "Count explicit elimination of the non-selected option or other causal alternatives."
    ),
    "defeasible_nli": (
        "Count explicit rejection of candidate inference relations shown inconsistent or defeasible."
    ),
    "goemotion": (
        "Count only explicit elimination of alternative emotion labels with trace-grounded justification."
    ),
    "musr": (
        "Count explicit rejection of narrative interpretations, suspects, or scenario explanations."
    ),
    "neulr_abductive": (
        "Count explicit elimination moves between competing abductive hypotheses using concrete mismatch evidence."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {
    "art": '''Shot 1
{
  "input": "Jim's action of finding something on a nearby shelf suggests that he was looking for a specific item that he needed for his project. Hypothesis 1, 'Jim found he was missing an item,' directly aligns with this scenario. Hypothesis 2, 'Jim needed a certain animal for it,' does not provide a clear connection to the observation of Jim finding something on a shelf, as animals are not typically stored on shelves for projects.",
  "output": {
    "overall_analysis": "The trace explicitly rules out one of the two ART hypotheses by showing that it does not fit the observed shelf detail.",
    "examples": [
      {
        "text": "Hypothesis 2, 'Jim needed a certain animal for it,' does not provide a clear connection to the observation of Jim finding something on a shelf",
        "explanation": "This eliminates Hypothesis 2 because it conflicts with the concrete observation about finding something on a shelf."
      }
    ]
  }
}

Shot 2
{
  "input": "Deb's desire to go shopping and her subsequent success in finding everything she needed with money left over suggests that she completed her shopping trip successfully. Hypothesis 1, that she went to a matinee movie instead, does not logically connect to the observations as it doesn't explain how she found everything she needed and had money left over. On the other hand, Hypothesis 2, that Deb had a lot of coupons, provides a plausible explanation.",
  "output": {
    "overall_analysis": "The reasoning performs an explicit elimination of the weaker ART hypothesis by showing that it cannot explain the observed outcome.",
    "examples": [
      {
        "text": "Hypothesis 1, that she went to a matinee movie instead, does not logically connect to the observations",
        "explanation": "This is a direct refutation of Hypothesis 1 based on mismatch with the shopping evidence."
      }
    ]
  }
}''',
    "copa_guess_effect": '''Shot 1
{
  "input": "When it gets dark outside, the primary reason for darkness is the absence of sunlight. Option 1, snowflakes beginning to fall, is not directly caused by darkness; it could happen but is not a typical or immediate effect of darkness. On the other hand, Option 2, the moon becoming visible, is a logical consequence of darkness.",
  "output": {
    "overall_analysis": "The trace rejects one COPA effect option by pointing out that it is not a direct consequence of the stated cause.",
    "examples": [
      {
        "text": "Option 1, snowflakes beginning to fall, is not directly caused by darkness",
        "explanation": "This explicitly eliminates Option 1 as an effect because the causal link is not supported."
      }
    ]
  }
}

Shot 2
{
  "input": "The need for irrigation implies that the farmland lacks sufficient natural water supply. A flood (Option 1) is typically an unintended consequence of excessive water release or heavy rainfall, which would not be directly caused by the need for irrigation. On the other hand, constructing a canal (Option 2) is a logical and common solution.",
  "output": {
    "overall_analysis": "The reasoning clearly rules out the non-selected causal alternative with a concrete mismatch explanation.",
    "examples": [
      {
        "text": "A flood (Option 1) is typically an unintended consequence of excessive water release or heavy rainfall, which would not be directly caused by the need for irrigation.",
        "explanation": "This eliminates Option 1 by explaining why it does not follow from the cause described."
      }
    ]
  }
}''',
    "defeasible_nli": '''Shot 1
{
  "input": "The original hypothesis suggests that PersonX yelled because they wanted to be nice. However, the update states that PersonX yelled a curse at PersonY. Yelling a curse typically implies anger or frustration rather than niceness. Therefore, the new information contradicts the initial hypothesis.",
  "output": {
    "overall_analysis": "The trace explicitly rejects the original inference after the update introduces incompatible evidence.",
    "examples": [
      {
        "text": "Therefore, the new information contradicts the initial hypothesis.",
        "explanation": "This is a direct elimination of the original hypothesis based on the updated behavior."
      }
    ]
  }
}

Shot 2
{
  "input": "The premise provides context that PersonX smells something burning, which typically would prompt them to investigate and possibly ventilate the area by opening a window. The update introduces new information that PersonX wants to join their friend in smoking marijuana. However, the desire to join in smoking marijuana might lead PersonX to avoid opening the window to maintain the atmosphere for smoking. Therefore, the update suggests that PersonX might not want to open the window due to their intention to smoke, which contradicts the initial hypothesis.",
  "output": {
    "overall_analysis": "The reasoning uses the update to rule out the earlier prediction about opening the window.",
    "examples": [
      {
        "text": "the update suggests that PersonX might not want to open the window due to their intention to smoke, which contradicts the initial hypothesis.",
        "explanation": "This eliminates the original hypothesis by identifying a concrete reason the predicted action would not occur."
      }
    ]
  }
}''',
    "goemotion": '''Shot 1
{
  "input": "The phrase 'Borrowed time is stupid' suggests a strong negative reaction to the concept of borrowed time. The word 'stupid' indicates a feeling of disbelief or frustration with something, which aligns closely with disappointment. There's no indication of humor or amusement here, so we can rule out amusement.",
  "output": {
    "overall_analysis": "The trace performs a direct elimination of an alternative emotion label using textual evidence.",
    "examples": [
      {
        "text": "There's no indication of humor or amusement here, so we can rule out amusement.",
        "explanation": "This explicitly eliminates the amusement label because the text provides no support for it."
      }
    ]
  }
}

Shot 2
{
  "input": "The phrase 'sounds an awful lot like' suggests a negative reaction and the use of 'awful' indicates strong disapproval or dislike. There's no indication of humor, excitement, or any other emotion listed, so we can rule out those. The context doesn't suggest confusion, curiosity, or any other nuanced emotion either.",
  "output": {
    "overall_analysis": "The reasoning explicitly rules out multiple competing emotion labels before settling on the best fit.",
    "examples": [
      {
        "text": "There's no indication of humor, excitement, or any other emotion listed, so we can rule out those.",
        "explanation": "This is a valid elimination move because the trace rejects alternative labels on the basis of missing textual evidence."
      },
      {
        "text": "The context doesn't suggest confusion, curiosity, or any other nuanced emotion either.",
        "explanation": "This adds a second elimination step by rejecting another set of candidate emotions."
      }
    ]
  }
}''',
    "medqa": '''Shot 1
{
  "input": "Renal artery stenosis (A) is less likely because it typically causes hypertension but not necessarily flank pain or urinary tract dilation. Benign prostatic hyperplasia (B) is unlikely as it primarily affects males over 50 years old and typically presents with lower urinary tract symptoms like difficulty urinating, not flank pain. Diabetic nephropathy (C) can cause proteinuria and kidney dysfunction but does not typically present with flank pain or ureteral dilation.",
  "output": {
    "overall_analysis": "The trace performs a sequence of explicit clinical eliminations, ruling out multiple answer choices with feature-level reasoning.",
    "examples": [
      {
        "text": "Renal artery stenosis (A) is less likely because it typically causes hypertension but not necessarily flank pain or urinary tract dilation.",
        "explanation": "This rules out choice A by contrasting its expected presentation with the case details."
      },
      {
        "text": "Benign prostatic hyperplasia (B) is unlikely as it primarily affects males over 50 years old and typically presents with lower urinary tract symptoms like difficulty urinating, not flank pain.",
        "explanation": "This eliminates choice B using demographic and symptom mismatch."
      },
      {
        "text": "Diabetic nephropathy (C) can cause proteinuria and kidney dysfunction but does not typically present with flank pain or ureteral dilation.",
        "explanation": "This rejects choice C because the expected disease profile does not match the observed findings."
      }
    ]
  }
}

Shot 2
{
  "input": "Hypoglycemia typically causes confusion and somnolence but does not explain the neurological findings like ataxia and diminished reflexes. Cerebral ischemia can cause altered mental status but would more likely present with focal neurological deficits rather than generalized findings like ataxia. Cannabis intoxication can cause somnolence and altered mental status but does not typically cause the specific neurological findings seen here.",
  "output": {
    "overall_analysis": "This MedQA trace rules out several diagnostic alternatives by checking each one against the observed neurologic pattern.",
    "examples": [
      {
        "text": "Hypoglycemia typically causes confusion and somnolence but does not explain the neurological findings like ataxia and diminished reflexes.",
        "explanation": "This eliminates hypoglycemia because it fails to account for the full exam picture."
      },
      {
        "text": "Cerebral ischemia can cause altered mental status but would more likely present with focal neurological deficits rather than generalized findings like ataxia.",
        "explanation": "This rejects cerebral ischemia by contrasting its usual presentation with the patient's findings."
      },
      {
        "text": "Cannabis intoxication can cause somnolence and altered mental status but does not typically cause the specific neurological findings seen here.",
        "explanation": "This rules out cannabis intoxication using a mismatch in expected symptoms."
      }
    ]
  }
}''',
    "musr": '''Shot 1
{
  "input": "While Rosemary has a strong motive and opportunity, her alibi makes her less likely. Harry has a strong motive, opportunity, and a recent argument with Timothy.",
  "output": {
    "overall_analysis": "The trace explicitly weakens one suspect by citing exculpatory narrative evidence.",
    "examples": [
      {
        "text": "While Rosemary has a strong motive and opportunity, her alibi makes her less likely.",
        "explanation": "This is a direct elimination move against Rosemary based on her alibi."
      }
    ]
  }
}

Shot 2
{
  "input": "Emma's bag and head are unlikely locations for a key, as they do not make practical sense for securing the key. The box initially held the key, but since it was found empty, it is less likely to be the current location. The zookeeper's office is where George moved the key for his convenience and security.",
  "output": {
    "overall_analysis": "The reasoning eliminates multiple object-location alternatives before choosing the strongest narrative location.",
    "examples": [
      {
        "text": "Emma's bag and head are unlikely locations for a key, as they do not make practical sense for securing the key.",
        "explanation": "This rules out two alternatives by appealing to narrative practicality."
      },
      {
        "text": "The box initially held the key, but since it was found empty, it is less likely to be the current location.",
        "explanation": "This explicitly eliminates the box as the current location using story evidence."
      }
    ]
  }
}

Shot 3
{
  "input": "Brenda is described as unreliable and inconsistent, which suggests she might struggle with the unpredictable nature of trash collection routes. Therefore, she should not be assigned to trash collection where such unpredictability could lead to inefficiencies.",
  "output": {
    "overall_analysis": "This MUSR team-allocation trace contains an explicit rejection of one assignment option grounded in character evidence.",
    "examples": [
      {
        "text": "Therefore, she should not be assigned to trash collection where such unpredictability could lead to inefficiencies.",
        "explanation": "This eliminates one candidate role for Brenda using concrete evidence from her profile."
      }
    ]
  }
}''',
    "neulr_abductive": '''Shot 1
{
  "input": "From Rule8, if something is ADPgOdWt0, then it is ADPAhr1Zq. This means if NPEW67G1 were ADPgOdWt0, it would also be ADPAhr1Zq, but since it is ADP7peGtC, it cannot be ADPgOdWt0. Therefore, the missing fact should clarify that NPEW67G1 is not ADPgOdWt0.",
  "output": {
    "overall_analysis": "The trace contains an explicit logical elimination of one possible property assignment.",
    "examples": [
      {
        "text": "but since it is ADP7peGtC, it cannot be ADPgOdWt0.",
        "explanation": "This directly rules out the ADPgOdWt0 alternative by appeal to the rule consequences."
      }
    ]
  }
}

Shot 2
{
  "input": "Therefore, the missing fact must be something that establishes NPqYoG7e as ADPWg1nCF, allowing us to use Rule9 to show that NPqYoG7e cannot be ADPcRkv6o and thus must be ADP2ha14i via Rule4.",
  "output": {
    "overall_analysis": "The reasoning explicitly eliminates a competing logical state as part of deriving the target conclusion.",
    "examples": [
      {
        "text": "allowing us to use Rule9 to show that NPqYoG7e cannot be ADPcRkv6o",
        "explanation": "This is a direct elimination step where one alternative predicate assignment is ruled out."
      }
    ]
  }
}''',
    "strategyqa": '''Shot 1
{
  "input": "The evidence states that the Kingdom of Hungary was a monarchy, and the legitimacy and governing power of the monarch varied from being purely symbolic to fully autocratic. Additionally, the definition of monarchy provided indicates that a monarch is typically head of state for life or until abdication, which contradicts the idea of regular elections by landowners.",
  "output": {
    "overall_analysis": "The trace rules out the questioned claim by pointing to evidence that directly contradicts it.",
    "examples": [
      {
        "text": "a monarch is typically head of state for life or until abdication, which contradicts the idea of regular elections by landowners.",
        "explanation": "This eliminates the target claim by citing direct incompatibility with the evidence."
      }
    ]
  }
}

Shot 2
{
  "input": "The Pacific states division is explicitly defined as including states that have coastlines on the Pacific Ocean. Florida does not have a coastline on the Pacific Ocean; it borders the Gulf of Mexico and the Atlantic Ocean instead.",
  "output": {
    "overall_analysis": "The reasoning rejects the yes-answer by using geographic evidence that makes the alternative impossible.",
    "examples": [
      {
        "text": "Florida does not have a coastline on the Pacific Ocean; it borders the Gulf of Mexico and the Atlantic Ocean instead.",
        "explanation": "This explicitly rules out Miami being on the West Coast by contradicting the required condition."
      }
    ]
  }
}''',
}
INCLUDE_FEW_SHOT: bool = False
INCLUDE_DATASET_SPECIFIC_NOTES: bool = True

SYSTEM_PROMPT = """\
You are an expert evaluator of abductive reasoning traces.

## What is Differential Elimination?

Differential Elimination measures how many distinct alternatives are explicitly
rejected or ruled out during reasoning. Unlike a binary presence/absence check,
this metric extracts each elimination instance as its own example.

## What COUNTS as a differential elimination instance

Extract an example when the trace explicitly:
1. Rules out an alternative hypothesis/option with a specific reason.
2. Shows contradiction between an alternative and observed details.
3. Uses conditional falsification ("If X were true, we would see Y, but we don't.").
4. Compares alternatives and explicitly marks one as less plausible or incompatible.

## What does NOT count

- Empty dismissals (e.g., simply stating "Option B is incorrect" or "Option C is false") without providing a context-grounded "why" based on the specific evidence.
- Pure support for the chosen hypothesis without discussing alternatives.
- Listing options without evaluating or eliminating them.
- Vague preference statements without a concrete elimination reason.
- Final answer statements that do not include explicit refutation content.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Extract each distinct elimination/refutation event as a separate example.
- Use `excerpt` as a short direct quote from the reasoning trace (preferably ≤ 30 words).
- Use `explanation` to state what alternative was eliminated and why.
- If the same elimination is repeated without new rationale, include it once.
- Do not paraphrase quoted text.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "overall_analysis": "Brief analysis of elimination behavior in this reasoning trace",
  "examples": [
    {
      "excerpt": "Quote showing explicit elimination of an alternative",
      "explanation": "What was eliminated and why this is a valid elimination instance"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Extract every explicit Differential Elimination instance from the following
reasoning trace. Return one entry per distinct elimination event.

<reasoning_trace>
{text}
</reasoning_trace>
"""
