"""
prompts/counting/backtracking.py
---------------------------------
Prompt for the Backtracking / Self-Correction counting metric.

The LLM returns a list of concrete examples of backtracking moments.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": (
        "Look for corrections in diagnostic/treatment reasoning or recognition of missed clinical details."
    ),
    "art": (
        "Look for true reconsideration of which hypothesis explains observations better, not simple option selection."
    ),
    "strategyqa": (
        "Look for revisions in multi-step reasoning chains, flawed inferences, or changed intermediate conclusions."
    ),
    "copa_guess_effect": (
        "Look for reconsideration of causal interpretation, not just restating a different option."
    ),
    "defeasible_nli": (
        "Look for reconsideration when an inference appears defeated or less robust."
    ),
    "goemotion": (
        "Backtracking is less common; count only genuine revisions in emotion-label selection."
    ),
    "musr": (
        "Look for revised interpretations of narrative details or changed conclusions."
    ),
    "neulr_abductive": (
        "Look for true reconsideration of which explanation best fits observations."
    ),
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {
    "art": '''Shot 1
{
  "input": "Pablo's change in behavior towards worms suggests a shift in his perception or knowledge about them. Hypothesis 1 implies that Pablo's initial liking of worms was based on a misconception about their taste or nutritional value. However, this doesn't explain why he would suddenly stop enjoying them. On the other hand, Hypothesis 2 suggests that Pablo learned something new about worms.",
  "output": {
    "overall_analysis": "The trace starts down one explanatory path, notices it fails to explain the behavioral change, and then pivots to the competing hypothesis.",
    "examples": [
      {
        "text": "However, this doesn't explain why he would suddenly stop enjoying them.",
        "explanation": "The reasoner explicitly identifies a flaw in the current explanation and backs away from it."
      }
    ]
  }
}

Shot 2
{
  "input": "Ben went to the beach on a sunny day, so it's likely he was enjoying the sun initially. However, he then crawled into his tent and napped, which suggests he wanted to be sheltered from the sun. Hypothesis 1 suggests he spent hours in the sun, but doesn't explain why he would need to nap afterward. Hypothesis 2 provides a reason for both his actions.",
  "output": {
    "overall_analysis": "The reasoning revises its first read of the situation after noticing that the tent-and-nap detail conflicts with the initial explanation.",
    "examples": [
      {
        "text": "Hypothesis 1 suggests he spent hours in the sun, but doesn't explain why he would need to nap afterward.",
        "explanation": "This is an explicit recognition that the first hypothesis is inadequate, triggering a change in direction."
      }
    ]
  }
}''',
    "copa_guess_effect": '''Shot 1
{
  "input": "The surfer seeing the wave suggests that she has already entered the ocean and is now observing a wave. If the wave carried her to the shore, it would imply that she was not actively paddling and was instead being transported by the wave, which is a direct effect of seeing the wave. However, if she paddled her board into the ocean, this action precedes the observation of the wave and does not directly result from seeing the wave.",
  "output": {
    "overall_analysis": "The trace corrects its causal interpretation by noticing that one candidate event happens before the stated cause and therefore cannot be the effect.",
    "examples": [
      {
        "text": "However, if she paddled her board into the ocean, this action precedes the observation of the wave and does not directly result from seeing the wave.",
        "explanation": "The reasoner explicitly catches a temporal mismatch and revises away from the earlier candidate."
      }
    ]
  }
}

Shot 2
{
  "input": "When a patient is dehydrated, their body lacks sufficient fluids, which can lead to various symptoms and complications. Testing reflexes, while important for assessing neurological function, is not typically the immediate response to dehydration. Instead, addressing the dehydration by providing fluids is the most direct and effective approach.",
  "output": {
    "overall_analysis": "The reasoning backs away from a less appropriate response and redirects to the correct immediate intervention.",
    "examples": [
      {
        "text": "Testing reflexes, while important for assessing neurological function, is not typically the immediate response to dehydration.",
        "explanation": "This sentence rejects the earlier line of reasoning and marks a genuine correction in approach."
      }
    ]
  }
}''',
    "defeasible_nli": '''Shot 1
{
  "input": "The premise provides context that PersonX wants to fly, which could imply a desire for flight through various means such as using a plane or other flying devices. The hypothesis suggests that PersonX buys a ticket, which is typically associated with air travel. However, the update describes PersonX jumping off the roof while making flapping motions, which does not align with the typical process of buying a ticket for air travel. Given this information, the update does not support the hypothesis that PersonX buys a ticket. Instead, it suggests a different method of achieving the goal of 'flying.'",
  "output": {
    "overall_analysis": "The trace explicitly revises its original interpretation of the hypothesis after the update points to a different mechanism entirely.",
    "examples": [
      {
        "text": "Instead, it suggests a different method of achieving the goal of 'flying.'",
        "explanation": "The reasoner abandons the earlier ticket-purchase reading and switches to a new interpretation."
      }
    ]
  }
}

Shot 2
{
  "input": "The hypothesis posits that PersonX wants to ensure everyone is aware of their presence, possibly to assert their status or importance. However, the update indicates that PersonX is actually trying to distract their friend who is sad. This new information suggests that PersonX's primary motivation is emotional support rather than self-promotion or making a statement about their presence.",
  "output": {
    "overall_analysis": "The trace changes its mind about PersonX's motivation after the update introduces a more plausible explanation.",
    "examples": [
      {
        "text": "This new information suggests that PersonX's primary motivation is emotional support rather than self-promotion or making a statement about their presence.",
        "explanation": "The reasoner explicitly replaces its earlier motivational account with a different one."
      }
    ]
  }
}''',
    "goemotion": '''Shot 1
{
  "input": "The text 'Possibly, but not if it actually succedes' does not contain any strong emotional language or expressions that would indicate clear-cut emotions like joy, sadness, anger, etc. This implies a sense of reluctance or dissatisfaction with the idea of success, which could be interpreted as mild disappointment or frustration. However, without more context, it's difficult to pinpoint a specific emotion. Therefore, the most appropriate emotion here is 'disappointment'.",
  "output": {
    "overall_analysis": "The reasoning backs off a stronger label claim after admitting the evidence is thin, then settles on the closest available emotion.",
    "examples": [
      {
        "text": "However, without more context, it's difficult to pinpoint a specific emotion.",
        "explanation": "The judge explicitly recognizes that its current label hypothesis is too confident and revises its stance."
      }
    ]
  }
}

Shot 2
{
  "input": "The text describes someone who is waiting for a long time and is urgently needing to use the bathroom. This suggests impatience and discomfort, but it doesn't directly align with any of the listed emotions like frustration or discomfort. However, the phrase 'bursting for a piss' implies a strong physical need that is causing some level of distress or urgency. Therefore, the closest match from the given list would be discomfort.",
  "output": {
    "overall_analysis": "The trace notices that its first label framing does not fit the inventory cleanly, then revises toward the nearest supported label.",
    "examples": [
      {
        "text": "it doesn't directly align with any of the listed emotions like frustration or discomfort.",
        "explanation": "This is a self-correction step where the reasoner flags that its current labeling path is not quite right."
      }
    ]
  }
}''',
    "medqa": '''Shot 1
{
  "input": "Pernicious anemia is associated with excessive secretion of gastrin, leading to hypergastrinemia and atrophic gastritis. Given these points, the most likely underlying cause is excessive somatostatin secretion. However, since the options do not include 'somatostatin', we need to consider the closest match. Ectopic secretion of gastrin is the most relevant and likely cause for the described symptoms and findings.",
  "output": {
    "overall_analysis": "The reasoning first lands on a mechanism, then explicitly revises when it notices that mechanism is not actually among the answer choices.",
    "examples": [
      {
        "text": "However, since the options do not include 'somatostatin', we need to consider the closest match.",
        "explanation": "The model catches that its current answer path is unusable and deliberately changes course."
      }
    ]
  }
}

Shot 2
{
  "input": "Given these considerations, the most appropriate choice for treatment is Doxycycline, although it is contraindicated in pregnancy, it remains the first-line antibiotic for Lyme disease when other options are not available. However, since Doxycycline cannot be used in this case due to pregnancy, the next best step would be to use Amoxicillin, which is safe and effective.",
  "output": {
    "overall_analysis": "The trace states an initial treatment choice, notices a contraindication, and then corrects to a safer option.",
    "examples": [
      {
        "text": "However, since Doxycycline cannot be used in this case due to pregnancy, the next best step would be to use Amoxicillin.",
        "explanation": "This is a clear self-correction driven by a newly acknowledged clinical constraint."
      }
    ]
  }
}''',
    "musr": '''Shot 1
{
  "input": "Given these points, Laura would most likely look for the logbook in the place where it was last seen, which is the chart table. However, since the logbook was moved to the cockpit dashboard and then found in Jack's personal bag, we need to consider where Laura would logically check based on her duties and the last known location of the logbook. However, considering the options provided and the fact that the logbook was moved to the cockpit dashboard and then found in Jack's personal bag, the most likely place Laura would look is the chart table. But since the logbook was moved to the cockpit dashboard and then found in Jack's personal bag, the most logical place for Laura to look would be the cockpit dashboard.",
  "output": {
    "overall_analysis": "The trace explicitly changes its answer after reconsidering later story details, making this a strong narrative backtracking example.",
    "examples": [
      {
        "text": "But since the logbook was moved to the cockpit dashboard and then found in Jack's personal bag, the most logical place for Laura to look would be the cockpit dashboard.",
        "explanation": "After previously favoring the chart table, the reasoner reverses course and selects a different location."
      }
    ]
  }
}

Shot 2
{
  "input": "Based on these observations: Liam should be assigned to the Roller Operator role. James should be assigned to Groundwork. Oliver should be assigned to Groundwork. Thus, the unique allocation is: Groundwork: James and Oliver, Roller Operator: Liam. This matches Choice 1. However, the correct choice based on the analysis is actually Choice 1 reversed, as James should be the one doing groundwork and not Oliver.",
  "output": {
    "overall_analysis": "The trace commits to a choice mapping, then notices an inconsistency between the analysis and the selected option and corrects itself.",
    "examples": [
      {
        "text": "However, the correct choice based on the analysis is actually Choice 1 reversed, as James should be the one doing groundwork and not Oliver.",
        "explanation": "The reasoner explicitly states that its earlier mapping was wrong and revises the assignment."
      }
    ]
  }
}''',
    "neulr_abductive": '''Shot 1
{
  "input": "Since Fact6 states that NP55Ong4 is ADPn3dmAA, and Rule6 requires ADP8L64xA for ADPGcTUiy, we need to establish a connection between ADPn3dmAA and ADP8L64xA. However, there is no direct rule connecting ADPn3dmAA to ADP8L64xA. Instead, we can use Rule7 which states that if something is ADPGcTUiy and not ADPgbsTcm, then it is ADPn3dmAA.",
  "output": {
    "overall_analysis": "The trace follows one derivation route, discovers it is blocked, and then deliberately switches to a different rule path.",
    "examples": [
      {
        "text": "However, there is no direct rule connecting ADPn3dmAA to ADP8L64xA. Instead, we can use Rule7",
        "explanation": "This is a textbook backtracking move: the current derivation fails, so the reasoner abandons it and tries another rule."
      }
    ]
  }
}

Shot 2
{
  "input": "We need to find a rule that connects ADP0GHGZA to ADPGWGODT or ADPmcb8gH. However, there is no direct fact connecting ADP0GHGZA to ADPGWGODT or ADPbavqG6. Instead, we should consider the reverse implication from Rule6.",
  "output": {
    "overall_analysis": "The reasoning recognizes that its intended connection is missing and then reorients to an alternative inference strategy.",
    "examples": [
      {
        "text": "However, there is no direct fact connecting ADP0GHGZA to ADPGWGODT or ADPbavqG6. Instead, we should consider the reverse implication from Rule6.",
        "explanation": "The trace explicitly abandons a dead-end search for a direct fact and changes direction."
      }
    ]
  }
}''',
    "strategyqa": '''Shot 1
{
  "input": "The evidence does not provide direct information about shrimp scampi being free of plastic. However, we can infer that scampi may include Nephrops norvegicus, and it is possible that some of this plastic could end up in the shrimp used for scampi dishes. However, the evidence does not explicitly state that scampi or shrimp necessarily contain plastic. Given the lack of direct evidence confirming the absence of plastic in shrimp scampi, we must conclude that the statement is not definitively supported.",
  "output": {
    "overall_analysis": "The trace starts to lean on an indirect inference, then explicitly walks that inference back when it notices the evidence is not strong enough.",
    "examples": [
      {
        "text": "However, the evidence does not explicitly state that scampi or shrimp necessarily contain plastic.",
        "explanation": "The reasoner corrects an overreach by retreating from its earlier speculative inference."
      }
    ]
  }
}

Shot 2
{
  "input": "While the evidence does not explicitly mention home remedies causing skin to turn blue, it does provide examples of how certain substances can lead to blue skin discoloration through argyria. Therefore, based on the evidence provided, it is possible that some home remedies involving silver could result in skin color turning blue. However, it is important to note that the evidence does not directly state that home remedies result in skin color turning blue.",
  "output": {
    "overall_analysis": "The trace makes a tentative extrapolation and then immediately tempers it by acknowledging that the source evidence does not directly support the broader claim.",
    "examples": [
      {
        "text": "However, it is important to note that the evidence does not directly state that home remedies result in skin color turning blue.",
        "explanation": "This is a self-correction that backs away from an inference the trace had just started to make."
      }
    ]
  }
}''',
}
INCLUDE_FEW_SHOT: bool = False
INCLUDE_DATASET_SPECIFIC_NOTES: bool = True

SYSTEM_PROMPT = """\
You are an expert reasoning analyst evaluating AI-generated reasoning traces.

## What is Backtracking?

Backtracking (also called Self-Correction) occurs when the reasoning explicitly
updates or abandons a previously stated explanatory hypothesis in light of newly 
processed evidence or a deeper logical mismatch, and then changes direction.

## What COUNTS as a backtracking moment

Extract an example when you see:
- Explicit admission of error: "Wait, that's wrong", "I made a mistake",
  "Actually, I need to reconsider …"
- Deliberate pausing and restarting: "Hold on, let me re-read the problem",
  "Let's go back to step 2"
- A change of strategy mid-reasoning: "Instead, let's try …",
  "On second thought …", "That approach doesn't work, so …"
- Realisation of a missed detail: "I realise I forgot to account for …",
  "This doesn't look right because …"

## What does NOT count

- Comparing two valid paths (that is Branchiness).
- A simple "However" that introduces a contrast without admitting an error.
- The final answer selection.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}

## Extraction rules

- Extract each explicit self-correction/backtracking event as a separate example.
- Use `excerpt` as a short direct quote from the reasoning trace (preferably ≤ 25 words).
- Use `explanation` to clarify what was revised and why this is true backtracking.
- Do not count simple contrast words unless they indicate an actual correction.
- Do not paraphrase quoted text.

## JSON output format

Return ONLY valid JSON with this structure:
{
  "overall_analysis": "Brief analysis of backtracking/self-correction in this reasoning trace",
  "examples": [
    {
      "excerpt": "Quote of the backtracking moment from the reasoning trace",
      "explanation": "Why this represents backtracking/self-correction"
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Analyze the following reasoning trace for Backtracking / Self-Correction and
extract concrete examples.

<reasoning_trace>
{text}
</reasoning_trace>
"""
