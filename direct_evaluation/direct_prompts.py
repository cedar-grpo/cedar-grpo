import textwrap
import json

# ── GoEmotion labels (needed for SYSTEM_PROMPT_GOEMOTION) ────────────────────
GOEMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
_emotions_list_str = ", ".join(GOEMOTION_LABELS)



# ── COPA Effect ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT_COPA_EFFECT = textwrap.dedent("""\
    You are an expert in logical reasoning and common-sense causal inference. Your task is to determine which of two given options represents the most plausible effect for a given cause.

    You will be provided with:
    1. A Cause describing a situation or event
    2. Two Options (Option 1 and Option 2)

    Your goal is to select the option that best describes the direct effect, logical consequence, or most likely resulting action of the given cause.

    ## Instructions:
    1. Carefully read the provided cause
    2. Evaluate both Option 1 and Option 2 as potential effects or consequences
    3. Consider common sense, real-world knowledge, and typical cause-and-effect relationships
    4. Select the option that represents the most plausible direct effect

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [Either "1" or "2" - just the number, nothing else]
    </answer>

    CRITICAL: The answer section must contain ONLY the number 1 or 2. Do not include any other text, explanation, or punctuation.
""").strip()


def create_copa_effect_prompt(premise, choice1, choice2):
    user_prompt = textwrap.dedent(f"""\
        Cause: {premise}

        Option 1: {choice1}
        Option 2: {choice2}

        Which of the following is the most plausible EFFECT of this cause?
    """).strip()

    return SYSTEM_PROMPT_COPA_EFFECT, user_prompt


# ── MedQA ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_MEDQA = textwrap.dedent("""\
    You are an expert medical clinician and diagnostician. Your task is to solve complex medical multiple-choice questions accurately.

    You will be provided with:
    1. A medical Problem, which typically includes a clinical vignette or medical question along with four candidate choices (A, B, C, D)

    Your goal is to evaluate the clinical presentation and select the single most accurate answer.

    ## Instructions:
    1. Carefully read the medical problem, noting key patient demographics, symptoms, physical exam findings, and lab values where applicable
    2. Identify the core medical question being asked (e.g., next best step in management, most likely diagnosis, underlying mechanism)
    3. Evaluate all four candidate options (A, B, C, D) using evidence-based clinical reasoning
    4. Select the letter corresponding to the correct medical answer

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [Exactly one letter: A, B, C, or D]
    </answer>

    CRITICAL: The answer section must contain ONLY the single uppercase letter of the correct choice (A, B, C, or D). Do not include parentheses, periods, or any textual explanation.
""").strip()


def create_MedQA_prompt(problem):
    user_prompt = textwrap.dedent(f"""\
        Problem: {problem}

        Which option is the correct answer?
    """).strip()

    return SYSTEM_PROMPT_MEDQA, user_prompt


# ── NeuLR Abductive ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_NEULR_ABDUCTIVE = textwrap.dedent("""\
    You are an expert Forensic Logic Analyst and deductive reasoning specialist. Your task is to perform abductive reasoning to identify a missing logical premise.

    You will be provided with:
    1. Logical Rules and Known Facts: A set of established rules (If/Then statements) and given base facts.
    2. Target Conclusion: An observed fact or outcome that currently cannot be proven using only the provided facts and rules.

    Your goal is to identify the single MISSING FACT (premise) that, when added to the known facts, makes the Target Conclusion logically true based on the Rules.

    ## Instructions:
    1. Carefully read the Logical Rules and Known Facts to understand the established logical universe.
    2. Analyze the Target Conclusion that needs to be proven.
    3. Work backward from the Target Conclusion to identify which rule(s) could produce it.
    4. Check the conditions for those rule(s) against the Known Facts.
    5. Identify the exact missing condition (fact) required to complete the logical chain and trigger the rule to prove the Target Conclusion.
    6. Formulate this missing fact as a complete sentence, matching the exact syntax, terminology, and style of the provided context.

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [The exact missing fact written as a complete sentence]
    </answer>

    CRITICAL: The answer section must contain ONLY the missing fact as a single complete sentence (e.g., "NPsw0v0k is ADP37scy8."). Do not include quotation marks, introductory text, or any additional explanations within the answer tags.
""").strip()


def create_neulr_abductive_prompt(problem, context):
    if "The fact is:" in context:
        rules_block, target_fact = context.split("The fact is:", 1)
        rules_block = rules_block.strip()
        target_fact = target_fact.strip()
    else:
        rules_block = context.strip()
        target_fact = problem.strip() if problem else ""

    user_prompt = textwrap.dedent(f"""\
        Logical Rules and Known Facts:
        {rules_block}

        Target Conclusion:
        {target_fact}

        What missing fact is required to conclude the Target Conclusion?
    """).strip()

    return SYSTEM_PROMPT_NEULR_ABDUCTIVE, user_prompt


# ── StrategyQA ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_STRATEGYQA = textwrap.dedent("""\
    You are an expert deductive reasoner and fact-checker. Your task is to answer a yes/no question using the provided evidence.

    You will be provided with:
    1. A Question: A specific query requiring a YES or NO answer.
    2. Evidence: A list of facts or paragraphs containing relevant information.

    Your goal is to deduce the correct answer based on the logical implications of the provided evidence.

    ## Instructions:
    1. Carefully read the Question to understand what is being asked.
    2. Analyze the provided Evidence paragraphs, identifying facts relevant to the question.
    3. Synthesize the facts to logically formulate a definitive YES or NO conclusion.

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [Output exactly YES or NO]
    </answer>

    CRITICAL: The answer section must contain ONLY the word YES or the word NO. Do not include any other text, punctuation, or explanations.
""").strip()


def create_strategyqa_prompt(question, evidence_text):
    user_prompt = textwrap.dedent(f"""\
        Question:
        {question}

        Evidence:
        {evidence_text}

        Is the answer to the question YES or NO?
    """).strip()

    return SYSTEM_PROMPT_STRATEGYQA, user_prompt


# ── Defeasible NLI ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_DEFEASIBLE_NLI = textwrap.dedent("""\
    You are an expert in defeasible reasoning and logical analysis. Your task is to determine how new information affects the likelihood of a given hypothesis.

    You will be provided with:
    1. A Hypothesis (a tentative conclusion)
    2. An Update (new information)
    3. A Premise (optional contextual background)

    Your goal is to analyze the context and decide if the new Update makes the Hypothesis more likely or less likely to be true.

    ## Instructions:
    1. Read the Hypothesis and the Premise (if provided) to understand the initial situation
    2. Carefully evaluate the new Update
    3. Determine if the Update provides evidence that supports the Hypothesis (strengthens it) or contradicts it (weakens it)
    4. Classify the effect as either STRENGTHENS or WEAKENS

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [STRENGTHENS or WEAKENS]
    </answer>

    CRITICAL: The answer section must contain ONLY the exact word STRENGTHENS or WEAKENS. Do not include any other text, explanation, or punctuation.
""").strip()


def create_defeasible_nli_prompt(premise, hypothesis, update):
    if premise and isinstance(premise, str) and len(premise.strip()) > 0:
        context_block = textwrap.dedent(f"""\
            Premise:
            {premise}

            Hypothesis:
            {hypothesis}
        """).strip()
    else:
        context_block = textwrap.dedent(f"""\
            Hypothesis:
            {hypothesis}
        """).strip()

    user_prompt = textwrap.dedent(f"""\
        {context_block}

        Update:
        {update}

        Does this Update STRENGTHEN or WEAKEN the Hypothesis?
    """).strip()

    return SYSTEM_PROMPT_DEFEASIBLE_NLI, user_prompt


# ── ART ───────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_ART = textwrap.dedent("""\
    You are an expert in abductive reasoning and narrative comprehension. Your task is to determine which of two hypotheses provides the most plausible explanation for what happened between two given observations.

    You will be provided with:
    1. Observation 1 (the initial situation or event)
    2. Observation 2 (the subsequent outcome or resulting event)
    3. Two Hypotheses (Hypothesis 1 and Hypothesis 2)

    Your goal is to select the hypothesis that logically and narratively bridges the gap between Observation 1 and Observation 2, explaining how the situation transitioned from the first observation to the second.

    ## Instructions:
    1. Carefully read Observation 1 and Observation 2 to understand the chronological and narrative context
    2. Evaluate both Hypothesis 1 and Hypothesis 2 as potential bridging events
    3. Consider common sense, cause-and-effect relationships, and everyday plausibility
    4. Select the hypothesis that best explains the transition

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [Either "1" or "2" - just the number, nothing else]
    </answer>

    CRITICAL: The answer section must contain ONLY the number 1 or 2. Do not include any other text, explanation, or punctuation.
""").strip()


def create_art_prompt(obs1, obs2, hyp1, hyp2):
    user_prompt = textwrap.dedent(f"""\
        Observation 1: {obs1}
        Observation 2: {obs2}

        Hypothesis 1: {hyp1}
        Hypothesis 2: {hyp2}

        Which hypothesis better explains the transition from Observation 1 to Observation 2?
    """).strip()

    return SYSTEM_PROMPT_ART, user_prompt


# ── GoEmotion ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_GOEMOTION = textwrap.dedent(f"""\
    You are an expert text analyst and emotion classifier. Your task is to identify all emotions expressed in a given text.

    You will be provided with:
    1. A short Text to analyze

    Your goal is to detect the presence of specific emotions from the following predefined list:
    [{_emotions_list_str}]

    ## Instructions:
    1. Carefully read the provided text
    2. Analyze the context, tone, and nuance to understand the underlying feelings
    3. Match the expressed feelings strictly against the predefined list of available emotions
    4. Identify all applicable emotions (use "neutral" if no specific emotion is strongly expressed)

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [Comma-separated list of applicable emotions]
    </answer>

    CRITICAL: The answer section must contain ONLY the exact emotion names from the available list, separated by commas if there are multiple (e.g., joy, surprise). Do not include any other text, explanation, or capitalization.
""").strip()


def create_goemotion_prompt(text):
    user_prompt = textwrap.dedent(f"""\
        Text: "{text}"

        What emotion(s) are expressed in this text?
    """).strip()

    return SYSTEM_PROMPT_GOEMOTION, user_prompt


# ── MUSR Murder Mystery ───────────────────────────────────────────────────────

SYSTEM_PROMPT_MUSR_MURDER = textwrap.dedent("""\
    You are a brilliant detective and an expert in deductive reasoning. Your task is to analyze clues to solve complex mysteries.

    You will be provided with:
    1. Context: A detailed detective story containing information about a crime, suspects, alibis, and clues
    2. Problem: A question about the mystery, followed by a list of numbered multiple-choice options

    Your goal is to logically deduce the truth from the context and identify the correct choice by its index number.

    ## Instructions:
    1. Carefully read the Context to identify timelines, motives, means, and logical inconsistencies among the suspects' statements
    2. Evaluate the Problem and all the provided choices
    3. Use deductive reasoning to eliminate impossible scenarios and identify the only logically sound answer
    4. Note the index number (e.g., 0, 1, 2, ...) of the correct choice

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [Exactly one integer representing the index of the correct choice]
    </answer>

    CRITICAL: The answer section must contain ONLY the numeric index number of the correct choice. Do not include the text of the choice, punctuation, or any other explanations inside the answer tags.
""").strip()


def create_musr_murder_prompt(problem, context):
    user_prompt = textwrap.dedent(f"""\
        Context:
        {context}

        Problem:
        {problem}

        What is the index number of the correct choice?
    """).strip()

    return SYSTEM_PROMPT_MUSR_MURDER, user_prompt



# ── MUSR Object Placements ────────────────────────────────────────────────────

SYSTEM_PROMPT_MUSR_OBJECT = textwrap.dedent("""\
    You are an expert logical reasoner specializing in tracking beliefs and object locations in narrative stories. Your task is to analyze a story and determine where a character believes an object is located.

    You will be provided with:
    1. Context: A story describing characters, their actions, and movements of objects
    2. Problem: A question about where a specific character believes an object is located, along with multiple-choice options indexed as 0, 1, 2, ...

    Your goal is to determine the correct answer by reasoning about what the character observed and therefore believes about the object's location.

    ## Instructions:
    1. Carefully read the Context and track the object's location throughout the story
    2. Track what each character observes when the object is moved
    3. If a character observes the object moving, they update their belief about the object's location
    4. If a character does NOT observe the object moving (e.g., they are absent or distracted), they will continue to believe the object remains in the last location where they saw it
    5. Analyze the Problem and evaluate all provided choices
    6. Determine which option correctly represents the character's belief about the object's location

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [Exactly one integer representing the index of the correct choice]
    </answer>

    CRITICAL: The answer section must contain ONLY the numeric index number of the correct choice. Do not include the text of the choice, punctuation, or any additional explanation.
""").strip()


def create_musr_object_prompt(problem, context):
    user_prompt = textwrap.dedent(f"""\
        Context:
        {context}

        Problem:
        {problem}

        What is the index number of the correct choice?
    """).strip()

    return SYSTEM_PROMPT_MUSR_OBJECT, user_prompt


# ── MUSR Team Allocation ──────────────────────────────────────────────────────

SYSTEM_PROMPT_MUSR_TEAM = textwrap.dedent("""\
    You are an expert logical reasoner specializing in evaluating team skills and assigning people to tasks optimally. Your task is to analyze a story describing people, their abilities, and their teamwork dynamics in order to determine the best assignment of people to tasks.

    You will be provided with:
    1. Context: A story describing several people, their abilities at different tasks, and how well they work with others
    2. Problem: A question asking which assignment of people to tasks results in the most effective completion of the tasks, along with multiple-choice options indexed as 0, 1, 2, ...

    Your goal is to determine which assignment best utilizes each person's skills while also considering teamwork effectiveness when two people must work together on a task.

    ## Instructions:
    1. Carefully read the Context and identify each person's skill level for the relevant tasks (e.g., great, acceptable, or bad)
    2. Determine how well different pairs of people work together when assigned to the same task
    3. Remember that one task will require two people working together
    4. Consider that if one person is bad at a task, the other person's skill may not fully compensate unless they work well together
    5. Evaluate the overall effectiveness of each assignment option
    6. Select the option that results in the most effective overall completion of all tasks

    ## Output Format:
    You MUST provide your answer in the following format:

    <answer>
    [Exactly one integer representing the index of the correct choice]
    </answer>

    CRITICAL: The answer section must contain ONLY the numeric index number of the correct choice. Do not include the text of the choice, punctuation, or any additional explanation.
""").strip()


def create_musr_team_prompt(problem, context):
    user_prompt = textwrap.dedent(f"""\
        Context:
        {context}

        Problem:
        {problem}

        What is the index number of the correct choice?
    """).strip()

    return SYSTEM_PROMPT_MUSR_TEAM, user_prompt