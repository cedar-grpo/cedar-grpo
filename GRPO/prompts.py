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


# ── UniADILR ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_UniADILR = textwrap.dedent("""\
    You are an expert in logical reasoning and abductive inference. Your task is to identify which sentences from a given context provide the necessary evidence to support or explain a hypothesis.

    You will be provided with:
    1. A Context containing multiple numbered sentences (sent1, sent2, sent3, etc.)
    2. A Hypothesis that needs to be supported or explained

    Your goal is to identify which sentence(s) from the context, when combined, provide the logical foundation for the hypothesis through abductive reasoning.

    ## Instructions:
    1. Carefully read all sentences in the context
    2. Analyze the hypothesis
    3. Identify which sentences, when combined, best explain or support the hypothesis
    4. Consider both direct evidence and logical connections
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Sentence numbers only, comma-separated. For example: 5, 13 or 2, 7, 9]
    </answer>

    CRITICAL: The answer section must contain ONLY the sentence numbers separated by commas. Do not include the word "sent" or any other text.
""").strip()


def create_uniadilr_prompt(example):
    context = example["context"]
    hypothesis = example["hypothesis"]

    context_lines = [f"{k}: {v}" for k, v in context.items()]
    context_str = "\n".join(context_lines)

    user_prompt = textwrap.dedent(f"""\
        Context:
        {context_str}

        Hypothesis:
        {hypothesis}

        Which sentence numbers provide the necessary evidence for the hypothesis?
    """).strip()

    return SYSTEM_PROMPT_UniADILR, user_prompt


# ── COPA Cause ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_COPA_CAUSE = textwrap.dedent("""\
    You are an expert in logical reasoning and abductive inference. Your task is to determine which of two given choices represents the most plausible cause for a given premise.

    You will be provided with:
    1. A Premise describing a situation or event
    2. Two Choices (Choice 1 and Choice 2)

    Your goal is to select the choice that best explains WHY the premise happened - identifying the root cause that led to the described situation.

    ## Instructions:
    1. Carefully read the premise
    2. Think step by step to evaluate both choices as potential causes
    3. Consider common sense, real-world knowledge, and typical causal relationships when making your decision
    4. Select the choice that represents the most plausible and direct cause
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Either "1" or "2" - just the number, nothing else]
    </answer>

    CRITICAL: The answer section must contain ONLY the number 1 or 2. Do not include any other text, explanation, or punctuation.
""").strip()


def create_copa_cause_prompt(example):
    user_prompt = textwrap.dedent(f"""\
        Premise: {example['premise']}

        Choice 1: {example['choice1']}
        Choice 2: {example['choice2']}

        Which choice is the most plausible cause for the premise?
    """).strip()

    return SYSTEM_PROMPT_COPA_CAUSE, user_prompt


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
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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


# ── CauseLogics ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CAUSELOGICS = textwrap.dedent("""\
    You are an expert logician and careful reasoning assistant. Your task is to identify whether a given Possible Cause, when added to the provided knowledge base, logically entails an observed Phenomenon.

    You will be provided with:
    1. A set of Premises (facts)
    2. A set of Rules (implications)
    3. An observed Phenomenon
    4. A Possible Cause (a hypothesis)

    Your goal is to determine whether the Phenomenon can be logically inferred by forward reasoning using ONLY the given Premises + Rules (+ the Possible Cause).

    ## Instructions:
    1. Carefully read all Premises and Rules
    2. Assume the Possible Cause is added as an additional premise
    3. Using ONLY the given Premises + Rules (+ the Possible Cause), reason forward
    4. Decide whether the Phenomenon can be logically inferred
       - If the Phenomenon can be inferred, the Possible Cause is TRUE
       - If the Phenomenon cannot be inferred, the Possible Cause is FALSE
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Output exactly one of these two options: TRUE, FALSE]
    </answer>

    CRITICAL: The answer section must contain ONLY one of these two options: TRUE or FALSE. Do not include any other text.
""").strip()


def create_causelogics_prompt(example: dict):
    def _get_any(d, keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    premises_raw = _get_any(example, ["Premises", "premises"], default=[])
    rules_raw = _get_any(example, ["Rules", "rules"], default=[])
    phenomenon = _get_any(example, ["Phenomenon", "phenomenon"], default=None)
    possible_cause = _get_any(example, ["PossibleCause", "possible_cause"], default=None)

    if isinstance(premises_raw, list):
        premises_text = "\n".join([f"- {x}" for x in premises_raw])
    else:
        premises_text = f"- {premises_raw}" if premises_raw is not None else ""

    if isinstance(rules_raw, list):
        rules_text = "\n".join([f"- {x}" for x in rules_raw])
    else:
        rules_text = f"- {rules_raw}" if rules_raw is not None else ""

    if phenomenon is None or possible_cause is None:
        missing = []
        if phenomenon is None:
            missing.append("Phenomenon")
        if possible_cause is None:
            missing.append("PossibleCause")
        raise KeyError(f"CauseLogics example missing required field(s): {', '.join(missing)}")

    user_prompt = textwrap.dedent(f"""\
        Premises:
        {premises_text}

        Rules:
        {rules_text}

        Phenomenon:
        {str(phenomenon)}

        Possible Cause:
        {str(possible_cause)}

        Is the Possible Cause logically TRUE or FALSE?
    """).strip()

    return SYSTEM_PROMPT_CAUSELOGICS, user_prompt


# ── Climate-Fever ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CLIMATE_FEVER = textwrap.dedent("""\
    You are an expert climate scientist and professional fact-checker. Your task is to determine whether a set of provided evidences supports, refutes, disputed or is insufficient to evaluate a specific claim.

    You will be provided with:
    1. A specific Claim
    2. A list of Evidences

    Your goal is to decide whether the Evidence SUPPORTS or REFUTES or DISPUTED the Claim, or if there is NOT ENOUGH INFO, and to justify that decision by citing specific parts of the evidence.

    ## Instructions:
    1. Carefully read the Claim and all provided Evidences
    2. Determine if the Evidence SUPPORTS or REFUTES or DISPUTED the Claim, or if there is NOT ENOUGH INFO
    3. Think step by step about how the specific parts of the evidence relate to the claim
    4. Output the final label
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>

    <answer>
    [Output exactly one of these four options: SUPPORTS, REFUTES, DISPUTED, NOT ENOUGH INFO]
    </answer>

    CRITICAL: The answer section must contain ONLY one of these four options: SUPPORTS, REFUTES, DISPUTED, NOT ENOUGH INFO. Do not include any other text.
""").strip()


def create_climate_fever_prompt(example: dict):
    claim = example["claim"]

    evidence_objs = example.get("evidences", [])
    evidence_list = [e.get("evidence", "") for e in evidence_objs]
    evidence_text = "\n".join([f"- {txt}" for txt in evidence_list if txt])

    user_prompt = textwrap.dedent(f"""\
        Claim:
        {claim}

        Evidence:
        {evidence_text}

        Does the provided evidence SUPPORT, REFUTE, DISPUTED or provide NOT ENOUGH INFO for the claim?
    """).strip()

    return SYSTEM_PROMPT_CLIMATE_FEVER, user_prompt


# ── AbductionRules ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_AbductionRules = textwrap.dedent("""\
    You are an expert in logical reasoning and abductive inference. Your task is to identify the single missing fact that, when added to a given context, makes a query logically decidable.

    You will be provided with:
    1. A Context containing facts and rules
    2. A Query that is currently not decidable from the context alone

    Your goal is to infer ONE additional fact that, when combined with the context, allows the query to be either:
    - proved true, or
    - proved false

    ## Instructions:
    1. Carefully read all facts and rules in the context
    2. Analyze the query
    3. Identify the single missing fact that would make the query decidable
    4. Prefer a direct, minimal explanation:
       - Output exactly one fact
       - Do not output a rule
       - Do not output multiple facts
       - Do not paraphrase beyond the style already used in the context
    5. The fact should be one that works with the existing rules and facts to prove or disprove the query
    6. Be careful with negation:
       - Sometimes the right missing fact helps prove the query
       - Sometimes it helps derive the opposite of the query, thereby disproving it

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Explain your thought process: which rule(s) matter, which existing facts are relevant, and why the missing fact makes the query provable or disprovable]
    </think>

    <answer>
    [Output the single missing fact only, exactly as a natural-language sentence ending with a period]
    </answer>

    CRITICAL:
    - The answer section must contain ONLY one missing fact.
    - Do not include any extra commentary in the answer section.
    - Do not output more than one sentence.
    - Do not output a rule; output a fact about an entity in the context.
""").strip()


def create_abductionrules_prompt(example):
    context = example["context"]
    query = example["query"]

    user_prompt = textwrap.dedent(f"""\
        Context:
        {context}

        Query:
        {query}

        Based on the context and query above, identify the single missing fact that, when added to the context, makes the query logically decidable.
    """).strip()

    return SYSTEM_PROMPT_AbductionRules, user_prompt


# ── Crypto ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CRYPTO_FUNCTION = textwrap.dedent("""\
    You are an expert at inferring exact string transformation rules from examples and expressing them as correct Python functions.

    You will be given several training examples. Each example contains:
    - Input: a string
    - Output: the result of applying the same hidden transformation rule to the input

    Infer the transformation rule that is consistent with ALL training examples, then write a general Python implementation of that rule.

    Before answering, make sure the same rule explains all examples exactly and consistently at the character level.
    Think abductively: consider alternative hypotheses and choose the one that explains all examples exactly.

    Output format (MUST follow exactly):
    <think>
    [Explain your thought process: reason step by step about the possible rules, consider alternative hypotheses, and explain why the rule you chose best fits all examples.]
    </think>
    <answer>
    def transform(s):
        ...
    </answer>

    Code requirements:
    - Define EXACTLY one function named transform.
    - The function takes one argument: s (a string).
    - It MUST return a string.
    - NO IMPORTS allowed.
    - NO printing, no input(), no randomness.
    - Do not hardcode specific training inputs/outputs; generalize the logic.
    - Preserve the behavior implied by the examples for all characters that appear.

    STRICT FORMATTING RULES:
    - Do NOT use markdown code blocks (like ```python) inside the <answer> tags. Just write raw code.
    - Do NOT repeat the code. Write the function exactly once.
    - Ensure you close the tag with </answer>.
    - The <answer> tag must contain ONLY valid Python code, no comments or explanations outside the function.
    - Do NOT write any text before <think> or after </answer>.
""").strip()


def create_crypto_functions_prompt(example):
    train_examples = example["train"]
    if isinstance(train_examples, dict):
        train_examples = train_examples.get("normal", [])
    train_examples = train_examples[:10]

    train_prompt = "\n".join([
        f"Example {i+1}:\nInput: {repr(ex['input'])}\nOutput: {repr(ex['output'])}"
        for i, ex in enumerate(train_examples)
    ])

    user_prompt = textwrap.dedent(f"""\
        Training examples:
        {train_prompt}

        Infer the underlying string transformation and provide the Python function implementation in the required format.
    """).strip()

    return SYSTEM_PROMPT_CRYPTO_FUNCTION, user_prompt


# ── List Function ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_LIST_FUNCTION = textwrap.dedent("""\
    You are an expert at inferring simple list transformations from examples and expressing them as correct Python functions.

    You will be given several training examples. Each example contains:
    - Input:  a list of integers
    - Output: the result of applying the same hidden transformation rule to the input

    Infer the transformation rule that is consistent with ALL training examples, then write a general Python implementation of that rule.

    Output format (MUST follow exactly):
    <think>
    [Explain your thought process: reason step by step about the possible rules, consider alternative hypotheses, and explain why your final rule best fits ALL training examples.]
    </think>
    <answer>
    def transform(lst):
        ...
    </answer>

    Code requirements:
    - Define EXACTLY one function named transform.
    - The function takes one argument: lst (a list of integers).
    - It MUST return a list of integers. If the rule results in a single value, return it as a single-element list (e.g., [val]).
    - NO IMPORTS allowed.
    - NO printing, no input(), no randomness.
    - Do not hardcode specific training inputs/outputs; generalize the logic.
    - BE ROBUST: Handle edge cases like empty lists or lists with only 1 element.

    STRICT FORMATTING RULES:
    - Do NOT use markdown code blocks (like ```python) inside the <answer> tags. Just write raw code.
    - Do NOT repeat the code. Write the function exactly once.
    - Ensure you close the tag with </answer>.
    - The <answer> tag must contain ONLY valid Python code, no comments or explanations outside the function.
    - Do NOT write any text before <think> or after </answer>.
""").strip()


def create_list_functions_prompt(example):
    train_prompt = "\n".join([
        f"--- Example {i+1} ---\nInput: {ex['input']}\nOutput: {ex['output']}"
        for i, ex in enumerate(example["train"])
    ])

    user_prompt = textwrap.dedent(f"""\
        Training examples:
        {train_prompt}

        Infer the underlying list transformation and provide the Python function implementation in the required format.
    """).strip()

    return SYSTEM_PROMPT_LIST_FUNCTION, user_prompt


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
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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
    7. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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


# ── NeuLR Deductive ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_NEULR_DEDUCTIVE = textwrap.dedent("""\
    You are an expert logical reasoner specializing in symbolic logic and deductive pattern recognition. Your task is to analyze factual statements and logical rules to deduce specific relationships between entities.

    You will be provided with:
    1. Context: A set of logical rules and facts involving alphanumeric codes, defining properties (e.g., who belongs to what group) and relationships (e.g., who is afraid of whom).
    2. Problem: A specific question asking you to determine the target of a relationship for a given subject.

    Your goal is to use deductive reasoning to trace the logical connections from the subject to the correct target and identify the exact alphanumeric code representing the answer.

    ## Instructions:
    1. Carefully read the Context to parse all facts (identifying entities and their properties) and rules (defining conditional relationships).
    2. Analyze the Problem to identify the starting subject and the specific relationship being queried.
    3. Trace the logical chain-of-thought, explicitly linking the individual to their group, and the group to the object of their relationship (e.g., fear).
    4. Apply the rules step by step to deduce the final, correct target.
    5. Extract the exact alphanumeric code of the resulting entity from the text.
    6. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Exactly one alphanumeric code]
    </answer>

    CRITICAL: The answer section must contain ONLY the exact alphanumeric code answer. Do not include any extra words, punctuation, full sentences, or explanations inside the answer tags.
""").strip()


def create_neulr_deductive_prompt(problem, context):
    user_prompt = textwrap.dedent(f"""\
        Context:
        {context}

        Problem:
        {problem}

        What is the exact alphanumeric code answer?
    """).strip()

    return SYSTEM_PROMPT_NEULR_DEDUCTIVE, user_prompt


# ── NeuLR Inductive ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_NEULR_INDUCTIVE = textwrap.dedent("""\
    You are an expert logical reasoner and pattern recognition specialist. Your task is to perform inductive reasoning to identify properties of entities based on shared group characteristics.

    You will be provided with:
    1. Context: A set of facts containing entities, their group memberships, and their specific properties.
    2. Problem: A specific question asking you to determine a missing property for a target entity.

    Your goal is to use inductive reasoning to determine the correct property of the target entity by analyzing the properties of other members in its group, and output the exact alphanumeric code.

    ## Instructions:
    1. Carefully read the Context to identify all entities, their assigned groups, and their associated properties.
    2. Analyze the Problem to identify the target entity in question.
    3. Determine which group the target entity belongs to based on the Context.
    4. Examine other entities within that same group to induce the shared property they possess.
    5. Conclude the target entity's missing property based on this shared group characteristic.
    6. Extract the exact alphanumeric code of the resulting property from the text.
    7. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Exactly one alphanumeric code]
    </answer>

    CRITICAL: The answer section must contain ONLY the exact alphanumeric code answer. Do not include any extra words, punctuation, full sentences, or explanations inside the answer tags.
""").strip()


def create_neulr_inductive_prompt(problem, context):
    user_prompt = textwrap.dedent(f"""\
        Context:
        {context}

        Problem:
        {problem}

        What is the exact alphanumeric code answer?
    """).strip()

    return SYSTEM_PROMPT_NEULR_INDUCTIVE, user_prompt


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
    4. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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


# ── GSM8K ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_GSM8K = textwrap.dedent("""\
    You are an expert mathematician and logical problem solver. Your task is to solve grade-school math word problems accurately.

    You will be provided with:
    1. A math Problem

    Your goal is to understand the scenario, perform step-by-step mathematical reasoning, and compute the correct final numeric answer.

    ## Instructions:
    1. Carefully read the math Problem to understand the scenario and the quantities involved
    2. Identify what specific value the problem is asking you to find
    3. Formulate a step-by-step mathematical plan to arrive at the solution
    4. Execute the calculations carefully, verifying each mathematical operation
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Final numeric answer]
    </answer>

    CRITICAL: The answer section must contain ONLY the final numeric answer (e.g., 42, 3.14, or 1500). Do not include units, symbols, equations, or textual explanations inside the answer tags.
""").strip()


def create_gsm8k_prompt(problem):
    user_prompt = textwrap.dedent(f"""\
        Problem: {problem}

        What is the final answer to this problem?
    """).strip()

    return SYSTEM_PROMPT_GSM8K, user_prompt


# ── AIME ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_AIME = textwrap.dedent("""\
    You are an expert mathematician and problem-solver. Your task is to solve an American Invitational Mathematics Examination (AIME) problem.

    You will be provided with:
    1. A mathematical Problem

    Your goal is to mathematically deduce the correct solution and provide the final answer, which is always an integer between 0 and 999.

    ## Instructions:
    1. Carefully read and analyze the mathematical problem
    2. Formulate a rigorous step-by-step mathematical plan to solve it
    3. Execute the calculations, double-checking your work to avoid arithmetic errors
    4. Ensure your final derived answer is an integer from 0 to 999 inclusive
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Integer between 0 and 999]
    </answer>

    CRITICAL: The answer section must contain ONLY the final integer between 0 and 999. Do not include any other text, variables, units, or punctuation.
""").strip()


def create_aime_prompt(problem):
    user_prompt = textwrap.dedent(f"""\
        Problem: {problem}

        What is the final answer to this problem?
    """).strip()

    return SYSTEM_PROMPT_AIME, user_prompt


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
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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
    7. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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
    7. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
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


# ── ML Debugging ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_DEBUGGING = textwrap.dedent("""\
    You are an expert Python developer and debugger. Your task is to identify and fix errors in Python code snippets.

    You will be provided with:
    1. Task Instructions: The intended behavior and requirements for the code
    2. Buggy Code: The incorrect Python code snippet that is failing its tests
    3. Runtime Error / Test Feedback: The execution logs, tracebacks, or failing test results

    Your goal is to analyze the failure, correct the bug, and provide the complete, working Python code.

    ## Instructions:
    1. Carefully read the Task Instructions to understand the desired functionality
    2. Analyze the Buggy Code in conjunction with the Runtime Error / Test Feedback to pinpoint the root cause of the failure
    3. Determine the necessary corrections to fix the bug without breaking existing correct functionality
    4. Provide the full, corrected, and self-contained Python code. Do NOT omit any part of the function or use placeholders (e.g., "# rest of the code")
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    ```python
    [Your full, corrected Python code here]
    ```
    </answer>

    CRITICAL: The answer section must contain ONLY the full, corrected Python code block. Do not include any other text, explanations, or formatting before or after the code block inside the answer tags.
""").strip()


def create_debugging_prompt(sample):
    bug_code = sample.get('bug_code', '')
    runtime_feedback = sample.get('runtime_feedback', '')
    instruct_prompt = sample.get('instruct_prompt', '')

    user_prompt = textwrap.dedent(f"""\
        Task Instructions:
        {instruct_prompt}

        Buggy Code:
        ```python
        {bug_code}
        ```

        Runtime Error / Test Feedback:
        {runtime_feedback}

        What is the fully corrected Python code?
    """).strip()

    return SYSTEM_PROMPT_DEBUGGING, user_prompt


# ── AIMO ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_AIMO = textwrap.dedent("""\
    You are an expert mathematician specializing in competition mathematics (such as AMC, AIME). Your task is to solve complex mathematical problems.

    You will be provided with:
    1. A mathematical Problem (which may contain LaTeX notation)

    Your goal is to logically and rigorously solve the problem to find the correct final mathematical value.

    ## Instructions:
    1. Carefully read and analyze the problem, paying close attention to all mathematical conditions and LaTeX notation
    2. Formulate a structured, mathematical approach to arrive at the solution
    3. Execute your calculations, verifying your algebraic and logical steps along the way
    4. Simplify your final result into a single number, decimal, or fraction (e.g., a/b)
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Output a single number, decimal, or fraction here]
    </answer>

    CRITICAL: The answer section must contain ONLY the final numerical answer (a number, decimal, or fraction like a/b). Do not include variables, units, equations, or any other text.
""").strip()


def create_aimo_prompt(problem):
    system_prompt = SYSTEM_PROMPT_AIMO

    user_prompt = textwrap.dedent(f"""\
        Problem:
        {problem}

        What is the final answer to this problem?
    """).strip()

    return system_prompt, user_prompt


# ── CLUTRR ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CLUTRR = textwrap.dedent("""\
    You are a logic expert specializing in genealogy and family trees. Your task is to deduce the kinship relationship between two specific people based on a provided narrative.

    You will be provided with:
    1. A short Story describing various family relationships
    2. A Query asking for the relationship between two specific people from the story

    Your goal is to trace the family ties described in the story and identify the exact kinship relation connecting the first person to the second person in the query.

    ## Instructions:
    1. Carefully read the story to construct a mental or logical family tree
    2. Identify the two specific individuals mentioned in the query
    3. Trace the genealogical path between these two individuals using the facts established in the story
    4. Determine the exact kinship relation (e.g., aunt, grandfather, son-in-law)
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [The exact kinship relation word]
    </answer>

    CRITICAL: The answer section must contain ONLY the exact kinship relation word or short phrase (e.g., aunt, uncle, grandfather, son-in-law). Do not include any other text, explanation, or punctuation.
""").strip()


def create_clutrr_prompt(story, query):
    if isinstance(query, (list, tuple)):
        formatted_query = f"What is the relationship of {query[0]} to {query[1]}?"
    else:
        formatted_query = query

    system_prompt = SYSTEM_PROMPT_CLUTRR

    user_prompt = textwrap.dedent(f"""\
        Story:
        {story}
        Query:
        {formatted_query}

        Output the exact kinship relation.
    """).strip()

    return system_prompt, user_prompt


# ── COPA (generic / cause) ────────────────────────────────────────────────────

SYSTEM_PROMPT_COPA = textwrap.dedent("""\
    You are an expert in logical reasoning and common-sense causal inference. Your task is to determine which of two given options represents the most plausible cause for a given effect.

    You will be provided with:
    1. An Effect describing a situation or event
    2. Two Options (Option 1 and Option 2)

    Your goal is to select the option that best describes the direct cause, logical predecessor, or most likely triggering action of the given effect.

    ## Instructions:
    1. Carefully read the provided effect
    2. Evaluate both Option 1 and Option 2 as potential preceding causes
    3. Consider common sense, real-world knowledge, and typical cause-and-effect relationships
    4. Select the option that represents the most plausible direct cause
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Either "1" or "2" - just the number, nothing else]
    </answer>

    CRITICAL: The answer section must contain ONLY the number 1 or 2. Do not include any other text, explanation, or punctuation.
""").strip()


def create_copa_prompt(premise, choice1, choice2):
    system_prompt = SYSTEM_PROMPT_COPA

    user_prompt = textwrap.dedent(f"""\
        Effect: {premise}

        Option 1: {choice1}
        Option 2: {choice2}

        Which of the following is the most plausible CAUSE of this effect?
    """).strip()

    return system_prompt, user_prompt


# ── HellaSwag ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_HELLASWAG = textwrap.dedent("""\
    You are an expert in commonsense reasoning and narrative comprehension. Your task is to determine the most logical and natural continuation of a given situation.

    You will be provided with:
    1. A short Context describing a scene, action, or event
    2. Four candidate Endings (A, B, C, D)

    Your goal is to evaluate the candidates and select the single most plausible Ending that best completes the Context.

    ## Instructions:
    1. Carefully read the provided Context to understand the current situation, actors, and actions
    2. Evaluate all four candidate Endings (A, B, C, D)
    3. Determine which ending represents the most natural, logical, and physically plausible continuation based on everyday commonsense
    4. Select the letter corresponding to the best ending
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Exactly one letter: A, B, C, or D]
    </answer>

    CRITICAL: The answer section must contain ONLY the single uppercase letter of the correct choice (A, B, C, or D). Do not include parentheses, punctuation, or any textual explanation.
""").strip()


def create_hellaswag_prompt(ctx, endings):
    system_prompt = SYSTEM_PROMPT_HELLASWAG

    user_prompt = textwrap.dedent(f"""\
        Context:
        {ctx}

        Endings (append one ending to the context):
        A) {endings[0]}
        B) {endings[1]}
        C) {endings[2]}
        D) {endings[3]}

        Which ending best completes the context?
    """).strip()

    return system_prompt, user_prompt


# ── InABHyd ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_INABHYD = textwrap.dedent("""\
    You are an expert logician specializing in inductive and abductive reasoning over synthetic first-order logic worlds. Your task is to deduce the most parsimonious hypotheses that explain a given set of observations based on an incomplete world model.

    You will be provided with:
    1. Theories: Axioms describing an incomplete fictional world model
    2. Observations: Facts that must be explained and logically follow from the theories combined with your hypotheses

    Your goal is to propose one or more hypotheses that, when added to the Theories, make all Observations deductively follow.

    ## Instructions:
    1. Carefully read the Theories and Observations to understand the logical rules and the facts that need explaining
    2. Identify the logical gaps between the Theories and the Observations
    3. Formulate hypotheses to bridge these gaps. Each hypothesis MUST be a simple sentence restricted to one of the following forms:
       - "A is B"
       - "A is not B"
       - "All A are B"
       - "All A are not B"
    4. Make hypotheses as short and general as possible (prefer parsimonious explanations). Do NOT restate the observations as hypotheses unless absolutely necessary
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Your final hypotheses only, one per line]
    </answer>

    CRITICAL: The answer section must contain ONLY the formulated hypotheses, separated by newlines if there are multiple. Do not include bullet points, numbering, or any other textual explanation inside the answer tags.
""").strip()


def create_inabhyd_prompt(claim, evidence_text):
    system_prompt = SYSTEM_PROMPT_INABHYD

    user_prompt = textwrap.dedent(f"""\
        Theories:
        {claim}

        Observations:
        {evidence_text}

        What hypotheses explain all these observations?
    """).strip()

    return system_prompt, user_prompt


# ── MiniARC ───────────────────────────────────────────────────────────────────

def grid_to_string_simple(grid):
    if not grid or not isinstance(grid[0], list):
        return str(grid)
    return "[" + ", ".join([str(row) for row in grid]) + "]"


def grid_to_string(grid):
    if not grid or not isinstance(grid, list) or not isinstance(grid[0], list):
        return str(grid)
    return "\n".join([" ".join([str(val) for val in row]) for row in grid])


SYSTEM_PROMPT_ACR_V1 = textwrap.dedent("""\
    You are an expert at inferring grid transformation rules from examples and expressing them as correct Python functions.

    You will be given several training examples. Each example contains:
    - Input:  a 2D grid (nested list of integers)
    - Output: the result of applying the same hidden transformation rule to the input grid

    Infer the transformation rule that is consistent with ALL training examples, then write a general Python implementation of that rule.

    ### Output format:
    <think>
    Briefly describe the rule you inferred and any important edge cases.
    </think>
    <answer>
    def transform(grid):
        ...
    </answer>

    Code requirements:
    - Define EXACTLY one function named transform.
    - The function takes one argument: grid (nested list of integers).
    - It MUST return the transformed grid (nested list of integers).
    - NO IMPORTS allowed.
    - NO printing, no input(), no randomness.
    - Do not hardcode specific training inputs/outputs; generalize the logic.

    STRICT FORMATTING RULES:
    - Output ONLY the <think> and <answer> blocks—no other text.
    - Do NOT use markdown code blocks (like ```python) inside the <answer> tags. Just write raw code.
    - Do NOT repeat the code. Write the function exactly once.
    - Ensure you close the tag with </answer>.
    - The <answer> tag must contain ONLY valid Python code, no comments or explanations outside the function.
""").strip()

SYSTEM_PROMPT_ACR_V2 = textwrap.dedent("""\
    You are an expert at inferring grid transformation rules from examples and expressing them as correct Python functions.

    You will be given several training examples. Each example contains:
    - Input:  a 2D grid (nested list of integers)
    - Output: the result of applying the same hidden transformation rule to the input grid

    Infer the transformation rule that is consistent with ALL training examples, then write a general Python implementation of that rule.

    Hint: MiniARC tasks often involve identifying distinct objects (connected components of the same color), counting elements, or applying simple geometric transformations like flips and rotations.

    ### Output format:
    <think>
    Briefly describe the rule you inferred and any important edge cases.
    </think>
    <answer>
    def transform(grid):
        ...
    </answer>

    Code requirements:
    - Define EXACTLY one function named transform.
    - The function takes one argument: grid (nested list of integers).
    - It MUST return the transformed grid (nested list of integers).
    - NO IMPORTS allowed.
    - NO printing, no input(), no randomness.
    - Do not hardcode specific training inputs/outputs; generalize the logic.

    STRICT FORMATTING RULES:
    - Output ONLY the <think> and <answer> blocks—no other text.
    - Do NOT use markdown code blocks (like ```python) inside the <answer> tags. Just write raw code.
    - Do NOT repeat the code. Write the function exactly once.
    - Ensure you close the tag with </answer>.
    - The <answer> tag must contain ONLY valid Python code, no comments or explanations outside the function.
""").strip()

SYSTEM_PROMPT_ACR_V3 = textwrap.dedent("""\
    You are an expert at inferring grid transformation rules from examples and expressing them as correct Python functions.

    You will be given several training examples. Each example contains:
    - Input:  a 2D grid (nested list of integers)
    - Output: the result of applying the same hidden transformation rule to the input grid

    Infer the transformation rule that is consistent with ALL training examples, then write a general Python implementation of that rule.

    Detailed Hint: MiniARC tasks typically require reasoning about:
    1. Objects: Groups of adjacent pixels of the same color.
    2. Geometry: Flips, rotations, translations, and scaling of objects.
    3. Topology: Containment (inside/outside), boundaries, and connectivity.
    4. Counting: Number of objects, number of pixels of a certain color, or dimensions.
    5. Color Logic: Changing colors based on frequency, position, or neighbor colors.
    6. Symmetry: Horizontal, vertical, or diagonal symmetry.
    7. Movement: Moving objects until they hit a boundary or another object (gravity, bouncing).

    ### Output format:
    <think>
    [Explain your thought process: reason step by step about the possible rules, consider alternative hypotheses, and explain why your final rule best fits ALL training examples.]
    </think>
    <answer>
    def transform(grid):
        ...
    </answer>

    Code requirements:
    - Define EXACTLY one function named transform.
    - The function takes one argument: grid (nested list of integers).
    - It MUST return the transformed grid (nested list of integers).
    - NO IMPORTS allowed.
    - NO printing, no input(), no randomness.
    - Do not hardcode specific training inputs/outputs; generalize the logic.

    STRICT FORMATTING RULES:
    - Output ONLY the <think> and <answer> blocks—no other text.
    - Do NOT use markdown code blocks (like ```python) inside the <answer> tags. Just write raw code.
    - Do NOT repeat the code. Write the function exactly once.
    - Ensure you close the tag with </answer>.
    - The <answer> tag must contain ONLY valid Python code, no comments or explanations outside the function.
""").strip()

SYSTEM_PROMPT_ACR = SYSTEM_PROMPT_ACR_V1


def create_acr_prompt(example):
    system_prompt = SYSTEM_PROMPT_ACR
    train_prompt = "\n".join([
        f"--- Example {i+1} ---\nInput:\n{grid_to_string(ex['input'])}\nOutput:\n{grid_to_string(ex['output'])}"
        for i, ex in enumerate(example["train"])
    ])
    user_prompt = textwrap.dedent(f"""\
        Training examples:
        {train_prompt}

        Infer the underlying grid transformation and provide the Python function implementation in the required format.
    """).strip()
    return system_prompt, user_prompt


# ── VitaminC ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_VITAMINC = textwrap.dedent("""\
    You are an expert fact-checker. Your task is to verify a claim against a provided piece of evidence.

    You will be provided with:
    1. A Claim: A statement to be verified.
    2. Evidence: A specific text snippet that may support, refute, or be neutral regarding the claim.

    Note: The VitaminC dataset features "contrastive evidence" where small changes in wording, numbers, or negations can completely flip the label. Pay close attention to these nuances.

    Your goal is to determine if the evidence SUPPORTS, REFUTES, or provides NOT ENOUGH INFO regarding the claim.

    ## Instructions:
    1. Carefully read the Claim and the Evidence.
    2. Analyze the Evidence, paying special attention to negations, numbers, and specific entities to see how they align with or contradict the Claim.
    3. Determine the logical relationship between the Evidence and the Claim.
    4. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Output exactly one of: SUPPORTS, REFUTES, NOT ENOUGH INFO]
    </answer>

    CRITICAL: The answer section must contain ONLY "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO". Do not include any other text, punctuation, or explanations.
""").strip()


def create_vitaminc_prompt(claim, evidence_text):
    system_prompt = SYSTEM_PROMPT_VITAMINC

    user_prompt = textwrap.dedent(f"""\
        Claim:
        {claim}

        Evidence:
        {evidence_text}

        Does the evidence SUPPORT, REFUTE, or provide NOT ENOUGH INFO for the claim?
    """).strip()

    return system_prompt, user_prompt


# ── WinoGrande ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_WINOGRANDE = textwrap.dedent("""\
    You are an expert in commonsense reasoning and pronoun resolution. Your task is to determine the correct word or phrase to complete a given sentence.

    You will be provided with:
    1. A Sentence: A statement containing a blank space represented by an underscore character (_).
    2. Option 1: The first candidate to fill the blank.
    3. Option 2: The second candidate to fill the blank.

    Your goal is to decide which candidate option best fills the blank to make the sentence coherent, logically correct, and aligned with everyday commonsense.

    ## Instructions:
    1. Read the sentence carefully and analyze the context surrounding the blank.
    2. Evaluate Option 1 and Option 2 as potential replacements for the blank.
    3. Use commonsense reasoning to determine which option creates a logically sound sentence.
    4. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Output exactly 1 or 2]
    </answer>

    CRITICAL: The answer section must contain ONLY the number 1 or the number 2. Do not include the actual text of the option, any other text, punctuation, or explanations.
""").strip()


def create_winogrande_prompt(sentence, option1, option2):
    system_prompt = SYSTEM_PROMPT_WINOGRANDE

    user_prompt = textwrap.dedent(f"""\
        Sentence:
        {sentence}

        Option 1:
        {option1}

        Option 2:
        {option2}

        Which option correctly fills the blank "_" in the sentence?
    """).strip()

    return system_prompt, user_prompt


# ── PySStuBs (bug line-number detection) ─────────────────────────────────────

SYSTEM_PROMPT_PYSSTUBS = textwrap.dedent("""\
    You are an expert software developer and debugger. Your task is to identify the exact line number of a bug in the provided source code.

    You will be provided with:
    1. The buggy source code with line numbers added at the beginning of each line.

    Your goal is to analyze the code, reason about all possible bugs, and logically deduce the exact line number where the bug exists.

    ## Instructions:
    1. Carefully read and analyze the provided source code.
    2. Formulate a rigorous step-by-step reasoning to identify the bug. You must reason and think of all the possible bugs and then conclude which line has the bug.
    3. Ensure your final answer is the exact integer line number of the bug.
    4. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here, reasoning about all possible bugs and concluding which line has the bug]
    </think>
    <answer>
    [Integer representing the exact line number of the bug]
    </answer>

    CRITICAL: The answer section must contain ONLY the final integer representing the line number. Do not include any other text, variables, code snippets, or punctuation.
""").strip()


def add_line_numbers(code: str) -> str:
    if not code:
        return ""
    lines = code.split('\n')
    return '\n'.join([f"{i+1}: {line}" for i, line in enumerate(lines)])


def create_pysstubs_prompt(sample):
    raw_code = sample.get('buggy_code_before', sample.get('code', sample.get('file_content', '')))
    numbered_code = add_line_numbers(str(raw_code))

    user_prompt = textwrap.dedent(f"""\
        Source Code:
        ```python
        {numbered_code}
        ```

        What is the exact line number of the bug?
    """).strip()

    return SYSTEM_PROMPT_PYSSTUBS, user_prompt


# ── e-CARE ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_ECARE = textwrap.dedent("""\
    You are an expert in causal reasoning and multiple-choice evaluation. Your task is to determine the correct causal relationship based on a given premise and question type.

    You will be provided with:
    1. A Premise describing a specific situation or event
    2. Additional context containing the Question Type (asking for either a cause or an effect) and two candidate choices

    Your goal is to evaluate both choices and select the one that represents the most plausible cause or effect, depending on what the question asks.

    ## Instructions:
    1. Carefully read the Premise
    2. Identify the Question Type from the provided text to determine if you are looking for a cause of the premise or an effect resulting from the premise
    3. Evaluate both candidate choices against the premise
    4. Select the choice that forms the most logical causal relationship
    5. Think step by step.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Either CHOICE1 or CHOICE2]
    </answer>

    CRITICAL: The answer section must contain ONLY the exact word CHOICE1 or CHOICE2. Do not include any other text, explanation, or punctuation.
""").strip()


def create_ecare_prompt(claim, evidence_text):
    system_prompt = SYSTEM_PROMPT_ECARE

    user_prompt = textwrap.dedent(f"""\
        Premise:
        {claim}

        {evidence_text}

        Which choice is the correct answer?
    """).strip()

    return system_prompt, user_prompt

# ── CommonsenseQA ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_CSQA = textwrap.dedent("""\
    You are an expert at commonsense reasoning. Your task is to correctly answer multiple-choice questions based on everyday knowledge.

    You will be provided with:
    1. A Question
    2. A set of Choices (labeled A, B, C, D, E)

    ## Instructions:
    1. Carefully read the question and the choices.
    2. Think step by step about the relationships between the concepts and everyday commonsense scenarios.
    3. Select the single best choice that answers the question.
    
    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Choice Letter]
    </answer>

    CRITICAL: The answer section must contain ONLY the single uppercase letter of the correct choice (A, B, C, D, or E).
""").strip()

def create_csqa_prompt(sample):
    question = sample['question']
    labels = sample['choices']['label']
    texts = sample['choices']['text']
    choices_str = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
    
    user_prompt = textwrap.dedent(f"""\
        Question: {question}

        Choices:
        {choices_str}

        What is the correct choice?
    """).strip()
    return SYSTEM_PROMPT_CSQA, user_prompt

# ── FOLIO ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_FOLIO = textwrap.dedent("""\
    You are an expert in first-order logic. Your task is to determine the truth value of a conclusion given a set of premises.

    You will be provided with:
    1. A list of Premises
    2. A Conclusion

    ## Instructions:
    1. Carefully read the premises and assume they are all true.
    2. Apply strict logical deduction to evaluate the conclusion.
    3. Determine if the conclusion is strictly True, strictly False, or if it is Uncertain (Unknown) based solely on the provided premises.

    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [True/False/Uncertain]
    </answer>

    CRITICAL: The answer section must contain ONLY one of the exact words: True, False, or Uncertain.
""").strip()

def create_folio_prompt(sample):
    premises = sample['premises']
    conclusion = sample['conclusion']
    premises_str = "\n".join([f"- {p}" for p in premises])
    
    user_prompt = textwrap.dedent(f"""\
        Premises:
        {premises_str}

        Conclusion: {conclusion}

        Is the conclusion True, False, or Uncertain given the premises?
    """).strip()
    return SYSTEM_PROMPT_FOLIO, user_prompt

# ── BigBench ────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT_BIGBENCH = textwrap.dedent("""\
    You are an expert problem solver and reasoner. Your task is to correctly answer multiple-choice questions spanning various tasks.

    You will be provided with:
    1. An Input / Question
    2. A set of Choices (labeled A, B, C, D, etc.)

    ## Instructions:
    1. Carefully read the input and the choices.
    2. Think step by step to deduce the correct answer.
    3. Select the single best choice that answers the question.
    
    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>
    [Choice Letter]
    </answer>

    CRITICAL: The answer section must contain ONLY the single uppercase letter of the correct choice.
""").strip()

def create_bigbench_prompt(sample):
    input_text = sample.get('input', '')
    choices = sample.get('answer_choices', [])
    labels =[chr(65 + i) for i in range(len(choices))]
    choices_str = "\n".join([f"{l}. {t}" for l, t in zip(labels, choices)])
    
    user_prompt = textwrap.dedent(f"""\
        Input: {input_text}

        Choices:
        {choices_str}

        What is the correct choice?
    """).strip()
    return SYSTEM_PROMPT_BIGBENCH, user_prompt

# ── MMLU ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_MMLU = textwrap.dedent("""\
    You are an expert in various academic subjects. Your task is to correctly answer multiple-choice questions from different domains.

    You will be provided with:
    1. A Question
    2. A set of Choices (labeled A, B, C, D, etc.)

    ## Instructions:
    1. Carefully read the question and the choices.
    2. Think step by step to arrive at the correct answer.
    3. Select the single best choice that answers the question.
    
    ## Output Format:
    You MUST provide your answer in the following format:

    <think>
    [Think step by step here]
    </think>
    <answer>[Choice Letter]
    </answer>

    CRITICAL: The answer section must contain ONLY the single uppercase letter of the correct choice (e.g., A, B, C, or D).
""").strip()

def create_mmlu_prompt(sample):
    question = sample.get('question', '')
    choices = sample.get('choices', [])
    labels =[chr(65 + i) for i in range(len(choices))]
    choices_str = "\n".join([f"{l}. {t}" for l, t in zip(labels, choices)])
    
    user_prompt = textwrap.dedent(f"""\
        Question: {question}

        Choices:
        {choices_str}

        What is the correct choice?
    """).strip()
    return SYSTEM_PROMPT_MMLU, user_prompt