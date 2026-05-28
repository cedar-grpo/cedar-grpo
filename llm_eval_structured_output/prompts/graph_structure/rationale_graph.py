"""
prompts/graph_structure/rationale_graph.py
------------------------------------------
Prompt for extracting a text-grounded directed rationale graph.

The judge must build a directed graph grounded only in exact rationale text spans.
"""

DATASET_SPECIFIC_NOTES: dict[str, str] = {
    "medqa": "Represent clinical entities, findings, and inferred links exactly as stated in the rationale.",
    "art": "Represent observations and hypothesis-linking statements exactly as stated in the rationale.",
    "strategyqa": "Represent factual claims and inferential links exactly as stated in the rationale.",
    "copa_guess_effect": "Represent cause/effect statements and directional links exactly as stated in the rationale.",
    "defeasible_nli": "Represent premise-claim links and defeasible transitions exactly as stated in the rationale.",
    "goemotion": "Represent textual cues and emotion-inference links exactly as stated in the rationale.",
    "musr": "Represent narrative facts and relational links exactly as stated in the rationale.",
    "neulr_abductive": "Represent observation and explanation links exactly as stated in the rationale.",
}

DATASET_FEW_SHOT_EXAMPLES: dict[str, str] = {
    "strategyqa":"",
    "neulr_abductive" : """
****** Example 1:

Reasoning trace: 
To prove that NPmIM6GH is not ADPfDggVP, we need to ensure that NPmIM6GH does not meet the conditions under which it would be ADPfDggVP according to Rule1 or Rule2.\n\nRule1 states: If something is ADPpL9Wve and ADPu0P6me then it is not ADPfDggVP.\nRule2 states: If NPmIM6GH is ADPpL9Wve and ADPfDggVP then NPmIM6GH is ADPCWzZmn.\n\nFrom the known facts, we know that NPmIM6GH is ADPpL9Wve (Fact7). For NPmIM6GH to be ADPfDggVP, it must also be ADPu0P6me. However, there is no direct evidence that NPmIM6GH is ADPu0P6me from the given facts.\n\nTherefore, the missing fact should be something that prevents NPmIM6GH from being ADPu0P6me, ensuring it cannot satisfy the condition in Rule1 for ADPfDggVP.\n\nSince Rule5 states that ADPu0P6me things are ADPCWzZmn, and Rule4 states that ADPCWzZmn things are ADPpL9Wve, we can infer that if NPmIM6GH were ADPu0P6me, it would have to be ADPCWzZmn and ADPpL9Wve, which contradicts the conclusion that NPmIM6GH is not ADPfDggVP.\n\nThus, the missing fact should be something that explicitly states NPmIM6GH is not ADPu0P6me.

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "NPmIM6GH is not ADPfDggVP",
      "description": "The target conclusion to be proved",
      "text_correspondence": "To prove that NPmIM6GH is not ADPfDggVP"
    },
    {
      "vertex_id": "v2",
      "label": "Rule1",
      "description": "Rule stating conditions under which something is not ADPfDggVP",
      "text_correspondence": "Rule1 states: If something is ADPpL9Wve and ADPu0P6me then it is not ADPfDggVP."
    },
    {
      "vertex_id": "v3",
      "label": "Rule2",
      "description": "Rule linking ADPpL9Wve and ADPfDggVP to ADPCWzZmn",
      "text_correspondence": "Rule2 states: If NPmIM6GH is ADPpL9Wve and ADPfDggVP then NPmIM6GH is ADPCWzZmn."
    },
    {
      "vertex_id": "v4",
      "label": "NPmIM6GH is ADPpL9Wve (Fact7)",
      "description": "Known fact about NPmIM6GH",
      "text_correspondence": "From the known facts, we know that NPmIM6GH is ADPpL9Wve (Fact7)."
    },
    {
      "vertex_id": "v5",
      "label": "NPmIM6GH must also be ADPu0P6me",
      "description": "Additional condition required for NPmIM6GH to be ADPfDggVP",
      "text_correspondence": "For NPmIM6GH to be ADPfDggVP, it must also be ADPu0P6me."
    },
    {
      "vertex_id": "v6",
      "label": "No direct evidence NPmIM6GH is ADPu0P6me",
      "description": "Observation that the facts do not support ADPu0P6me for NPmIM6GH",
      "text_correspondence": "However, there is no direct evidence that NPmIM6GH is ADPu0P6me from the given facts."
    },
    {
      "vertex_id": "v7",
      "label": "Rule5: ADPu0P6me things are ADPCWzZmn",
      "description": "Rule linking ADPu0P6me to ADPCWzZmn",
      "text_correspondence": "Since Rule5 states that ADPu0P6me things are ADPCWzZmn"
    },
    {
      "vertex_id": "v8",
      "label": "Rule4: ADPCWzZmn things are ADPpL9Wve",
      "description": "Rule linking ADPCWzZmn to ADPpL9Wve",
      "text_correspondence": "Rule4 states that ADPCWzZmn things are ADPpL9Wve"
    },
    {
      "vertex_id": "v9",
      "label": "If NPmIM6GH were ADPu0P6me, it would be ADPCWzZmn and ADPpL9Wve",
      "description": "Hypothetical inference chain if NPmIM6GH were ADPu0P6me",
      "text_correspondence": "we can infer that if NPmIM6GH were ADPu0P6me, it would have to be ADPCWzZmn and ADPpL9Wve"
    },
    {
      "vertex_id": "v10",
      "label": "Contradicts conclusion NPmIM6GH is not ADPfDggVP",
      "description": "The contradiction that arises from the hypothetical",
      "text_correspondence": "which contradicts the conclusion that NPmIM6GH is not ADPfDggVP"
    },
    {
      "vertex_id": "v11",
      "label": "Missing fact: NPmIM6GH is not ADPu0P6me",
      "description": "The identified missing fact needed to complete the proof",
      "text_correspondence": "Thus, the missing fact should be something that explicitly states NPmIM6GH is not ADPu0P6me."
    }
  ],
  "edges": [
    {
      "source_vertex_label": "Rule1",
      "target_vertex_label": "NPmIM6GH is not ADPfDggVP",
      "edge_label": "conditions for conclusion",
      "description": "Rule1 provides conditions under which NPmIM6GH would not be ADPfDggVP",
      "text_correspondence": "we need to ensure that NPmIM6GH does not meet the conditions under which it would be ADPfDggVP according to Rule1 or Rule2."
    },
    {
      "source_vertex_label": "Rule2",
      "target_vertex_label": "NPmIM6GH is not ADPfDggVP",
      "edge_label": "conditions for conclusion",
      "description": "Rule2 also provides conditions relevant to whether NPmIM6GH is ADPfDggVP",
      "text_correspondence": "we need to ensure that NPmIM6GH does not meet the conditions under which it would be ADPfDggVP according to Rule1 or Rule2."
    },
    {
      "source_vertex_label": "NPmIM6GH is ADPpL9Wve (Fact7)",
      "target_vertex_label": "NPmIM6GH must also be ADPu0P6me",
      "edge_label": "additional condition needed",
      "description": "Since NPmIM6GH is already ADPpL9Wve, the remaining condition is ADPu0P6me",
      "text_correspondence": "From the known facts, we know that NPmIM6GH is ADPpL9Wve (Fact7). For NPmIM6GH to be ADPfDggVP, it must also be ADPu0P6me."
    },
    {
      "source_vertex_label": "NPmIM6GH must also be ADPu0P6me",
      "target_vertex_label": "No direct evidence NPmIM6GH is ADPu0P6me",
      "edge_label": "lack of evidence",
      "description": "The required condition ADPu0P6me has no supporting evidence in the facts",
      "text_correspondence": "For NPmIM6GH to be ADPfDggVP, it must also be ADPu0P6me. However, there is no direct evidence that NPmIM6GH is ADPu0P6me from the given facts."
    },
    {
      "source_vertex_label": "Rule5: ADPu0P6me things are ADPCWzZmn",
      "target_vertex_label": "If NPmIM6GH were ADPu0P6me, it would be ADPCWzZmn and ADPpL9Wve",
      "edge_label": "inference chain",
      "description": "Rule5 contributes to the hypothetical inference about NPmIM6GH",
      "text_correspondence": "Since Rule5 states that ADPu0P6me things are ADPCWzZmn... we can infer that if NPmIM6GH were ADPu0P6me, it would have to be ADPCWzZmn and ADPpL9Wve"
    },
    {
      "source_vertex_label": "Rule4: ADPCWzZmn things are ADPpL9Wve",
      "target_vertex_label": "If NPmIM6GH were ADPu0P6me, it would be ADPCWzZmn and ADPpL9Wve",
      "edge_label": "inference chain",
      "description": "Rule4 contributes to the hypothetical inference about NPmIM6GH",
      "text_correspondence": "Rule4 states that ADPCWzZmn things are ADPpL9Wve, we can infer that if NPmIM6GH were ADPu0P6me, it would have to be ADPCWzZmn and ADPpL9Wve"
    },
    {
      "source_vertex_label": "If NPmIM6GH were ADPu0P6me, it would be ADPCWzZmn and ADPpL9Wve",
      "target_vertex_label": "Contradicts conclusion NPmIM6GH is not ADPfDggVP",
      "edge_label": "leads to contradiction",
      "description": "The hypothetical inference contradicts the target conclusion",
      "text_correspondence": "we can infer that if NPmIM6GH were ADPu0P6me, it would have to be ADPCWzZmn and ADPpL9Wve, which contradicts the conclusion that NPmIM6GH is not ADPfDggVP."
    },
    {
      "source_vertex_label": "No direct evidence NPmIM6GH is ADPu0P6me",
      "target_vertex_label": "Missing fact: NPmIM6GH is not ADPu0P6me",
      "edge_label": "motivates missing fact",
      "description": "The lack of evidence motivates identifying the missing fact",
      "text_correspondence": "the missing fact should be something that prevents NPmIM6GH from being ADPu0P6me"
    },
    {
      "source_vertex_label": "Contradicts conclusion NPmIM6GH is not ADPfDggVP",
      "target_vertex_label": "Missing fact: NPmIM6GH is not ADPu0P6me",
      "edge_label": "resolves contradiction",
      "description": "The missing fact resolves the potential contradiction",
      "text_correspondence": "Thus, the missing fact should be something that explicitly states NPmIM6GH is not ADPu0P6me."
    }
  ]
}




********** Example 2:

Reasoning trace: 
To prove that NP2cFbYU is ADPK9s7iO, we need to use Rule5 which states that ADP8PfbGX, ADP6Ys2yo things are ADPK9s7iO. We already know from Fact8 that NP2cFbYU is ADP6Ys2yo. Therefore, we need to establish that NP2cFbYU is also ADP8PfbGX to satisfy the conditions for Rule5.\n\nLet's trace back to see if there's any rule or fact that can help us confirm this:\n- Fact8 tells us NP2cFbYU is ADP6Ys2yo.\n- We need to find a way to link ADP6Ys2yo to ADP8PfbGX.\n\nLooking at the rules, Rule4 states that if something is ADP8PfbGX then it is ADPlPZL25. However, we don't have direct information about NP2cFbYU being ADP8PfbGX. Instead, we can use Rule1 indirectly by checking if there's any contradiction or additional information that might lead us to ADP8PfbGX.\n\nSince Rule1 involves ADP8PfbGX and ADP6Riq7F, and we don't have direct information about NP2cFbYU being ADP8PfbGX, we need to ensure there's no conflicting information. The only way to ensure NP2cFbYU can be ADPK9s7iO is if it is ADP8PfbGX.\n\nThus, the missing fact should be something that directly links NP2cFbYU to ADP8PfbGX.

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "NP2cFbYU is ADPK9s7iO",
      "description": "The goal to be proved",
      "text_correspondence": "To prove that NP2cFbYU is ADPK9s7iO"
    },
    {
      "vertex_id": "v2",
      "label": "Rule5",
      "description": "Rule stating that ADP8PfbGX, ADP6Ys2yo things are ADPK9s7iO",
      "text_correspondence": "we need to use Rule5 which states that ADP8PfbGX, ADP6Ys2yo things are ADPK9s7iO"
    },
    {
      "vertex_id": "v3",
      "label": "NP2cFbYU is ADP6Ys2yo (Fact8)",
      "description": "Known fact from Fact8",
      "text_correspondence": "We already know from Fact8 that NP2cFbYU is ADP6Ys2yo"
    },
    {
      "vertex_id": "v4",
      "label": "NP2cFbYU is ADP8PfbGX",
      "description": "The missing condition needed to satisfy Rule5",
      "text_correspondence": "we need to establish that NP2cFbYU is also ADP8PfbGX to satisfy the conditions for Rule5"
    },
    {
      "vertex_id": "v5",
      "label": "Rule4",
      "description": "Rule stating if something is ADP8PfbGX then it is ADPlPZL25",
      "text_correspondence": "Rule4 states that if something is ADP8PfbGX then it is ADPlPZL25"
    },
    {
      "vertex_id": "v6",
      "label": "No direct information about NP2cFbYU being ADP8PfbGX",
      "description": "Observation that there is no direct fact linking NP2cFbYU to ADP8PfbGX",
      "text_correspondence": "we don't have direct information about NP2cFbYU being ADP8PfbGX"
    },
    {
      "vertex_id": "v7",
      "label": "Rule1 involves ADP8PfbGX and ADP6Riq7F",
      "description": "Observation about Rule1's involvement",
      "text_correspondence": "Rule1 involves ADP8PfbGX and ADP6Riq7F"
    },
    {
      "vertex_id": "v8",
      "label": "Missing fact links NP2cFbYU to ADP8PfbGX",
      "description": "Conclusion about what the missing fact should be",
      "text_correspondence": "the missing fact should be something that directly links NP2cFbYU to ADP8PfbGX"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "Rule5",
      "target_vertex_label": "NP2cFbYU is ADPK9s7iO",
      "edge_label": "used to prove",
      "description": "Rule5 is the rule needed to prove the goal",
      "text_correspondence": "To prove that NP2cFbYU is ADPK9s7iO, we need to use Rule5"
    },
    {
      "source_vertex_label": "NP2cFbYU is ADP6Ys2yo (Fact8)",
      "target_vertex_label": "Rule5",
      "edge_label": "satisfies one condition of",
      "description": "Fact8 satisfies the ADP6Ys2yo condition of Rule5",
      "text_correspondence": "We already know from Fact8 that NP2cFbYU is ADP6Ys2yo"
    },
    {
      "source_vertex_label": "NP2cFbYU is ADP8PfbGX",
      "target_vertex_label": "Rule5",
      "edge_label": "needed to satisfy conditions for",
      "description": "ADP8PfbGX is the other condition needed for Rule5",
      "text_correspondence": "we need to establish that NP2cFbYU is also ADP8PfbGX to satisfy the conditions for Rule5"
    },
    {
      "source_vertex_label": "No direct information about NP2cFbYU being ADP8PfbGX",
      "target_vertex_label": "Missing fact links NP2cFbYU to ADP8PfbGX",
      "edge_label": "leads to conclusion",
      "description": "The absence of direct information leads to identifying the missing fact",
      "text_correspondence": "The only way to ensure NP2cFbYU can be ADPK9s7iO is if it is ADP8PfbGX. Thus, the missing fact should be something that directly links NP2cFbYU to ADP8PfbGX"
    },
    {
      "source_vertex_label": "Rule4",
      "target_vertex_label": "No direct information about NP2cFbYU being ADP8PfbGX",
      "edge_label": "checked but insufficient",
      "description": "Rule4 was checked but does not provide direct information about NP2cFbYU being ADP8PfbGX",
      "text_correspondence": "Rule4 states that if something is ADP8PfbGX then it is ADPlPZL25. However, we don't have direct information about NP2cFbYU being ADP8PfbGX"
    },
    {
      "source_vertex_label": "Missing fact links NP2cFbYU to ADP8PfbGX",
      "target_vertex_label": "NP2cFbYU is ADP8PfbGX",
      "edge_label": "establishes",
      "description": "The missing fact would establish that NP2cFbYU is ADP8PfbGX",
      "text_correspondence": "the missing fact should be something that directly links NP2cFbYU to ADP8PfbGX"
    }
  ]
}

""",
    "musr":"""
****** Example 1:

Reasoning trace: 
1. Julius was seen with a sickle, which matched the murder weapon, and he had a motive as he stood to inherit Mackenzie's assets.\n2. However, the sickle was used for religious rituals, and Julius had an alibi for the time of the murder.\n3. Yolanda was seen on a tour with Mackenzie and had a history of fraudulent activities.\n4. Yolanda had a motive to silence Mackenzie who had evidence of her wrongdoing.\n5. Yolanda's behavior suggested she was nervous and possibly guilty when questioned about Mackenzie.\n6. The evidence from the lawyer's office confirmed Yolanda's past fraudulent activities and Mackenzie's intention to expose her.\n\nGiven these points, Yolanda seems more likely to be the murderer due to her direct involvement in the fraudulent activities and her motive to silence Mackenzie.

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "Julius seen with sickle",
      "description": "Observation that Julius was seen with a sickle matching the murder weapon",
      "text_correspondence": "Julius was seen with a sickle, which matched the murder weapon"
    },
    {
      "vertex_id": "v2",
      "label": "Julius motive: inherit Mackenzie's assets",
      "description": "Julius had a motive as he stood to inherit Mackenzie's assets",
      "text_correspondence": "he had a motive as he stood to inherit Mackenzie's assets"
    },
    {
      "vertex_id": "v3",
      "label": "Sickle used for religious rituals",
      "description": "The sickle had a legitimate purpose for religious rituals",
      "text_correspondence": "the sickle was used for religious rituals"
    },
    {
      "vertex_id": "v4",
      "label": "Julius had alibi",
      "description": "Julius had an alibi for the time of the murder",
      "text_correspondence": "Julius had an alibi for the time of the murder"
    },
    {
      "vertex_id": "v5",
      "label": "Yolanda seen on tour with Mackenzie",
      "description": "Yolanda was seen on a tour with Mackenzie",
      "text_correspondence": "Yolanda was seen on a tour with Mackenzie"
    },
    {
      "vertex_id": "v6",
      "label": "Yolanda history of fraudulent activities",
      "description": "Yolanda had a history of fraudulent activities",
      "text_correspondence": "had a history of fraudulent activities"
    },
    {
      "vertex_id": "v7",
      "label": "Yolanda motive: silence Mackenzie",
      "description": "Yolanda had a motive to silence Mackenzie who had evidence of her wrongdoing",
      "text_correspondence": "Yolanda had a motive to silence Mackenzie who had evidence of her wrongdoing"
    },
    {
      "vertex_id": "v8",
      "label": "Yolanda nervous and possibly guilty",
      "description": "Yolanda's behavior suggested she was nervous and possibly guilty when questioned",
      "text_correspondence": "Yolanda's behavior suggested she was nervous and possibly guilty when questioned about Mackenzie"
    },
    {
      "vertex_id": "v9",
      "label": "Lawyer's office evidence",
      "description": "Evidence from the lawyer's office confirmed Yolanda's past fraud and Mackenzie's intention to expose her",
      "text_correspondence": "The evidence from the lawyer's office confirmed Yolanda's past fraudulent activities and Mackenzie's intention to expose her"
    },
    {
      "vertex_id": "v10",
      "label": "Yolanda likely the murderer",
      "description": "Conclusion that Yolanda is more likely the murderer",
      "text_correspondence": "Yolanda seems more likely to be the murderer due to her direct involvement in the fraudulent activities and her motive to silence Mackenzie"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "Sickle used for religious rituals",
      "target_vertex_label": "Julius seen with sickle",
      "edge_label": "explains away",
      "description": "The religious ritual purpose undermines the suspicion from Julius having the sickle",
      "text_correspondence": "However, the sickle was used for religious rituals"
    },
    {
      "source_vertex_label": "Julius had alibi",
      "target_vertex_label": "Julius motive: inherit Mackenzie's assets",
      "edge_label": "undermines",
      "description": "The alibi weakens the case against Julius despite his motive",
      "text_correspondence": "However, the sickle was used for religious rituals, and Julius had an alibi for the time of the murder"
    },
    {
      "source_vertex_label": "Yolanda history of fraudulent activities",
      "target_vertex_label": "Yolanda motive: silence Mackenzie",
      "edge_label": "establishes motive",
      "description": "Yolanda's fraudulent history gives her motive to silence Mackenzie who had evidence",
      "text_correspondence": "Yolanda had a motive to silence Mackenzie who had evidence of her wrongdoing"
    },
    {
      "source_vertex_label": "Lawyer's office evidence",
      "target_vertex_label": "Yolanda history of fraudulent activities",
      "edge_label": "confirms",
      "description": "The lawyer's office evidence confirmed Yolanda's past fraudulent activities",
      "text_correspondence": "The evidence from the lawyer's office confirmed Yolanda's past fraudulent activities"
    },
    {
      "source_vertex_label": "Lawyer's office evidence",
      "target_vertex_label": "Yolanda motive: silence Mackenzie",
      "edge_label": "confirms",
      "description": "The evidence confirmed Mackenzie's intention to expose Yolanda, reinforcing her motive",
      "text_correspondence": "The evidence from the lawyer's office confirmed Yolanda's past fraudulent activities and Mackenzie's intention to expose her"
    },
    {
      "source_vertex_label": "Yolanda motive: silence Mackenzie",
      "target_vertex_label": "Yolanda likely the murderer",
      "edge_label": "supports conclusion",
      "description": "Yolanda's motive to silence Mackenzie supports the conclusion she is the murderer",
      "text_correspondence": "Yolanda seems more likely to be the murderer due to her direct involvement in the fraudulent activities and her motive to silence Mackenzie"
    },
    {
      "source_vertex_label": "Yolanda history of fraudulent activities",
      "target_vertex_label": "Yolanda likely the murderer",
      "edge_label": "supports conclusion",
      "description": "Direct involvement in fraudulent activities supports the conclusion",
      "text_correspondence": "Yolanda seems more likely to be the murderer due to her direct involvement in the fraudulent activities"
    },
    {
      "source_vertex_label": "Yolanda nervous and possibly guilty",
      "target_vertex_label": "Yolanda likely the murderer",
      "edge_label": "supports conclusion",
      "description": "Nervous behavior when questioned suggests guilt",
      "text_correspondence": "Yolanda's behavior suggested she was nervous and possibly guilty when questioned about Mackenzie"
    }
  ]
}



********** Example 2:

Reasoning trace: 
1. The story establishes that both Sarah and Emma are meticulous about organizing tools and keeping them in specific locations.\n2. The secateurs are mentioned being relocated to the backyard by Emma, indicating it is a place where they might be used.\n3. The front garden is where Sarah places the trowel for immediate use, suggesting it is a location for temporary placement rather than storage.\n4. The tool shed is explicitly mentioned as the primary storage location for tools like the trowel and secateurs.\n5. Given Mr. Brown's awareness of the locations of tools and his reliance on Sarah and Emma's organization, it is logical to assume he would expect the secateurs to be in the tool shed.\n\nTherefore, the most likely place Mr. Brown would look for the secateurs is the tool shed.

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "Sarah and Emma are meticulous organizers",
      "description": "Establishes that both Sarah and Emma carefully organize tools and keep them in specific locations",
      "text_correspondence": "The story establishes that both Sarah and Emma are meticulous about organizing tools and keeping them in specific locations."
    },
    {
      "vertex_id": "v2",
      "label": "Secateurs relocated to backyard by Emma",
      "description": "Fact that Emma moved the secateurs to the backyard, indicating a place of use",
      "text_correspondence": "The secateurs are mentioned being relocated to the backyard by Emma, indicating it is a place where they might be used."
    },
    {
      "vertex_id": "v3",
      "label": "Front garden is temporary placement location",
      "description": "Sarah places the trowel in the front garden for immediate use, marking it as a temporary location",
      "text_correspondence": "The front garden is where Sarah places the trowel for immediate use, suggesting it is a location for temporary placement rather than storage."
    },
    {
      "vertex_id": "v4",
      "label": "Tool shed is primary storage location",
      "description": "The tool shed is explicitly the main storage place for tools like the trowel and secateurs",
      "text_correspondence": "The tool shed is explicitly mentioned as the primary storage location for tools like the trowel and secateurs."
    },
    {
      "vertex_id": "v5",
      "label": "Mr. Brown's awareness and reliance on organization",
      "description": "Mr. Brown knows where tools are and relies on Sarah and Emma's organizational habits",
      "text_correspondence": "Given Mr. Brown's awareness of the locations of tools and his reliance on Sarah and Emma's organization"
    },
    {
      "vertex_id": "v6",
      "label": "Mr. Brown would look in the tool shed",
      "description": "Conclusion that Mr. Brown would expect the secateurs to be in the tool shed",
      "text_correspondence": "it is logical to assume he would expect the secateurs to be in the tool shed."
    }
  ],
  "edges": [
    {
      "source_vertex_label": "Secateurs relocated to backyard by Emma",
      "target_vertex_label": "Tool shed is primary storage location",
      "edge_label": "backyard is use-location vs. storage-location",
      "description": "The backyard is where secateurs might be used, but the tool shed is the primary storage location, contrasting temporary use with permanent storage",
      "text_correspondence": "The secateurs are mentioned being relocated to the backyard by Emma, indicating it is a place where they might be used. ... The tool shed is explicitly mentioned as the primary storage location for tools like the trowel and secateurs."
    },
    {
      "source_vertex_label": "Front garden is temporary placement location",
      "target_vertex_label": "Tool shed is primary storage location",
      "edge_label": "temporary placement vs. primary storage",
      "description": "The front garden is contrasted as a temporary placement location against the tool shed as the primary storage location",
      "text_correspondence": "The front garden is where Sarah places the trowel for immediate use, suggesting it is a location for temporary placement rather than storage. The tool shed is explicitly mentioned as the primary storage location"
    },
    {
      "source_vertex_label": "Mr. Brown's awareness and reliance on organization",
      "target_vertex_label": "Mr. Brown would look in the tool shed",
      "edge_label": "leads to expectation",
      "description": "Mr. Brown's awareness of tool locations and reliance on Sarah and Emma's organization leads him to expect the secateurs in the tool shed",
      "text_correspondence": "Given Mr. Brown's awareness of the locations of tools and his reliance on Sarah and Emma's organization, it is logical to assume he would expect the secateurs to be in the tool shed."
    },
    {
      "source_vertex_label": "Sarah and Emma are meticulous organizers",
      "target_vertex_label": "Mr. Brown's awareness and reliance on organization",
      "edge_label": "Mr. Brown relies on their organization",
      "description": "Mr. Brown's reliance is grounded in Sarah and Emma's meticulous organizing habits",
      "text_correspondence": "The story establishes that both Sarah and Emma are meticulous about organizing tools and keeping them in specific locations. ... his reliance on Sarah and Emma's organization"
    },
    {
      "source_vertex_label": "Tool shed is primary storage location",
      "target_vertex_label": "Mr. Brown would look in the tool shed",
      "edge_label": "expected default location",
      "description": "Because the tool shed is the primary storage location, Mr. Brown would expect the secateurs there",
      "text_correspondence": "The tool shed is explicitly mentioned as the primary storage location for tools like the trowel and secateurs. ... he would expect the secateurs to be in the tool shed."
    }
  ]
}

""",
    "goemotion": """
****** Example 1:

Reasoning trace: 
The text includes a series of laughing emojis followed by a statement indicating that something is incorrect. The laughter suggests amusement or amusement mixed with mild frustration, while the statement implies confusion or irritation at receiving the wrong item. However, the primary emotion conveyed is amusement as the person finds the situation humorous despite the error.

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "Series of laughing emojis",
      "description": "Textual cue observed in the text",
      "text_correspondence": "The text includes a series of laughing emojis"
    },
    {
      "vertex_id": "v2",
      "label": "Statement indicating something is incorrect",
      "description": "Textual cue observed in the text",
      "text_correspondence": "followed by a statement indicating that something is incorrect"
    },
    {
      "vertex_id": "v3",
      "label": "Amusement",
      "description": "Emotion suggested by the laughing emojis",
      "text_correspondence": "The laughter suggests amusement"
    },
    {
      "vertex_id": "v4",
      "label": "Amusement mixed with mild frustration",
      "description": "Alternative emotional interpretation of the laughter",
      "text_correspondence": "amusement mixed with mild frustration"
    },
    {
      "vertex_id": "v5",
      "label": "Confusion or irritation at receiving the wrong item",
      "description": "Emotion implied by the statement about incorrectness",
      "text_correspondence": "the statement implies confusion or irritation at receiving the wrong item"
    },
    {
      "vertex_id": "v6",
      "label": "Primary emotion is amusement",
      "description": "Final emotion classification conclusion",
      "text_correspondence": "the primary emotion conveyed is amusement as the person finds the situation humorous despite the error"
    },
    {
      "vertex_id": "v7",
      "label": "Person finds the situation humorous despite the error",
      "description": "Reasoning supporting the primary emotion classification",
      "text_correspondence": "the person finds the situation humorous despite the error"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "Series of laughing emojis",
      "target_vertex_label": "Amusement",
      "edge_label": "suggests",
      "description": "The laughing emojis are interpreted as suggesting amusement",
      "text_correspondence": "The laughter suggests amusement"
    },
    {
      "source_vertex_label": "Series of laughing emojis",
      "target_vertex_label": "Amusement mixed with mild frustration",
      "edge_label": "suggests",
      "description": "The laughter also suggests amusement mixed with mild frustration",
      "text_correspondence": "The laughter suggests amusement or amusement mixed with mild frustration"
    },
    {
      "source_vertex_label": "Statement indicating something is incorrect",
      "target_vertex_label": "Confusion or irritation at receiving the wrong item",
      "edge_label": "implies",
      "description": "The statement about incorrectness implies confusion or irritation",
      "text_correspondence": "the statement implies confusion or irritation at receiving the wrong item"
    },
    {
      "source_vertex_label": "Person finds the situation humorous despite the error",
      "target_vertex_label": "Primary emotion is amusement",
      "edge_label": "supports",
      "description": "Finding the situation humorous despite the error supports amusement as the primary emotion",
      "text_correspondence": "the primary emotion conveyed is amusement as the person finds the situation humorous despite the error"
    }
  ]
}



********** Example 2:

Reasoning trace: 
The text describes someone who frequently hides and is often the subject of complaints. The phrase \"stop dying\" suggests frustration or annoyance with something (likely a video game character), while \"hide BEFORE he hits 4 meat\" implies a repetitive and possibly exasperating situation where the character is being hit repeatedly. This context points towards the emotion of annoyance.

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "someone who frequently hides",
      "description": "Subject described in the text with hiding behavior",
      "text_correspondence": "The text describes someone who frequently hides"
    },
    {
      "vertex_id": "v2",
      "label": "subject of complaints",
      "description": "Characteristic of the person being described",
      "text_correspondence": "is often the subject of complaints"
    },
    {
      "vertex_id": "v3",
      "label": "phrase 'stop dying'",
      "description": "Textual cue indicating emotional tone",
      "text_correspondence": "The phrase \"stop dying\""
    },
    {
      "vertex_id": "v4",
      "label": "frustration or annoyance",
      "description": "Emotion suggested by textual cue",
      "text_correspondence": "suggests frustration or annoyance with something"
    },
    {
      "vertex_id": "v5",
      "label": "video game character",
      "description": "Likely subject of the frustration",
      "text_correspondence": "(likely a video game character)"
    },
    {
      "vertex_id": "v6",
      "label": "phrase 'hide BEFORE he hits 4 meat'",
      "description": "Textual cue implying repetitive situation",
      "text_correspondence": "\"hide BEFORE he hits 4 meat\""
    },
    {
      "vertex_id": "v7",
      "label": "repetitive and possibly exasperating situation",
      "description": "Situation inferred from textual cue",
      "text_correspondence": "implies a repetitive and possibly exasperating situation where the character is being hit repeatedly"
    },
    {
      "vertex_id": "v8",
      "label": "context",
      "description": "Combined situational understanding",
      "text_correspondence": "This context"
    },
    {
      "vertex_id": "v9",
      "label": "emotion of annoyance",
      "description": "Final emotion classification",
      "text_correspondence": "points towards the emotion of annoyance"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "phrase 'stop dying'",
      "target_vertex_label": "frustration or annoyance",
      "edge_label": "suggests",
      "description": "Textual phrase indicates emotional state",
      "text_correspondence": "The phrase \"stop dying\" suggests frustration or annoyance"
    },
    {
      "source_vertex_label": "frustration or annoyance",
      "target_vertex_label": "video game character",
      "edge_label": "with",
      "description": "Emotion directed at likely subject",
      "text_correspondence": "suggests frustration or annoyance with something (likely a video game character)"
    },
    {
      "source_vertex_label": "phrase 'hide BEFORE he hits 4 meat'",
      "target_vertex_label": "repetitive and possibly exasperating situation",
      "edge_label": "implies",
      "description": "Textual phrase indicates nature of situation",
      "text_correspondence": "\"hide BEFORE he hits 4 meat\" implies a repetitive and possibly exasperating situation"
    },
    {
      "source_vertex_label": "context",
      "target_vertex_label": "emotion of annoyance",
      "edge_label": "points towards",
      "description": "Overall context leads to emotion classification",
      "text_correspondence": "This context points towards the emotion of annoyance"
    }
  ]
}

""",
    "defeasible_nli": """
****** Example 1:

Reasoning trace: 
The premise indicates that PersonX practices day and night, which suggests a strong interest or dedication to playing the piano. The hypothesis states that before, PersonX needed to buy a piano, implying that they did not have one previously. The update mentions that PersonX researched piano deals online, which could indicate several possibilities: they might be looking for a better deal on a piano, considering alternatives to buying one, or simply gathering information. However, the fact that PersonX is researching piano deals online does not necessarily imply that they need to buy a piano; they might already own one or be exploring other options like renting or borrowing.\n\nGiven these considerations, the update does not provide direct evidence that strengthens the hypothesis that PersonX needed to buy a piano. Instead, it introduces some ambiguity about their current situation regarding the piano. Therefore, the update slightly weakens the hypothesis by introducing uncertainty about whether PersonX still needs to buy a piano.

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "PersonX practices day and night",
      "description": "Premise observation about PersonX's behavior",
      "text_correspondence": "The premise indicates that PersonX practices day and night"
    },
    {
      "vertex_id": "v2",
      "label": "Strong interest or dedication to playing the piano",
      "description": "Inference drawn from the practice behavior",
      "text_correspondence": "which suggests a strong interest or dedication to playing the piano"
    },
    {
      "vertex_id": "v3",
      "label": "PersonX needed to buy a piano",
      "description": "Hypothesis being evaluated",
      "text_correspondence": "The hypothesis states that before, PersonX needed to buy a piano"
    },
    {
      "vertex_id": "v4",
      "label": "PersonX did not have a piano previously",
      "description": "Implication from the hypothesis",
      "text_correspondence": "implying that they did not have one previously"
    },
    {
      "vertex_id": "v5",
      "label": "PersonX researched piano deals online",
      "description": "Update information about PersonX's action",
      "text_correspondence": "The update mentions that PersonX researched piano deals online"
    },
    {
      "vertex_id": "v6",
      "label": "Several possibilities indicated",
      "description": "Multiple interpretations of the research action",
      "text_correspondence": "which could indicate several possibilities"
    },
    {
      "vertex_id": "v7",
      "label": "Looking for a better deal on a piano",
      "description": "First possible interpretation",
      "text_correspondence": "they might be looking for a better deal on a piano"
    },
    {
      "vertex_id": "v8",
      "label": "Considering alternatives to buying",
      "description": "Second possible interpretation",
      "text_correspondence": "considering alternatives to buying one"
    },
    {
      "vertex_id": "v9",
      "label": "Simply gathering information",
      "description": "Third possible interpretation",
      "text_correspondence": "or simply gathering information"
    },
    {
      "vertex_id": "v10",
      "label": "Research does not necessarily imply need to buy",
      "description": "Key reasoning point about the update",
      "text_correspondence": "the fact that PersonX is researching piano deals online does not necessarily imply that they need to buy a piano"
    },
    {
      "vertex_id": "v11",
      "label": "Might already own one or exploring other options",
      "description": "Alternative explanations for the research",
      "text_correspondence": "they might already own one or be exploring other options like renting or borrowing"
    },
    {
      "vertex_id": "v12",
      "label": "Update does not strengthen hypothesis",
      "description": "Conclusion about relationship between update and hypothesis",
      "text_correspondence": "the update does not provide direct evidence that strengthens the hypothesis that PersonX needed to buy a piano"
    },
    {
      "vertex_id": "v13",
      "label": "Ambiguity about current piano situation",
      "description": "Effect of the update",
      "text_correspondence": "it introduces some ambiguity about their current situation regarding the piano"
    },
    {
      "vertex_id": "v14",
      "label": "Update weakens hypothesis",
      "description": "Final conclusion",
      "text_correspondence": "the update slightly weakens the hypothesis by introducing uncertainty about whether PersonX still needs to buy a piano"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "PersonX practices day and night",
      "target_vertex_label": "Strong interest or dedication to playing the piano",
      "edge_label": "suggests",
      "description": "The practice behavior suggests dedication",
      "text_correspondence": "The premise indicates that PersonX practices day and night, which suggests a strong interest or dedication to playing the piano"
    },
    {
      "source_vertex_label": "PersonX needed to buy a piano",
      "target_vertex_label": "PersonX did not have a piano previously",
      "edge_label": "implies",
      "description": "The hypothesis implies prior lack of ownership",
      "text_correspondence": "The hypothesis states that before, PersonX needed to buy a piano, implying that they did not have one previously"
    },
    {
      "source_vertex_label": "PersonX researched piano deals online",
      "target_vertex_label": "Several possibilities indicated",
      "edge_label": "could indicate",
      "description": "The research action opens multiple interpretations",
      "text_correspondence": "The update mentions that PersonX researched piano deals online, which could indicate several possibilities"
    },
    {
      "source_vertex_label": "PersonX researched piano deals online",
      "target_vertex_label": "Research does not necessarily imply need to buy",
      "edge_label": "does not necessarily imply",
      "description": "The research action does not confirm the need to buy",
      "text_correspondence": "the fact that PersonX is researching piano deals online does not necessarily imply that they need to buy a piano"
    },
    {
      "source_vertex_label": "Update does not strengthen hypothesis",
      "target_vertex_label": "Ambiguity about current piano situation",
      "edge_label": "instead introduces",
      "description": "Rather than strengthening, the update creates ambiguity",
      "text_correspondence": "the update does not provide direct evidence that strengthens the hypothesis that PersonX needed to buy a piano. Instead, it introduces some ambiguity about their current situation regarding the piano"
    },
    {
      "source_vertex_label": "Ambiguity about current piano situation",
      "target_vertex_label": "Update weakens hypothesis",
      "edge_label": "therefore leads to",
      "description": "The ambiguity results in weakening the hypothesis",
      "text_correspondence": "it introduces some ambiguity about their current situation regarding the piano. Therefore, the update slightly weakens the hypothesis by introducing uncertainty about whether PersonX still needs to buy a piano"
    }
  ]
}



********** Example 2:

Reasoning trace: 
The premise establishes that PersonX has helped PersonY by tuning their guitar. The hypothesis suggests that as a result of this action, PersonX now wants help. The update informs us that PersonY is learning the guitar, which implies they might need additional assistance or guidance in the future.\n\nGiven that PersonY is still learning the guitar, it is plausible that PersonX might want help from others to assist PersonY effectively, especially if PersonX has experience or knowledge that could be useful. This suggests that PersonX might feel the need for support in some capacity, which supports the hypothesis that PersonX wants help.\n\nTherefore, the update provides a context that makes the hypothesis more likely. STRENGTHENS

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "PersonX helped PersonY by tuning their guitar",
      "description": "The premise establishing the initial action",
      "text_correspondence": "The premise establishes that PersonX has helped PersonY by tuning their guitar."
    },
    {
      "vertex_id": "v2",
      "label": "PersonX now wants help",
      "description": "The hypothesis about the result of the action",
      "text_correspondence": "The hypothesis suggests that as a result of this action, PersonX now wants help."
    },
    {
      "vertex_id": "v3",
      "label": "PersonY is learning the guitar",
      "description": "The update providing new context",
      "text_correspondence": "The update informs us that PersonY is learning the guitar"
    },
    {
      "vertex_id": "v4",
      "label": "PersonY might need additional assistance or guidance",
      "description": "Implication drawn from the update",
      "text_correspondence": "which implies they might need additional assistance or guidance in the future."
    },
    {
      "vertex_id": "v5",
      "label": "PersonX might want help from others to assist PersonY effectively",
      "description": "Plausible consequence given PersonY is still learning",
      "text_correspondence": "it is plausible that PersonX might want help from others to assist PersonY effectively"
    },
    {
      "vertex_id": "v6",
      "label": "PersonX might feel the need for support",
      "description": "PersonX's resulting disposition supporting the hypothesis",
      "text_correspondence": "This suggests that PersonX might feel the need for support in some capacity"
    },
    {
      "vertex_id": "v7",
      "label": "Update makes hypothesis more likely",
      "description": "Final judgment on the defeasible relation",
      "text_correspondence": "the update provides a context that makes the hypothesis more likely."
    },
    {
      "vertex_id": "v8",
      "label": "STRENGTHENS",
      "description": "The final classification of the defeasible NLI relation",
      "text_correspondence": "STRENGTHENS"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "PersonX helped PersonY by tuning their guitar",
      "target_vertex_label": "PersonX now wants help",
      "edge_label": "hypothesis suggests result",
      "description": "The hypothesis proposes that the premise action leads to PersonX wanting help",
      "text_correspondence": "The hypothesis suggests that as a result of this action, PersonX now wants help."
    },
    {
      "source_vertex_label": "PersonY is learning the guitar",
      "target_vertex_label": "PersonY might need additional assistance or guidance",
      "edge_label": "implies",
      "description": "Learning the guitar implies needing further help",
      "text_correspondence": "PersonY is learning the guitar, which implies they might need additional assistance or guidance in the future."
    },
    {
      "source_vertex_label": "PersonY is learning the guitar",
      "target_vertex_label": "PersonX might want help from others to assist PersonY effectively",
      "edge_label": "makes plausible",
      "description": "PersonY still learning makes it plausible PersonX seeks help to assist them",
      "text_correspondence": "Given that PersonY is still learning the guitar, it is plausible that PersonX might want help from others to assist PersonY effectively"
    },
    {
      "source_vertex_label": "PersonX might want help from others to assist PersonY effectively",
      "target_vertex_label": "PersonX might feel the need for support",
      "edge_label": "suggests",
      "description": "Wanting help from others suggests feeling the need for support",
      "text_correspondence": "This suggests that PersonX might feel the need for support in some capacity"
    },
    {
      "source_vertex_label": "PersonX might feel the need for support",
      "target_vertex_label": "PersonX now wants help",
      "edge_label": "supports hypothesis",
      "description": "Feeling the need for support backs the hypothesis that PersonX wants help",
      "text_correspondence": "which supports the hypothesis that PersonX wants help."
    },
    {
      "source_vertex_label": "Update makes hypothesis more likely",
      "target_vertex_label": "STRENGTHENS",
      "edge_label": "therefore",
      "description": "The update making the hypothesis more likely leads to the STRENGTHENS judgment",
      "text_correspondence": "the update provides a context that makes the hypothesis more likely."
    }
  ]
}

""",
    "copa_guess_effect":"""
****** Example 1:

Reasoning trace: 
When someone violates parole, the typical response from the legal system is to enforce the terms of the original sentence, which often includes being sent back to jail. While it's possible that as part of the violation, she might have committed additional crimes like stealing money from a church, the direct and most immediate consequence would be her return to jail for violating the conditions of her parole. Therefore, being sent back to jail is a more direct and common response compared to committing another crime.

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "someone violates parole",
      "description": "initial condition/trigger event",
      "text_correspondence": "When someone violates parole"
    },
    {
      "vertex_id": "v2",
      "label": "typical response from legal system",
      "description": "system reaction to parole violation",
      "text_correspondence": "the typical response from the legal system"
    },
    {
      "vertex_id": "v3",
      "label": "enforce terms of original sentence",
      "description": "action taken by legal system",
      "text_correspondence": "is to enforce the terms of the original sentence"
    },
    {
      "vertex_id": "v4",
      "label": "being sent back to jail",
      "description": "consequence of enforcement",
      "text_correspondence": "which often includes being sent back to jail"
    },
    {
      "vertex_id": "v5",
      "label": "return to jail for violating parole conditions",
      "description": "direct and immediate consequence",
      "text_correspondence": "the direct and most immediate consequence would be her return to jail for violating the conditions of her parole"
    },
    {
      "vertex_id": "v6",
      "label": "being sent back to jail is more direct and common response",
      "description": "conclusion/comparison outcome",
      "text_correspondence": "being sent back to jail is a more direct and common response compared to committing another crime"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "someone violates parole",
      "target_vertex_label": "typical response from legal system",
      "edge_label": "triggers",
      "description": "parole violation causes legal system to respond",
      "text_correspondence": "When someone violates parole, the typical response from the legal system"
    },
    {
      "source_vertex_label": "typical response from legal system",
      "target_vertex_label": "enforce terms of original sentence",
      "edge_label": "is_to",
      "description": "legal system's response action",
      "text_correspondence": "the typical response from the legal system is to enforce the terms of the original sentence"
    },
    {
      "source_vertex_label": "enforce terms of original sentence",
      "target_vertex_label": "being sent back to jail",
      "edge_label": "often_includes",
      "description": "enforcement frequently results in jail",
      "text_correspondence": "enforce the terms of the original sentence, which often includes being sent back to jail"
    },
    {
      "source_vertex_label": "someone violates parole",
      "target_vertex_label": "return to jail for violating parole conditions",
      "edge_label": "direct_consequence",
      "description": "parole violation directly leads to jail return",
      "text_correspondence": "the direct and most immediate consequence would be her return to jail for violating the conditions of her parole"
    },
    {
      "source_vertex_label": "return to jail for violating parole conditions",
      "target_vertex_label": "being sent back to jail is more direct and common response",
      "edge_label": "supports_conclusion",
      "description": "direct consequence supports final reasoning",
      "text_correspondence": "her return to jail for violating the conditions of her parole. Therefore, being sent back to jail is a more direct and common response"
    }
  ]
}


********** Example 2:

Reasoning trace: 
The woman's desire to change her environment could lead to various actions, but we need to consider what typically constitutes changing one's environment. Going to church might be seen as a change in social and spiritual environment, while traveling abroad involves a significant change in physical location, living conditions, and cultural surroundings. Traveling abroad is generally more drastic and comprehensive in altering one's environment compared to attending church, which primarily changes one's social and spiritual circles rather than their physical surroundings.

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "woman's desire to change her environment",
      "description": "The initial condition or motivation being analyzed",
      "text_correspondence": "The woman's desire to change her environment"
    },
    {
      "vertex_id": "v2",
      "label": "various actions",
      "description": "Potential outcomes from the desire",
      "text_correspondence": "could lead to various actions"
    },
    {
      "vertex_id": "v3",
      "label": "going to church",
      "description": "First action option being evaluated",
      "text_correspondence": "Going to church"
    },
    {
      "vertex_id": "v4",
      "label": "change in social and spiritual environment",
      "description": "Effect of going to church",
      "text_correspondence": "might be seen as a change in social and spiritual environment"
    },
    {
      "vertex_id": "v5",
      "label": "traveling abroad",
      "description": "Second action option being evaluated",
      "text_correspondence": "traveling abroad"
    },
    {
      "vertex_id": "v6",
      "label": "significant change in physical location, living conditions, and cultural surroundings",
      "description": "Effect of traveling abroad",
      "text_correspondence": "involves a significant change in physical location, living conditions, and cultural surroundings"
    },
    {
      "vertex_id": "v7",
      "label": "traveling abroad more drastic and comprehensive",
      "description": "Comparative conclusion about traveling abroad",
      "text_correspondence": "Traveling abroad is generally more drastic and comprehensive in altering one's environment"
    },
    {
      "vertex_id": "v8",
      "label": "attending church changes social and spiritual circles rather than physical surroundings",
      "description": "Limitation of church attendance as environmental change",
      "text_correspondence": "attending church, which primarily changes one's social and spiritual circles rather than their physical surroundings"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "woman's desire to change her environment",
      "target_vertex_label": "various actions",
      "edge_label": "could lead to",
      "description": "The desire is stated as potentially causing various actions",
      "text_correspondence": "The woman's desire to change her environment could lead to various actions"
    },
    {
      "source_vertex_label": "going to church",
      "target_vertex_label": "change in social and spiritual environment",
      "edge_label": "might be seen as",
      "description": "Going to church is characterized as resulting in this type of change",
      "text_correspondence": "Going to church might be seen as a change in social and spiritual environment"
    },
    {
      "source_vertex_label": "traveling abroad",
      "target_vertex_label": "significant change in physical location, living conditions, and cultural surroundings",
      "edge_label": "involves",
      "description": "Traveling abroad is stated to involve these changes",
      "text_correspondence": "traveling abroad involves a significant change in physical location, living conditions, and cultural surroundings"
    }
  ]
}

""",
    "art": """
****** Example 1:

Reasoning trace: 
The first observation indicates that the scientist wanted to collect bacteria samples, while the second observation states that the bacteria was non-harmful. Hypothesis 1 suggests that the scientist collected samples and tested them, which would imply that the bacteria was handled and possibly analyzed for its properties. This aligns well with the idea that the bacteria could be non-harmful as a result of testing. On the other hand, Hypothesis 2 only states that the scientist collected the bacteria and froze it, without any mention of analysis or further action. While this could still lead to non-harmful bacteria, it does not directly explain why the bacteria would be non-harmful in the context of collection and handling. Therefore, Hypothesis 1 provides a more comprehensive explanation for the transition from wanting to collect samples to having non-harmful bacteria.\n

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "First observation: scientist wanted to collect bacteria samples",
      "description": "The first observation in the dataset",
      "text_correspondence": "The first observation indicates that the scientist wanted to collect bacteria samples"
    },
    {
      "vertex_id": "v2",
      "label": "Second observation: bacteria was non-harmful",
      "description": "The second observation in the dataset",
      "text_correspondence": "the second observation states that the bacteria was non-harmful"
    },
    {
      "vertex_id": "v3",
      "label": "Hypothesis 1: scientist collected samples and tested them",
      "description": "First hypothesis suggesting collection and testing",
      "text_correspondence": "Hypothesis 1 suggests that the scientist collected samples and tested them"
    },
    {
      "vertex_id": "v4",
      "label": "Bacteria was handled and possibly analyzed for its properties",
      "description": "Implication of Hypothesis 1 regarding handling and analysis",
      "text_correspondence": "which would imply that the bacteria was handled and possibly analyzed for its properties"
    },
    {
      "vertex_id": "v5",
      "label": "Bacteria could be non-harmful as a result of testing",
      "description": "Alignment between Hypothesis 1 and the second observation",
      "text_correspondence": "This aligns well with the idea that the bacteria could be non-harmful as a result of testing"
    },
    {
      "vertex_id": "v6",
      "label": "Hypothesis 2: scientist collected bacteria and froze it",
      "description": "Second hypothesis suggesting collection and freezing",
      "text_correspondence": "Hypothesis 2 only states that the scientist collected the bacteria and froze it"
    },
    {
      "vertex_id": "v7",
      "label": "No mention of analysis or further action",
      "description": "Limitation of Hypothesis 2",
      "text_correspondence": "without any mention of analysis or further action"
    },
    {
      "vertex_id": "v8",
      "label": "Does not directly explain why bacteria would be non-harmful",
      "description": "Weakness of Hypothesis 2 in explaining the second observation",
      "text_correspondence": "it does not directly explain why the bacteria would be non-harmful in the context of collection and handling"
    },
    {
      "vertex_id": "v9",
      "label": "Hypothesis 1 provides more comprehensive explanation",
      "description": "Final conclusion favoring Hypothesis 1",
      "text_correspondence": "Hypothesis 1 provides a more comprehensive explanation for the transition from wanting to collect samples to having non-harmful bacteria"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "Hypothesis 1: scientist collected samples and tested them",
      "target_vertex_label": "Bacteria was handled and possibly analyzed for its properties",
      "edge_label": "implies",
      "description": "Hypothesis 1 implies that bacteria was handled and analyzed",
      "text_correspondence": "Hypothesis 1 suggests that the scientist collected samples and tested them, which would imply that the bacteria was handled and possibly analyzed for its properties"
    },
    {
      "source_vertex_label": "Bacteria was handled and possibly analyzed for its properties",
      "target_vertex_label": "Bacteria could be non-harmful as a result of testing",
      "edge_label": "aligns with",
      "description": "Analysis of bacteria aligns with the finding that it is non-harmful",
      "text_correspondence": "This aligns well with the idea that the bacteria could be non-harmful as a result of testing"
    },
    {
      "source_vertex_label": "Hypothesis 2: scientist collected bacteria and froze it",
      "target_vertex_label": "No mention of analysis or further action",
      "edge_label": "lacks",
      "description": "Hypothesis 2 lacks mention of analysis or further action",
      "text_correspondence": "Hypothesis 2 only states that the scientist collected the bacteria and froze it, without any mention of analysis or further action"
    },
    {
      "source_vertex_label": "No mention of analysis or further action",
      "target_vertex_label": "Does not directly explain why bacteria would be non-harmful",
      "edge_label": "leads to",
      "description": "Lack of analysis means Hypothesis 2 cannot directly explain non-harmful bacteria",
      "text_correspondence": "it does not directly explain why the bacteria would be non-harmful in the context of collection and handling"
    },
    {
      "source_vertex_label": "Bacteria could be non-harmful as a result of testing",
      "target_vertex_label": "Hypothesis 1 provides more comprehensive explanation",
      "edge_label": "supports",
      "description": "The alignment of Hypothesis 1 with observations supports it as the more comprehensive explanation",
      "text_correspondence": "Hypothesis 1 provides a more comprehensive explanation for the transition from wanting to collect samples to having non-harmful bacteria"
    },
    {
      "source_vertex_label": "Does not directly explain why bacteria would be non-harmful",
      "target_vertex_label": "Hypothesis 1 provides more comprehensive explanation",
      "edge_label": "contrasts with",
      "description": "Hypothesis 2's weakness contrasts with Hypothesis 1's strength, leading to the conclusion",
      "text_correspondence": "Therefore, Hypothesis 1 provides a more comprehensive explanation for the transition from wanting to collect samples to having non-harmful bacteria"
    }
  ]
}


********** Example 2:

Reasoning trace: 
Liv wanting to dance ballet (Observation 1) could be a result of her being talented or her mother signing her up. However, the reaction of her classmates being jealous and shocked (Observation 2) suggests that Liv must have some level of talent or skill that sets her apart, making Hypothesis 2 (She was not very talented) less likely. Therefore, Hypothesis 1 (Liv's mother signed her up) better explains why her classmates would react with jealousy and shock, as it implies she has been given an opportunity others did not have.\n

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "Liv wanting to dance ballet",
      "description": "First observation about Liv's desire",
      "text_correspondence": "Liv wanting to dance ballet (Observation 1)"
    },
    {
      "vertex_id": "v2",
      "label": "Classmates being jealous and shocked",
      "description": "Second observation about classmates' reaction",
      "text_correspondence": "the reaction of her classmates being jealous and shocked (Observation 2)"
    },
    {
      "vertex_id": "v3",
      "label": "Liv has talent or skill that sets her apart",
      "description": "Inference from classmates' reaction",
      "text_correspondence": "suggests that Liv must have some level of talent or skill that sets her apart"
    },
    {
      "vertex_id": "v4",
      "label": "Hypothesis 2 less likely",
      "description": "Conclusion about Hypothesis 2",
      "text_correspondence": "making Hypothesis 2 (She was not very talented) less likely"
    },
    {
      "vertex_id": "v5",
      "label": "Hypothesis 1 better explains classmates' reaction",
      "description": "Conclusion about Hypothesis 1's explanatory power",
      "text_correspondence": "Hypothesis 1 (Liv's mother signed her up) better explains why her classmates would react with jealousy and shock"
    },
    {
      "vertex_id": "v6",
      "label": "Liv given opportunity others did not have",
      "description": "Implication of Hypothesis 1",
      "text_correspondence": "as it implies she has been given an opportunity others did not have"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "Classmates being jealous and shocked",
      "target_vertex_label": "Liv has talent or skill that sets her apart",
      "edge_label": "suggests",
      "description": "The classmates' reaction suggests Liv's talent",
      "text_correspondence": "the reaction of her classmates being jealous and shocked (Observation 2) suggests that Liv must have some level of talent or skill that sets her apart"
    },
    {
      "source_vertex_label": "Liv has talent or skill that sets her apart",
      "target_vertex_label": "Hypothesis 2 less likely",
      "edge_label": "makes",
      "description": "Liv's talent makes Hypothesis 2 less likely",
      "text_correspondence": "Liv must have some level of talent or skill that sets her apart, making Hypothesis 2 (She was not very talented) less likely"
    },
    {
      "source_vertex_label": "Hypothesis 1 better explains classmates' reaction",
      "target_vertex_label": "Liv given opportunity others did not have",
      "edge_label": "implies",
      "description": "Hypothesis 1 implies Liv received a special opportunity",
      "text_correspondence": "Hypothesis 1 (Liv's mother signed her up) better explains why her classmates would react with jealousy and shock, as it implies she has been given an opportunity others did not have"
    }
  ]
}
""",
    "medqa": """
****** Example 1:

Reasoning trace: 
1. The patient has a history of unstable angina pectoris treated with stenting, and now presents with decreased urinary output and malaise.\n2. Physical examination reveals mottled, reticulated purplish discoloration of the feet, which could indicate ischemia or microvascular disease.\n3. Laboratory findings include elevated serum urea nitrogen (25 mg/dL) and creatinine (4.2 mg/dL), indicating renal dysfunction.\n4. Renal biopsy shows intravascular spindle-shaped vacuoles, which are characteristic of cholesterol embolism.\n5. Cholesterol embolism can occur after invasive procedures like cardiac catheterization, leading to acute renal failure and systemic manifestations such as purpura due to microvascular occlusion.\n\nBased on the clinical presentation and laboratory findings, the most likely cause is cholesterol embolism.\n\n

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "History of unstable angina pectoris treated with stenting",
      "description": "Patient's relevant medical history including prior invasive cardiac procedure",
      "text_correspondence": "The patient has a history of unstable angina pectoris treated with stenting"
    },
    {
      "vertex_id": "v2",
      "label": "Decreased urinary output and malaise",
      "description": "Current presenting symptoms of the patient",
      "text_correspondence": "now presents with decreased urinary output and malaise"
    },
    {
      "vertex_id": "v3",
      "label": "Mottled, reticulated purplish discoloration of the feet",
      "description": "Physical examination finding suggesting ischemia or microvascular disease",
      "text_correspondence": "Physical examination reveals mottled, reticulated purplish discoloration of the feet"
    },
    {
      "vertex_id": "v4",
      "label": "Ischemia or microvascular disease",
      "description": "Possible pathological process indicated by the skin findings",
      "text_correspondence": "which could indicate ischemia or microvascular disease"
    },
    {
      "vertex_id": "v5",
      "label": "Elevated serum urea nitrogen and creatinine",
      "description": "Laboratory findings indicating renal dysfunction",
      "text_correspondence": "Laboratory findings include elevated serum urea nitrogen (25 mg/dL) and creatinine (4.2 mg/dL)"
    },
    {
      "vertex_id": "v6",
      "label": "Renal dysfunction",
      "description": "Clinical interpretation of the laboratory abnormalities",
      "text_correspondence": "indicating renal dysfunction"
    },
    {
      "vertex_id": "v7",
      "label": "Intravascular spindle-shaped vacuoles on renal biopsy",
      "description": "Histopathological finding on renal biopsy characteristic of cholesterol embolism",
      "text_correspondence": "Renal biopsy shows intravascular spindle-shaped vacuoles"
    },
    {
      "vertex_id": "v8",
      "label": "Cholesterol embolism",
      "description": "The diagnosed condition and most likely cause of the patient's presentation",
      "text_correspondence": "which are characteristic of cholesterol embolism"
    },
    {
      "vertex_id": "v9",
      "label": "Invasive procedures like cardiac catheterization",
      "description": "Known precipitant of cholesterol embolism",
      "text_correspondence": "Cholesterol embolism can occur after invasive procedures like cardiac catheterization"
    },
    {
      "vertex_id": "v10",
      "label": "Acute renal failure",
      "description": "Systemic consequence of cholesterol embolism",
      "text_correspondence": "leading to acute renal failure"
    },
    {
      "vertex_id": "v11",
      "label": "Purpura due to microvascular occlusion",
      "description": "Systemic manifestation of cholesterol embolism",
      "text_correspondence": "systemic manifestations such as purpura due to microvascular occlusion"
    }
  ],
  "edges": [
    {
      "source_vertex_label": "Mottled, reticulated purplish discoloration of the feet",
      "target_vertex_label": "Ischemia or microvascular disease",
      "edge_label": "could indicate",
      "description": "The skin finding on physical examination suggests ischemia or microvascular disease",
      "text_correspondence": "mottled, reticulated purplish discoloration of the feet, which could indicate ischemia or microvascular disease"
    },
    {
      "source_vertex_label": "Elevated serum urea nitrogen and creatinine",
      "target_vertex_label": "Renal dysfunction",
      "edge_label": "indicating",
      "description": "Elevated lab values point to renal dysfunction",
      "text_correspondence": "elevated serum urea nitrogen (25 mg/dL) and creatinine (4.2 mg/dL), indicating renal dysfunction"
    },
    {
      "source_vertex_label": "Intravascular spindle-shaped vacuoles on renal biopsy",
      "target_vertex_label": "Cholesterol embolism",
      "edge_label": "characteristic of",
      "description": "The biopsy finding is characteristic of cholesterol embolism",
      "text_correspondence": "intravascular spindle-shaped vacuoles, which are characteristic of cholesterol embolism"
    },
    {
      "source_vertex_label": "Invasive procedures like cardiac catheterization",
      "target_vertex_label": "Cholesterol embolism",
      "edge_label": "can lead to",
      "description": "Invasive procedures are a known precipitant of cholesterol embolism",
      "text_correspondence": "Cholesterol embolism can occur after invasive procedures like cardiac catheterization"
    },
    {
      "source_vertex_label": "Cholesterol embolism",
      "target_vertex_label": "Acute renal failure",
      "edge_label": "leading to",
      "description": "Cholesterol embolism causes acute renal failure",
      "text_correspondence": "Cholesterol embolism can occur after invasive procedures like cardiac catheterization, leading to acute renal failure"
    },
    {
      "source_vertex_label": "Cholesterol embolism",
      "target_vertex_label": "Purpura due to microvascular occlusion",
      "edge_label": "leading to",
      "description": "Cholesterol embolism causes systemic manifestations including purpura",
      "text_correspondence": "leading to acute renal failure and systemic manifestations such as purpura due to microvascular occlusion"
    },
    {
      "source_vertex_label": "History of unstable angina pectoris treated with stenting",
      "target_vertex_label": "Invasive procedures like cardiac catheterization",
      "edge_label": "treated with (invasive procedure)",
      "description": "Stenting for unstable angina constitutes the invasive procedure that can precipitate cholesterol embolism",
      "text_correspondence": "a history of unstable angina pectoris treated with stenting...Cholesterol embolism can occur after invasive procedures like cardiac catheterization"
    }
  ]
}



********** Example 2:

Reasoning trace: 
1. The patient had a history of gastric cancer treated with subtotal gastrectomy.\n2. At autopsy, there is unilateral atrophy of the neurons in the spinal cord area indicated by the arrow.\n3. The location of the lesion is crucial for determining the neurological deficits. Since the lesion is unilateral, it will affect the corresponding side of the body.\n4. The spinal cord at the level where the lesion is located is responsible for sensory and motor functions of the ipsilateral side of the body.\n5. The options provided relate to different sensory modalities:\n   - Temperature sensation is processed in the spinothalamic tract.\n   - Strength of muscles is related to motor function, not sensory.\n   - Vibratory sense is processed in the dorsal columns.\n   - Positional sense is also processed in the dorsal columns.\n   - Crude touch is processed in the dorsal columns.\n6. Given the context of the lesion being unilateral and affecting the spinal cord, the most likely deficit would be a loss of temperature sensation on the same side as the lesion due to involvement of the spinothalamic tract.\n\nTherefore, the most likely finding during the neurological examination would be a decreased sense of temperature in the ipsilateral arm.\n

Answer:
{
  "vertices": [
    {
      "vertex_id": "v1",
      "label": "History of gastric cancer",
      "description": "Clinical background of the patient",
      "text_correspondence": "The patient had a history of gastric cancer treated with subtotal gastrectomy."
    },
    {
      "vertex_id": "v2",
      "label": "Subtotal gastrectomy",
      "description": "Surgical treatment the patient received",
      "text_correspondence": "gastric cancer treated with subtotal gastrectomy"
    },
    {
      "vertex_id": "v3",
      "label": "Unilateral atrophy of spinal cord neurons",
      "description": "Autopsy finding indicating the lesion",
      "text_correspondence": "At autopsy, there is unilateral atrophy of the neurons in the spinal cord area indicated by the arrow."
    },
    {
      "vertex_id": "v4",
      "label": "Location of the lesion",
      "description": "Anatomical site determining neurological deficits",
      "text_correspondence": "The location of the lesion is crucial for determining the neurological deficits."
    },
    {
      "vertex_id": "v5",
      "label": "Unilateral lesion affects corresponding side",
      "description": "Laterality principle of the lesion",
      "text_correspondence": "Since the lesion is unilateral, it will affect the corresponding side of the body."
    },
    {
      "vertex_id": "v6",
      "label": "Sensory and motor functions of ipsilateral side",
      "description": "Functions governed by the affected spinal cord level",
      "text_correspondence": "The spinal cord at the level where the lesion is located is responsible for sensory and motor functions of the ipsilateral side of the body."
    },
    {
      "vertex_id": "v7",
      "label": "Temperature sensation — spinothalamic tract",
      "description": "Sensory modality and its associated tract",
      "text_correspondence": "Temperature sensation is processed in the spinothalamic tract."
    },
    {
      "vertex_id": "v8",
      "label": "Strength of muscles — motor function",
      "description": "Motor modality distinguished from sensory",
      "text_correspondence": "Strength of muscles is related to motor function, not sensory."
    },
    {
      "vertex_id": "v9",
      "label": "Vibratory sense — dorsal columns",
      "description": "Sensory modality and its associated tract",
      "text_correspondence": "Vibratory sense is processed in the dorsal columns."
    },
    {
      "vertex_id": "v10",
      "label": "Positional sense — dorsal columns",
      "description": "Sensory modality and its associated tract",
      "text_correspondence": "Positional sense is also processed in the dorsal columns."
    },
    {
      "vertex_id": "v11",
      "label": "Crude touch — dorsal columns",
      "description": "Sensory modality and its associated tract",
      "text_correspondence": "Crude touch is processed in the dorsal columns."
    },
    {
      "vertex_id": "v12",
      "label": "Involvement of the spinothalamic tract",
      "description": "Tract affected by the unilateral lesion",
      "text_correspondence": "the most likely deficit would be a loss of temperature sensation on the same side as the lesion due to involvement of the spinothalamic tract"
    },
    {
      "vertex_id": "v13",
      "label": "Loss of temperature sensation on same side",
      "description": "Predicted neurological deficit from the lesion",
      "text_correspondence": "a loss of temperature sensation on the same side as the lesion"
    },
    {
      "vertex_id": "v14",
      "label": "Decreased sense of temperature in ipsilateral arm",
      "description": "Final answer — most likely neurological examination finding",
      "text_correspondence": "the most likely finding during the neurological examination would be a decreased sense of temperature in the ipsilateral arm."
    }
  ],
  "edges": [
    {
      "source_vertex_label": "History of gastric cancer",
      "target_vertex_label": "Subtotal gastrectomy",
      "edge_label": "treated with",
      "description": "Gastric cancer was treated by subtotal gastrectomy",
      "text_correspondence": "gastric cancer treated with subtotal gastrectomy"
    },
    {
      "source_vertex_label": "Location of the lesion",
      "target_vertex_label": "Unilateral lesion affects corresponding side",
      "edge_label": "determines laterality",
      "description": "The lesion location and its unilateral nature determine which side is affected",
      "text_correspondence": "The location of the lesion is crucial for determining the neurological deficits. Since the lesion is unilateral, it will affect the corresponding side of the body."
    },
    {
      "source_vertex_label": "Unilateral atrophy of spinal cord neurons",
      "target_vertex_label": "Location of the lesion",
      "edge_label": "defines",
      "description": "The autopsy finding of unilateral atrophy defines the lesion location",
      "text_correspondence": "there is unilateral atrophy of the neurons in the spinal cord area indicated by the arrow. The location of the lesion is crucial for determining the neurological deficits."
    },
    {
      "source_vertex_label": "Unilateral lesion affects corresponding side",
      "target_vertex_label": "Sensory and motor functions of ipsilateral side",
      "edge_label": "affects",
      "description": "The unilateral lesion affects ipsilateral sensory and motor functions",
      "text_correspondence": "Since the lesion is unilateral, it will affect the corresponding side of the body. The spinal cord at the level where the lesion is located is responsible for sensory and motor functions of the ipsilateral side of the body."
    },
    {
      "source_vertex_label": "Unilateral atrophy of spinal cord neurons",
      "target_vertex_label": "Involvement of the spinothalamic tract",
      "edge_label": "causes",
      "description": "The unilateral spinal cord lesion involves the spinothalamic tract",
      "text_correspondence": "the lesion being unilateral and affecting the spinal cord, the most likely deficit would be a loss of temperature sensation on the same side as the lesion due to involvement of the spinothalamic tract"
    },
    {
      "source_vertex_label": "Involvement of the spinothalamic tract",
      "target_vertex_label": "Loss of temperature sensation on same side",
      "edge_label": "results in",
      "description": "Spinothalamic tract involvement leads to ipsilateral temperature sensation loss",
      "text_correspondence": "a loss of temperature sensation on the same side as the lesion due to involvement of the spinothalamic tract"
    },
    {
      "source_vertex_label": "Temperature sensation — spinothalamic tract",
      "target_vertex_label": "Loss of temperature sensation on same side",
      "edge_label": "explains modality",
      "description": "Temperature sensation being processed in the spinothalamic tract explains the specific deficit",
      "text_correspondence": "Temperature sensation is processed in the spinothalamic tract. ... the most likely deficit would be a loss of temperature sensation"
    },
    {
      "source_vertex_label": "Loss of temperature sensation on same side",
      "target_vertex_label": "Decreased sense of temperature in ipsilateral arm",
      "edge_label": "manifests as",
      "description": "The loss of temperature sensation manifests as the examination finding",
      "text_correspondence": "a loss of temperature sensation on the same side as the lesion ... the most likely finding during the neurological examination would be a decreased sense of temperature in the ipsilateral arm."
    }
  ]
}

""",

}
INCLUDE_FEW_SHOT: bool = False
INCLUDE_DATASET_SPECIFIC_NOTES: bool = False

SYSTEM_PROMPT = """\
You are an expert reasoning-graph annotator.

## Task
Build an exhaustive, directed reasoning graph from the rationale text that maps the complete logical trace and argumentation path.
- Vertices are atomic logical propositions, concepts, observations, or facts explicitly present in the rationale.
- Directed edges are explicit logical entailments, reasoning steps, and conclusions made in the rationale.
- Focus strictly on the reasoning trace (argumentation, deductions, logic). Do NOT map purely grammatical or syntactical linguistic dependencies.

## Core constraints (strict)
1. Be Exhaustive: Capture the entire narrative, including all intermediate reasoning steps, minor explicit entailments, and final conclusions.
2. Maintain Granularity Balance: Vertices should represent distinct, atomic propositions. Do not bundle multiple reasoning steps into one oversized vertex, but do not divide a single cohesive proposition into meaningless fragmented pieces. Strongly emphasize separating distinct evidences, observations, prior facts, and conclusions into their own distinct vertices instead of bundling them together.
3. Strict Grounding: Every vertex and edge must be based on an exact occurrence/quote from the rationale.
4. No Inference: Do NOT add missing knowledge or infer hidden intermediate nodes. If a logical step is implied but not explicitly written, exclude it.
5. Graph may be disconnected (multiple isolated subgraphs are allowed).
6. Keep labels concise but faithful to quoted text.

## Vertex rules
For each vertex include:
- vertex_id: unique id like v1, v2, v3
- label: concise label
- description: what logical role this vertex plays (e.g., premise, intermediate deduction, observation, conclusion)
- text_correspondence: exact quote from rationale grounding the vertex

## Edge rules
For each edge include:
- source_vertex_label: label of source vertex (CRITICAL: You must use the exact vertex "label" here, NOT the vertex_id. This confusion should not happen.)
- target_vertex_label: label of target vertex (CRITICAL: You must use the exact vertex "label" here, NOT the vertex_id. This confusion should not happen.)
- edge_label: concise relation label denoting the logical connection (e.g., ENTAILS, CONTRADICTS, CONSEQUENCE_OF, SUPPORTS, REFUTES, CAUSES, EXEMPLIFIES, PREVENTS)
- description: why the logical relation exists in the rationale
- text_correspondence: exact quote from rationale grounding this explicit edge relation

Only output edges whose source and target correspond to listed vertices.

## Quality checks before final output
- Completeness: Does the graph capture the WHOLE reasoning path and explicit conclusions?
- Granularity: Are vertices balanced (neither bundling too much info nor splitting atomic thoughts)?
- Logical Focus: Do edges represent actual reasoning and argumentation rather than mere syntax?
- Grounding & Inference: Are all vertices and edges exactly quoted with ZERO inferred external logic? (Must be YES).
- Preserve directionality exactly as stated in the text.

## Dataset-specific note (current dataset only)

{dataset_specific_note}

## Few-shot demonstrations

{dataset_few_shot_examples}
"""


USER_PROMPT_TEMPLATE = """\
Dataset: {dataset}

Construct the directed rationale graph from the reasoning trace below.
Use only explicit occurrences from the trace.

<reasoning_trace>
{text}
</reasoning_trace>
"""
