"""
Dataset-specific prompt note rendering.
"""

from __future__ import annotations

import importlib
import re

CANONICAL_DATASET_ALIASES: dict[str, str] = {
    "unknown": "unknown",
    "art": "art",
    "medqa": "medqa",
    "strategyqa": "strategyqa",
    "copa": "copa_guess_effect",
    "copa_guess_effect": "copa_guess_effect",
    "defeasible_nli": "defeasible_nli",
    "goemotion": "goemotion",
    "musr_murder": "musr",
    "musr_object": "musr",
    "musr_team": "musr",
    "musr": "musr",
    "neulr_abductive": "neulr_abductive",
}

DEFAULT_DATASET_NOTE = (
    "Apply this metric strictly to the current dataset using only evidence from the reasoning trace."
)
DEFAULT_FEW_SHOT_BLOCK = ""

METRIC_PROMPT_MODULES: dict[str, str] = {
    "branchiness": "prompts.counting.branchiness",
    "backtracking": "prompts.counting.backtracking",
    "uncertainty_markers": "prompts.counting.uncertainty_markers",
    "prior": "prompts.counting.prior",
    "differential_elimination": "prompts.counting.differential_elimination",
    "evidence_explanation_directionality": "prompts.binary.evidence_explanation_directionality",
    "evidence_explanation_directionality_scorebased": "prompts.scorebased.evidence_explanation_directionality_scorebased",
    "observation_coverage": "prompts.coverage.observation_coverage",
    "rationale_graph": "prompts.graph_structure.rationale_graph",
}


def _normalize_dataset_name(dataset: str | None) -> str:
    if dataset is None:
        return "unknown"
    key = re.sub(r"[^a-z0-9_]+", "_", dataset.strip().lower()).strip("_")
    if not key:
        return "unknown"
    return CANONICAL_DATASET_ALIASES.get(key, key)


def _load_metric_prompt_maps(metric_name: str) -> tuple[dict[str, str], dict[str, str]]:
    module_path = METRIC_PROMPT_MODULES.get(metric_name)
    if not module_path:
        return {}, {}

    module = importlib.import_module(module_path)
    notes = getattr(module, "DATASET_SPECIFIC_NOTES", {})
    few_shots = getattr(module, "DATASET_FEW_SHOT_EXAMPLES", {})

    notes_map = notes if isinstance(notes, dict) else {}
    few_shots_map = few_shots if isinstance(few_shots, dict) else {}
    return notes_map, few_shots_map


def _render_dataset_note(metric_name: str, dataset: str | None) -> str:
    notes_map, _ = _load_metric_prompt_maps(metric_name)
    ds_key = _normalize_dataset_name(dataset)
    return notes_map.get(ds_key, DEFAULT_DATASET_NOTE)


def _render_dataset_few_shot(metric_name: str, dataset: str | None) -> str:
    _, few_shots_map = _load_metric_prompt_maps(metric_name)
    ds_key = _normalize_dataset_name(dataset)
    return few_shots_map.get(ds_key, DEFAULT_FEW_SHOT_BLOCK)


def _load_few_shot_flag(metric_name: str) -> bool:
    """Return the INCLUDE_FEW_SHOT flag from the metric's prompt module.

    Defaults to True when the attribute is absent (backwards-compatible).
    """
    module_path = METRIC_PROMPT_MODULES.get(metric_name)
    if not module_path:
        return True
    module = importlib.import_module(module_path)
    return bool(getattr(module, "INCLUDE_FEW_SHOT", True))


def _load_dataset_specific_notes_flag(metric_name: str) -> bool:
    """Return the INCLUDE_DATASET_SPECIFIC_NOTES flag from the metric's prompt module.

    Defaults to True when the attribute is absent (backwards-compatible).
    """
    module_path = METRIC_PROMPT_MODULES.get(metric_name)
    if not module_path:
        return True
    module = importlib.import_module(module_path)
    return bool(getattr(module, "INCLUDE_DATASET_SPECIFIC_NOTES", True))


def render_system_prompt(system_prompt: str, metric_name: str, dataset: str | None) -> str:
    prompt = system_prompt
    include_few_shot = _load_few_shot_flag(metric_name)
    include_dataset_notes = _load_dataset_specific_notes_flag(metric_name)
    dataset_note = _render_dataset_note(metric_name, dataset) if include_dataset_notes else ""
    few_shot = _render_dataset_few_shot(metric_name, dataset) if include_few_shot else ""
    has_note_placeholder = "{dataset_specific_note}" in prompt
    has_few_shot_placeholder = "{dataset_few_shot_examples}" in prompt

    # Strip the dataset-specific-note section when the flag is off or no content.
    if has_note_placeholder and (not include_dataset_notes or not dataset_note):
        prompt = re.sub(
            r"\n*## Dataset-specific note \(current dataset only\)\n+\{dataset_specific_note\}",
            "",
            prompt,
        )
        has_note_placeholder = False

    # Strip the entire few-shot section when the flag is off, or when there
    # is no content for the current dataset (avoids a dangling section header).
    if has_few_shot_placeholder and (not include_few_shot or not few_shot):
        prompt = re.sub(
            r"\n*## Few-shot demonstrations\n+\{dataset_few_shot_examples\}",
            "",
            prompt,
        )
        has_few_shot_placeholder = False

    if has_note_placeholder:
        prompt = prompt.replace("{dataset_specific_note}", dataset_note)
    if has_few_shot_placeholder:
        prompt = prompt.replace("{dataset_few_shot_examples}", few_shot)

    # Fallback: append sections when prompts lack placeholders entirely.
    if not has_note_placeholder and include_dataset_notes and dataset_note:
        prompt = (
            f"{prompt}\n\n## Dataset-specific note (current dataset only)\n\n{dataset_note}"
        )
    if not has_few_shot_placeholder and include_few_shot and few_shot:
        prompt = f"{prompt}\n\n## Few-shot demonstrations\n\n{few_shot}"
    return prompt
