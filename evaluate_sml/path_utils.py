"""
Utilities for handling experiment result paths and naming conventions.
"""
import os
import re
from typing import Tuple


def create_prompt_identifier(prompt_type: str) -> str:
    """
    Convert a prompt type into a clean, filesystem-safe identifier.
    
    Args:
        prompt_type: The original prompt type string
        
    Returns:
        A clean identifier suitable for directory names
        
    Examples:
        "Only Final Answer" -> "only_final_answer"
        "Chain of Thought" -> "chain_of_thought"
        "Chain of Thought and Think Abductively" -> "chain_of_thought_and_think_abductively"
    """
    # Convert to lowercase and replace spaces/special chars with underscores
    identifier = re.sub(r'[^a-zA-Z0-9]+', '_', prompt_type.lower())
    # Remove leading/trailing underscores and collapse multiple underscores
    identifier = re.sub(r'_+', '_', identifier).strip('_')
    return identifier


def create_model_identifier(model_name: str) -> str:
    """
    Convert a model name into a clean, filesystem-safe identifier.
    
    Args:
        model_name: The original model name
        
    Returns:
        A clean identifier suitable for directory names
        
    Examples:
        "DeepSeek-V3.1" -> "deepseek-v3.1"
        "Qwen3-32B" -> "qwen3-32b"
    """
    # Keep alphanumeric, dots, and hyphens, convert to lowercase
    identifier = re.sub(r'[^a-zA-Z0-9.-]', '-', model_name.lower())
    # Collapse multiple hyphens
    identifier = re.sub(r'-+', '-', identifier).strip('-')
    return identifier


def create_experiment_path(
    dataset_name: str, 
    model_name: str, 
    prompt_type: str,
    base_dir: str = "results"
) -> str:
    """
    Create a structured experiment path following the new organization.
    
    Args:
        dataset_name: Name of the dataset (e.g., "medqa", "uniadilr")
        model_name: Name of the model (e.g., "gpt-4o", "DeepSeek-V3.1")
        prompt_type: Type of prompt used
        base_dir: Base directory for results (default: "results")
        
    Returns:
        Full path string for the experiment
        
    Example:
        create_experiment_path("medqa", "gpt-4o", "Chain of Thought")
        -> "results/medqa/gpt-4o/chain_of_thought"
    """
    model_id = create_model_identifier(model_name)
    prompt_id = create_prompt_identifier(prompt_type)
    
    return os.path.join(base_dir, dataset_name, model_id, prompt_id)


def parse_experiment_path(path: str) -> Tuple[str, str, str]:
    """
    Parse an experiment path to extract dataset, model, and prompt components.
    
    Args:
        path: The experiment path to parse
        
    Returns:
        Tuple of (dataset_name, model_identifier, prompt_identifier)
        
    Example:
        parse_experiment_path("results/medqa/gpt-4o/chain_of_thought")
        -> ("medqa", "gpt-4o", "chain_of_thought")
    """
    parts = path.strip('/').split('/')
    if len(parts) < 3:
        raise ValueError(f"Invalid experiment path: {path}")
    
    # Take the last 3 parts as dataset/model/prompt
    return parts[-3], parts[-2], parts[-1]


def find_experiments(base_dir: str = "results") -> list:
    """
    Find all existing experiments in the new directory structure.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of experiment paths
    """
    experiments = []
    
    if not os.path.exists(base_dir):
        return experiments
    
    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        for model in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model)
            if not os.path.isdir(model_path):
                continue
                
            for prompt in os.listdir(model_path):
                prompt_path = os.path.join(model_path, prompt)
                if os.path.isdir(prompt_path):
                    experiments.append(os.path.join(base_dir, dataset, model, prompt))
    
    return experiments


def ensure_experiment_dir(experiment_path: str) -> str:
    """
    Ensure the experiment directory exists and return the full path.
    
    Args:
        experiment_path: The experiment path to create
        
    Returns:
        The absolute path to the experiment directory
    """
    os.makedirs(experiment_path, exist_ok=True)
    return os.path.abspath(experiment_path)


def get_results_files(experiment_path: str) -> dict:
    """
    Get the standard result file paths for an experiment.
    
    Args:
        experiment_path: Path to the experiment directory
        
    Returns:
        Dictionary with paths to results.jsonl, run_details.json, and analysis_report.txt
    """
    return {
        'results': os.path.join(experiment_path, 'results.jsonl'),
        'run_details': os.path.join(experiment_path, 'run_details.json'),
        'analysis_report': os.path.join(experiment_path, 'analysis_report.txt')
    }


if __name__ == "__main__":
    # Test the functions
    test_cases = [
        ("medqa", "gpt-4o", "Only Final Answer"),
        ("uniadilr", "DeepSeek-V3.1", "Chain of Thought"),
        ("medmcqa", "Qwen3-32B", "Chain of Thought and Think Abductively"),
    ]
    
    print("Testing path creation:")
    for dataset, model, prompt in test_cases:
        path = create_experiment_path(dataset, model, prompt)
        print(f"  {dataset}/{model}/{prompt} -> {path}")
    
    print("\nTesting path parsing:")
    for dataset, model, prompt in test_cases:
        path = create_experiment_path(dataset, model, prompt)
        parsed = parse_experiment_path(path)
        print(f"  {path} -> {parsed}")
