from datasets import load_dataset
from typing import Dict, Any, List
import json
import yaml
import os

def load_med_qa_dataset(n_samples: int = -1) -> List[Dict[str, Any]]:
    """
    Loads MedQA US 4 options test split from the path specified in the config.
    
    Returns a list of samples (rows).
    """
    # Find config.yaml relative to this file's location
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    file_path = config["datasets"]["medqa"]["file_path"]
    # Make path relative to project root (parent of evaluate_sml)
    project_root = os.path.dirname(os.path.dirname(__file__))
    absolute_file_path = os.path.join(project_root, file_path)
    dataset_samples = []
    with open(absolute_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset_samples.append(json.loads(line))
    if n_samples == -1:
        return dataset_samples
    else:
        return dataset_samples[:n_samples]

def load_med_mcqa_dataset(n_samples: int = -1) -> List[Dict[str, Any]]:
    """
    Loads MedMCQA validation split from the Hugging Face path specified in the config.
    
    Returns a list of samples (rows).
    """
    # Find config.yaml relative to this file's location
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hf_path = config["datasets"]["medmcqa"]["hf_path"]

    ds = load_dataset(hf_path, split="validation")

    if n_samples == -1:
        return list(ds)
    else:
        return list(ds)[:n_samples]


def load_uniadilr_hgc_dataset(n_samples: int = -1) -> List[Dict[str, Any]]:
    """
    Loads UniADILR-HGc dataset from the path specified in the config.
    
    Returns a list of samples (rows).
    """
    # Find config.yaml relative to this file's location
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    file_path = config["datasets"]["uniadilr"]["file_path"]
    # Make path relative to project root (parent of evaluate_sml)
    project_root = os.path.dirname(os.path.dirname(__file__))
    absolute_file_path = os.path.join(project_root, file_path)
    dataset_samples = []
    with open(absolute_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset_samples.append(json.loads(line))
    if n_samples == -1:
        return dataset_samples
    else:
        return dataset_samples[:n_samples]

if __name__ == "__main__":
    print(load_uniadilr_hgc_dataset(n_samples=1))