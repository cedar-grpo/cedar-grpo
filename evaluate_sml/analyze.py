import yaml
import json
import os
from sklearn.metrics import classification_report
import numpy as np
from path_utils import create_experiment_path, find_experiments, get_results_files, parse_experiment_path

def analyze_results_by_path(experiment_path: str):
    """
    Analyzes model evaluation results by experiment path. For multi-label tasks like 'uniadilr',
    it calculates exact match accuracy and subset relationships. For standard
    classification tasks, it uses classification_report.

    Args:
        experiment_path (str): The path to the experiment directory.
    """
    # Parse experiment path to get dataset info
    try:
        dataset_name, model_name, prompt_name = parse_experiment_path(experiment_path)
    except ValueError as e:
        print(f"Error parsing experiment path: {e}")
        return
    
    # Get result file paths
    result_files = get_results_files(experiment_path)
    results_file = result_files['results']
    run_details_path = result_files['run_details']
    run_data = {}
    if os.path.exists(run_details_path):
        with open(run_details_path, 'r') as file:
            run_data = json.load(file)
    else:
        print(f"Warning: run_details.json not found at {run_details_path}")


    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return

    total_samples = 0
    unsuccessful_api_calls = 0
    wrong_format = 0
    y_true = []
    y_pred = []
    prompt_tokens_list = []
    completion_tokens_list = []

    with open(results_file, "r") as f:
        for line in f:
            total_samples += 1
            data = json.loads(line)

            usage = data.get("token_usage")
            if usage:
                prompt_tokens_list.append(usage.get('prompt_tokens', 0))
                completion_tokens_list.append(usage.get('completion_tokens', 0))

            if not data.get("successful_api_call", False):
                unsuccessful_api_calls += 1
            
            if not data.get("right_format", False):
                wrong_format += 1
            
            if data.get("successful_api_call", False) and data.get("right_format", False):
                true_answer = data["correct_answer"]
                model_answer = data["model_answer"]

                if dataset_name == "uniadilr":
                    y_true.append(set(true_answer))
                    y_pred.append(set(model_answer))
                elif dataset_name == "medqa" or dataset_name == "medmcqa":
                    y_true.append(true_answer)
                    y_pred.append(model_answer)

    percent_unsuccessful_api = (unsuccessful_api_calls / total_samples) * 100 if total_samples > 0 else 0
    percent_wrong_format = (wrong_format / total_samples) * 100 if total_samples > 0 else 0

    token_usage_report = "Token usage data not available in results."
    if prompt_tokens_list and completion_tokens_list:
        prompt_mean = np.mean(prompt_tokens_list)
        prompt_q1, prompt_median, prompt_q3 = np.percentile(prompt_tokens_list, [25, 50, 75])
        
        completion_mean = np.mean(completion_tokens_list)
        completion_q1, completion_median, completion_q3 = np.percentile(completion_tokens_list, [25, 50, 75])
        
        token_usage_report = (
            f"    Prompt Tokens:\n"
            f"       - Mean: {prompt_mean:.2f}\n"
            f"       - 25th Percentile (Q1): {prompt_q1}\n"
            f"       - 50th Percentile (Median): {prompt_median}\n"
            f"       - 75th Percentile (Q3): {prompt_q3}\n\n"
            f"    Completion Tokens:\n"
            f"       - Mean: {completion_mean:.2f}\n"
            f"       - 25th Percentile (Q1): {completion_q1}\n"
            f"       - 50th Percentile (Median): {completion_median}\n"
            f"       - 75th Percentile (Q3): {completion_q3}"
        )

    metrics_title = "Evaluation Metrics"
    metrics_report = "Not enough valid samples to generate metrics."
    num_valid_samples = len(y_true)

    if num_valid_samples > 0:
        if dataset_name == "uniadilr":
            precision_scores = []
            recall_scores = []
            f1_scores = []
            em_scores = []

            for true_set, pred_set in zip(y_true, y_pred):
                # Calculate intersection
                intersection = pred_set.intersection(true_set)
                intersection_size = len(intersection)
                pred_size = len(pred_set)
                true_size = len(true_set)
                
                # Precision: |P ∩ G| / |P|
                if pred_size == 0:
                    # Edge case: if model predicts nothing
                    precision = 1.0 if true_size == 0 else 0.0
                else:
                    precision = intersection_size / pred_size
                
                # Recall: |P ∩ G| / |G|
                if true_size == 0:
                    # Edge case: if golden set is empty
                    recall = 1.0 if pred_size == 0 else 0.0
                else:
                    recall = intersection_size / true_size
                
                # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                
                # Exact Match: 1 if P = G, else 0
                em = 1.0 if pred_set == true_set else 0.0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                em_scores.append(em)
            
            # Calculate averages across all samples
            avg_precision = np.mean(precision_scores)
            avg_recall = np.mean(recall_scores)
            avg_f1 = np.mean(f1_scores)
            avg_em = np.mean(em_scores)
            
            metrics_report = (
                f"1. Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)\n"
                f"   - Of all the sentences the model predicted as relevant, what fraction were actually relevant.\n\n"
                f"2. Recall: {avg_recall:.4f} ({avg_recall*100:.2f}%)\n"
                f"   - Of all the sentences that were actually relevant, what fraction did the model find.\n\n"
                f"3. F1-Score: {avg_f1:.4f} ({avg_f1*100:.2f}%)\n"
                f"   - Harmonic mean of Precision and Recall, balancing both concerns.\n\n"
                f"4. Exact Match (EM): {avg_em:.4f} ({avg_em*100:.2f}%)\n"
                f"   - Percentage of questions where the model's predicted set matched the golden set perfectly."
            )

        elif dataset_name == "medqa" or dataset_name == "medmcqa":
            metrics_title = "Classification Metrics"
            metrics_report = classification_report(y_true, y_pred, zero_division=0)

    analysis_content = f"""
    Analysis Report for {experiment_path}
    Dataset: {dataset_name}
    Model: {model_name}
    Prompt: {prompt_name}
    
    Config:

    {json.dumps(run_data.get('config', 'Not available'), indent=4)}
    ======================================================

    1. Total Samples: {total_samples}

    2. Unsuccessful API Calls:
        - Count: {unsuccessful_api_calls}
        - Percentage: {percent_unsuccessful_api:.2f}%

    3. Wrong Formatting:
        - Count: {wrong_format}
        - Percentage: {percent_wrong_format:.2f}%

    4. Token Usage Statistics:
    ----------------------------------------------------------------------
    {token_usage_report}

    5. {metrics_title} (for successful and correctly formatted samples):
    ----------------------------------------------------------------------
    {metrics_report}
    """

    report_path = result_files['analysis_report']
    with open(report_path, "w") as f:
        f.write(analysis_content)
    
    print(f"Analysis complete for {experiment_path}")
    print(f"Report saved to {report_path}")

def analyze_results(dataset_name: str, model_name: str, prompt_type: str):
    """
    Analyzes model evaluation results using the new directory structure.
    
    Args:
        dataset_name (str): The name of the dataset.
        model_name (str): The name of the model.
        prompt_type (str): The type of prompt used.
    """
    experiment_path = create_experiment_path(dataset_name, model_name, prompt_type)
    
    if not os.path.exists(experiment_path):
        print(f"Experiment not found: {experiment_path}")
        return
    
    analyze_results_by_path(experiment_path)


def analyze_all_experiments(base_dir: str = "results"):
    """
    Analyzes all experiments found in the results directory.
    
    Args:
        base_dir (str): Base directory to search for experiments.
    """
    experiments = find_experiments(base_dir)
    
    if not experiments:
        print(f"No experiments found in {base_dir}")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp_path in experiments:
        try:
            dataset, model, prompt = parse_experiment_path(exp_path)
            print(f"  - {dataset}/{model}/{prompt}")
        except ValueError:
            print(f"  - {exp_path} (invalid path format)")
    
    print("\nAnalyzing experiments...")
    for exp_path in experiments:
        try:
            print(f"\nAnalyzing: {exp_path}")
            analyze_results_by_path(exp_path)
        except Exception as e:
            print(f"Error analyzing {exp_path}: {e}")


if __name__ == "__main__":
    # Example usage
    analyze_results("uniadilr", "Qwen3-32B", "Chain of Thought")