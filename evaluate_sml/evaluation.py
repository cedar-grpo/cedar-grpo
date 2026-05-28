import os
import re
import json
import yaml
import time
import shutil
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from loader import load_med_qa_dataset, load_med_mcqa_dataset, load_uniadilr_hgc_dataset
from api_handler import get_model_response
from path_utils import create_experiment_path, ensure_experiment_dir, get_results_files

def process_sample(sample, idx, model_name, api_key, max_tokens, temperature, thinking, prompt_content, sleep_time, dataset_name):
    """
    Processes a single data sample: formats prompt, calls API, parses result.
    This function runs sequentially.
    """
    time.sleep(sleep_time)

    if dataset_name == "medqa":
        input_text = sample["question"] + "\n" + str(sample["options"]) + "\n" + prompt_content
    elif dataset_name == "medmcqa":
        options = {
            "A": sample["opa"],
            "B": sample["opb"],
            "C": sample["opc"],
            "D": sample["opd"],
        }
        input_text = sample["question"] + "\n" + str(options) + "\n" + prompt_content
    elif dataset_name == "uniadilr":
        input_text = "Sentences: " + str(sample["context"]) + "\n" + "Hypothesis: " + sample["hypothesis"] + "\n" + prompt_content
    
    error_message = None
    model_output = None
    successful_api_call = False

    try:
        model_output, usage = get_model_response(model_name, api_key, input_text, max_tokens, temperature)
        successful_api_call = True
    except Exception as e:
        error_message = f"API call failed for sample {idx}: {e}"

    right_format = False

    if dataset_name == "medqa" or dataset_name == "medmcqa":
        extracted_letter = None
        if successful_api_call:
            if thinking:
                cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL)
            else:
                cleaned_text = model_output
            pattern = r"<a>([A-D])</a>"
            matches = re.findall(pattern, cleaned_text)
            if matches:
                extracted_letter = matches[-1]

            if extracted_letter:
                right_format = True
        else:
            usage = None

        cop_to_idx = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
        }
        correct_answer = cop_to_idx[sample["cop"]] if dataset_name == "medmcqa" else sample["answer_idx"]
    
        return {
            "idx": idx,
            "raw_data": sample,
            "successful_api_call": successful_api_call,
            "right_format": right_format,
            "input_text": input_text,
            "model_output": model_output,
            "model_answer": extracted_letter,
            "correct_answer": correct_answer,
            "error": error_message, 
            "token_usage": usage,
        }
    elif dataset_name == "uniadilr":
        sentence_ids = None
        if successful_api_call:
            if thinking:
                cleaned_text = re.sub(r"<think>.*?</think>", "", model_output, flags=re.DOTALL)
            else:
                cleaned_text = model_output
            matches = re.findall(r'<a>(.*?)</a>', cleaned_text)
            if matches:
                cleaned_text = matches[-1]
                matches = re.findall(r'sent(\d+)', cleaned_text)
                if matches:
                    sentence_ids = [int(num) for num in matches]

            if sentence_ids:
                right_format = True
        else:
            usage = None

        correct_answer = [int(num) for num in re.findall(r'sent(\d+)', sample["proof"])]

        return {
            "idx": idx,
            "raw_data": sample,
            "successful_api_call": successful_api_call,
            "right_format": right_format,
            "input_text": input_text,
            "model_output": model_output,
            "model_answer": sentence_ids,
            "correct_answer": correct_answer,
            "error": error_message, 
            "token_usage": usage,
        }

def evaluate_model(
    dataset_name: str, 
    model_name: str, 
    prompt_type: str,
    api_key: str, 
    use_cache: bool,
    parallel: bool = False,
    n_samples: int = -1,
):
    """
    Main function to run the model evaluation pipeline.
    Can run either sequentially or in parallel with up to 4 workers.
    
    Args:
        dataset_name: Name of the dataset to evaluate on
        model_name: Name of the model to evaluate
        prompt_type: Type of prompt to use
        api_key: API key for the model service
        use_cache: Whether to use cached results from previous runs
        parallel: Whether to run evaluation in parallel
        n_samples: Number of samples to evaluate on (-1 for all samples)
    """
    start_time = time.time()
    run_timestamp = datetime.now()

    # Find config.yaml relative to this file's location
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    thinking = config["models"][model_name]["thinking"]

    sleep_time = config['sleep_time']
    dataset_config = config["datasets"][dataset_name]
    prompt_content = None
    for prompt in dataset_config["prompts"]:
        if prompt["type"] == prompt_type:
            if "content" in prompt:
                # Legacy format - direct content
                prompt_content = prompt["content"]
            elif "template" in prompt:
                # New template-based format
                template_name = prompt["template"]
                if template_name in config["prompt_templates"]:
                    prompt_content = config["prompt_templates"][template_name]
                else:
                    raise ValueError(f"Template {template_name} not found in prompt_templates")
            else:
                raise ValueError(f"Prompt {prompt_type} has neither content nor template")
            break
    if prompt_content is None:
        raise ValueError(f"Prompt type {prompt_type} not found in dataset {dataset_name}")
    
    max_tokens = config["models"][model_name]["max_tokens_by_prompt_type"][prompt_type]
    temperature = config["models"][model_name]["temperature"]

    # Create new structured experiment path: results/{dataset}/{model}/{prompt}
    experiment_path = create_experiment_path(dataset_name, model_name, prompt_type)
    experiment_dir = ensure_experiment_dir(experiment_path)
    
    # Get standard file paths
    result_files = get_results_files(experiment_dir)
    results_file = result_files['results']
    run_details_file = result_files['run_details']
    
    print(f"Starting experiment: {dataset_name}/{model_name}/{prompt_type}")
    print(f"Results will be saved in: {experiment_dir}")

    initial_run_details = {
        "experiment_path": experiment_path,
        "status": "running",
        "run_time_start": run_timestamp.isoformat(),
        "config": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "prompt_type": prompt_type,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prompt_content": prompt_content,
        }
    }
    with open(run_details_file, "w") as f:
        json.dump(initial_run_details, f, indent=4)

    results = []
    processed_indices = set()
    
    if use_cache:
        # Check if experiment already exists (cache is implicit with new structure)
        if os.path.exists(results_file) and os.path.exists(run_details_file):
            with open(run_details_file, "r") as f:
                prev_run_details = json.load(f)
            
            is_same_config = (
                prev_run_details["config"].get("model_name") == model_name and
                prev_run_details["config"].get("prompt_type") == prompt_type and
                prev_run_details["config"].get("max_tokens") == max_tokens and
                prev_run_details["config"].get("temperature") == temperature and 
                prev_run_details["config"].get("prompt_content") == prompt_content
            )

            if is_same_config:
                print(f"Found existing experiment with same config. Resuming...")
                
                with open(results_file, "r") as f:
                    for line in f:
                        cached_result = json.loads(line)
                        if cached_result.get("successful_api_call") and cached_result.get("right_format"):
                            results.append(cached_result)
                            processed_indices.add(cached_result["idx"])
                
                print(f"Loaded {len(results)} valid results from cache. Failed/invalid samples will be re-processed.")
            else:
                print("Existing experiment has different config. Creating backup and starting fresh.")
                # Create backup of existing results
                backup_dir = experiment_dir + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copytree(experiment_dir, backup_dir)
                print(f"Backup created at: {backup_dir}")
        else:
            print("No existing experiment found. Starting from scratch.")
    else:
        print("`use_cache` is False. Starting from scratch.")
        # If not using cache but files exist, create backup
        if os.path.exists(experiment_dir) and os.listdir(experiment_dir):
            backup_dir = experiment_dir + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copytree(experiment_dir, backup_dir)
            print(f"Existing results backed up to: {backup_dir}")
            # Clean the directory for fresh start
            shutil.rmtree(experiment_dir)
            os.makedirs(experiment_dir, exist_ok=True)

    if dataset_name == "medqa":
        dataset = load_med_qa_dataset(n_samples=n_samples)
    elif dataset_name == "medmcqa":
        dataset = load_med_mcqa_dataset(n_samples=n_samples)
    elif dataset_name == "uniadilr":
        dataset = load_uniadilr_hgc_dataset(n_samples=n_samples)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    unprocessed_samples = [(sample, idx) for idx, sample in enumerate(dataset) if idx not in processed_indices]
    print(f"Total samples: {len(dataset)}. Processed from cache: {len(processed_indices)}. Remaining: {len(unprocessed_samples)}")

    if not unprocessed_samples:
        print("All samples were processed in the cached run.")
    
    if parallel:
        # Load max_workers from config
        max_workers = config["api"].get("max_workers", 3)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_sample, sample, idx, model_name, api_key, max_tokens, temperature, thinking, prompt_content, sleep_time, dataset_name): (sample, idx) for sample, idx in unprocessed_samples}
            for future in tqdm(as_completed(futures), desc="Processing Samples", unit="sample", total=len(unprocessed_samples)):
                try:
                    result = future.result()
                    if result.get("error"):
                        tqdm.write(result["error"])
                    results.append(result)

                    result_to_write = result.copy()
                    result_to_write.pop('error', None)
                    
                    with open(results_file, "a") as f:
                        f.write(json.dumps(result_to_write) + "\n")
                except Exception as exc:
                    sample, idx = futures[future]
                    tqdm.write(f'Sample {idx} generated an unhandled exception: {exc}')


    else :
        progress_bar = tqdm(unprocessed_samples, desc="Processing Samples", unit="sample")
        for sample, idx in progress_bar:
            try:
                result = process_sample(
                    sample, idx, model_name, api_key, max_tokens, temperature, thinking, prompt_content, sleep_time, dataset_name
                )
                
                if result.get("error"):
                    tqdm.write(result["error"])

                results.append(result)

                result_to_write = result.copy()
                result_to_write.pop('error', None)
                
                with open(results_file, "a") as f:
                    f.write(json.dumps(result_to_write) + "\n")

            except Exception as exc:
                tqdm.write(f'Sample {idx} generated an unhandled exception: {exc}')

    execution_time = time.time() - start_time
    
    results.sort(key=lambda r: r['idx'])
    
    run_summary = {
        **initial_run_details,
        "status": "completed",
        "run_time_end": datetime.now().isoformat(),
        "execution_time_seconds": round(execution_time, 2),
        "summary": {
            "total_samples": len(dataset),
            "processed_samples": len(results),
            "successful_api_calls": sum(1 for r in results if r["successful_api_call"]),
            "correctly_formatted_answers": sum(1 for r in results if r["right_format"]),
        }
    }

    with open(run_details_file, "w") as f:
        json.dump(run_summary, f, indent=4)

    print("\n--- Evaluation Complete ---")
    print(json.dumps(run_summary, indent=4))
    print(f"Results saved to {results_file}")
    print(f"Run summary saved to {run_details_file}")

if __name__ == "__main__":
    # Load config for test
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    evaluate_model(
        dataset_name="uniadilr",
        model_name="Qwen3-32B",
        prompt_type="Chain of Thought",
        api_key=config["api"]["api_key"],
        use_cache=False,
        parallel=True,
        n_samples=10
    )