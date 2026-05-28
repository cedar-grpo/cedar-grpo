 # Multi-Evaluation Orchestrator

A Python script for running multiple evaluation scripts in parallel with organized logging and configurable parameters.

## Overview

This orchestrator runs multiple evaluation scripts (AIMO, AIME, COPA, ART, GoEmotion, GSM8K) against both raw and fine-tuned models, with support for parallel execution, real-time log streaming, and comprehensive result tracking.

## Features

- **Parallel Execution**: Run multiple evaluations simultaneously on different GPUs
- **Real-time Logging**: Stream output to console and log files simultaneously
- **Organized Results**: Timestamped directories with individual and master logs
- **Flexible Configuration**: Override settings via command line or configuration section
- **Path Injection**: Automatically inject model and output paths into evaluation scripts
- **Comprehensive Reporting**: Master log with execution summary and individual results

## Installation

No additional dependencies beyond standard Python libraries and your evaluation scripts.

```bash
chmod +x run_evaluations.py
```

## Configuration

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent


### Setting Up Paths

Edit the configuration section at the top of the script:

```python
# Model and training paths
RAW_MODEL_PATH = "/path/to/your/raw/model"
TRAINING_DIR = "/path/to/your/training/results"
BASE_OUTPUT_DIR = "/path/to/output/directory"
```

### Configuring CUDA Devices

Specify which GPUs to use for parallel execution:

```python
# GPUs will be used in order (cycling through if needed)
CUDA_DEVICES = ['2', '3']  # Use GPUs 2 and 3
```

### Adding/Removing Evaluation Scripts

Modify the `EVALUATION_SCRIPTS` list (you can comment others if you want to evaluate just a specific set):

```python
EVALUATION_SCRIPTS = [
    {
        'script': str(SCRIPT_DIR / 'evaluate_aimo_raw_vs_finetuned.py'),
        'name': 'AIMO Dataset Evaluation',
        'output_subdir': 'aimo_evaluation_results',
        'params': {
            'split': 'test',
        },
        'override_terminal': False  # If True, script params override CLI args
    },
    # Add more scripts here...
]
```

## Usage

### Basic Usage

Run all evaluations sequentially with default settings:

```bash
python run_evaluations.py
```

### Parallel Execution

Run 2 evaluations in parallel (will cycle through `CUDA_DEVICES`):

```bash
python run_evaluations.py --parallel 2
```

### Using Checkpoints

#### Option 1: Specific Checkpoint Path

```bash
python run_evaluations.py --checkpoint_path /path/to/checkpoint-640
```

This will use the exact checkpoint you specify for all evaluations.

#### Option 2: Checkpoint Directory

```bash
python run_evaluations.py --checkpoint_dir /path/to/checkpoints
```

This passes the directory to evaluation scripts, which may select checkpoints based on their own logic.

### Setting Training Direction

The training direction is determined by the `TRAINING_DIR` variable:

```bash
# Override via command line
python run_evaluations.py --training_dir /path/to/specific/training/run
```

Or edit in the configuration section:

```python
TRAINING_DIR = "/home/user/training/results/abductive_dt10.25.17:43_e20_..."
```

### Common Parameters

```bash
# Limit samples for faster testing
python run_evaluations.py --max_samples 100

# Change batch size
python run_evaluations.py --batch_size 4

# Use specific dataset split
python run_evaluations.py --split test

# Skip raw model evaluation
python run_evaluations.py --skip_raw

# Skip fine-tuned model evaluation
python run_evaluations.py --skip_finetuned
```

### Advanced Examples

```bash
# Run 3 evaluations in parallel with custom settings
python run_evaluations.py --parallel 3 --batch_size 4 --max_samples 200

# Use custom checkpoint and disable real-time console output
python run_evaluations.py --checkpoint_path /path/to/checkpoint-1280 --no_realtime

# Override all paths
python run_evaluations.py \
  --raw_model_path /custom/raw/model \
  --training_dir /custom/training/results \
  --base_output_dir /custom/output
```

## Command Line Arguments

### Orchestrator Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--parallel` | int | 2 | Number of scripts to run in parallel |
| `--output_dir` | str | `./multi_evaluation_results` | Output directory for results |
| `--no_realtime` | flag | False | Disable real-time log streaming to console |

### Path Override Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--raw_model_path` | str | Override raw model path |
| `--training_dir` | str | Override training directory |
| `--base_output_dir` | str | Override base output directory |

### Evaluation Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--max_samples` | int | Maximum samples to evaluate |
| `--cuda_device` | str | CUDA device (only for sequential) |
| `--batch_size` | int | Batch size for evaluation |
| `--split` | str | Dataset split (`train`, `test`, `validation`) |
| `--skip_raw` | flag | Skip raw model evaluation |
| `--skip_finetuned` | flag | Skip fine-tuned model evaluation |
| `--checkpoint_path` | str | Path to specific checkpoint |
| `--checkpoint_dir` | str | Directory containing checkpoints |

## Output Structure

multi_evaluation_results/
└── run_20251105_143022/
    ├── master_log.txt                          # Consolidated summary
    ├── 01_evaluate_aimo_raw_vs_finetuned.txt  # Individual logs
    ├── 02_evaluate_aime_raw_vs_finetuned.txt
    ├── 03_evaluate_copa_raw_vs_finetuned_guess_cause.txt
    └── ...


### Master Log Contents

- Execution summary (success/failure counts)
- Path configuration used
- Individual script results with:
  - Duration
  - Status
  - Error messages (if any)
  - Output directories
  - Full command executed

## How It Works

### Path Injection

The orchestrator injects paths into evaluation scripts via environment variables:

- `EVAL_RAW_MODEL_PATH`: Raw model path
- `EVAL_TRAINING_DIR`: Training directory
- `EVAL_OUTPUT_DIR`: Output directory for each evaluation

Your evaluation scripts should read these variables:

```python
raw_model_path = os.getenv('EVAL_RAW_MODEL_PATH', 'default/path')
training_dir = os.getenv('EVAL_TRAINING_DIR', 'default/path')
output_dir = os.getenv('EVAL_OUTPUT_DIR', 'default/path')
```

### Parameter Priority

For each script, parameters are merged in this order (later overrides earlier):

1. `DEFAULT_PARAMS` (in configuration)
2. Command line arguments (terminal args)
3. Script-specific `params` (in `EVALUATION_SCRIPTS`)

**Exception**: If `override_terminal: True`, script params take highest priority.

### Parallel Execution

- Scripts are distributed across `CUDA_DEVICES` in round-robin fashion
- Each script gets exclusive access to one GPU
- Real-time logs are thread-safe and properly attributed

## Troubleshooting

### Script Fails Immediately

Check that:
- Evaluation scripts exist and are executable
- Paths in configuration section are correct
- Required models and datasets are accessible

### GPU Out of Memory

- Reduce `--batch_size`
- Reduce `--parallel` count
- Use fewer/different GPUs in `CUDA_DEVICES`

### Missing Checkpoint

Ensure either:
- `--checkpoint_path` points to valid checkpoint file, or
- `--checkpoint_dir` contains checkpoint files, or
- Evaluation scripts can find checkpoints from `TRAINING_DIR`

### Logs Not Appearing in Console

- Remove `--no_realtime` flag
- Check that scripts are actually producing output

## Exit Codes

- `0`: All evaluations succeeded
- `1`: One or more evaluations failed

## Notes

- Each evaluation's output directory is created under `BASE_OUTPUT_DIR/<output_subdir>`
- Logs are buffered line-by-line for real-time streaming
- Master log provides complete audit trail of all executions
- Failed evaluations don't stop other evaluations from running