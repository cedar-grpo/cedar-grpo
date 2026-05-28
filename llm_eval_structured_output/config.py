"""
config.py
---------
All runtime settings for the LLM evaluation pipeline.
Edit this file to change models, sampling, API credentials, and which metrics to run.
"""

import datetime
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
RUN_ID: str = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
N_SAMPLES: int = 10          # How many items to evaluate per dataset per checkpoint
MAX_WORKERS: int = 1         # Parallel threads for LLM calls (keep low for testing)
RANDOM_SEED: int = 42        # Fixed seed for reproducible sampling

# Fraction of N_SAMPLES drawn from correct items is NO LONGER SUPPORTED.
# ground-truth was removed from this pipeline.
# SAMPLE_CORRECT_RATIO was removed.

# Directory that contains pre-generated sample index files.
# When N_SAMPLES matches one of these files (e.g. random_samples/samples_10.json),
# those fixed indices are used instead of random sampling, ensuring exact
# reproducibility across runs without relying on RANDOM_SEED.
# Supported sizes: 3, 5, 10, 50, 100, 200  (add more files to extend this set).
# Set to None or "" to always use random sampling.
RANDOM_SAMPLES_DIR: str = "random_samples"

# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------
# JUDGE_MODEL: str = "gpt-4o-mini"
JUDGE_MODEL: str = "gemini-2.0-flash"

# Reasoning effort for GPT-5+ models ("low" | "medium" | "high").
# Ignored for older models that do not support this parameter.
REASONING_EFFORT: str = "low"

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
# Base URL for OpenAI-compatible chat/completions endpoints.
OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "")
# Base URL for Gemini SDK endpoint (used when JUDGE_MODEL starts with "gemini").
GEMINI_BASE_URL: str = os.environ.get("GEMINI_BASE_URL", "")

API_TIMEOUT: float = 60.0   # seconds
API_MAX_RETRIES: int = 2
# Default maximum completion length per LLM call.
# If responses fail with "length limit was reached", increase this value.
MAX_COMPLETION_TOKENS: int = 4096
# Per-metric overrides for MAX_COMPLETION_TOKENS.
# Metrics not listed here fall back to MAX_COMPLETION_TOKENS.
METRIC_MAX_COMPLETION_TOKENS: dict[str, int] = {
    "observation_coverage": 8192,
    "rationale_graph": 8192,
}

# If True, continue the run even when the API connectivity check fails.
# Useful for non-interactive or offline runs.
CONTINUE_ON_API_FAILURE: bool = os.environ.get("CONTINUE_ON_API_FAILURE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_OUTPUT_DIR: str = "results"
LOG_DIR: str = os.path.join(BASE_OUTPUT_DIR, "llm_logs")
UNNORM_DIR: str = os.path.join(BASE_OUTPUT_DIR, "unnormalized")
NORM_DIR: str = os.path.join(BASE_OUTPUT_DIR, "normalized")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# If True, deletes existing JSONL log files at the start of the run so that
# all LLM calls are re-executed (useful when prompts changes).
CLEAR_PREVIOUS_OUTPUTS: bool = True

# ---------------------------------------------------------------------------
# Metrics to run
# Controlled by the metric registry (metrics/registry.py).
# List only the metric names you want to evaluate.
# Available metrics: "backtracking", "branchiness", "uncertainty_markers",
#                    "prior", "differential_elimination",
#                    "observation_coverage",
#                    "evidence_explanation_directionality", "rationale_graph",
#                    "evidence_explanation_directionality_scorebased"
# An empty list activates ALL registered metrics.
# ---------------------------------------------------------------------------
ACTIVE_METRICS: list[str] = ["backtracking", "branchiness", "uncertainty_markers",
                            "prior", "differential_elimination", "observation_coverage",
                            "evidence_explanation_directionality_scorebased"]  # Empty list activates all metrics

# ---------------------------------------------------------------------------
# Datasets to evaluate
# List only the dataset names (folder names inside each checkpoint) to include.
# An empty list evaluates ALL datasets found in each checkpoint.
# ---------------------------------------------------------------------------
ACTIVE_DATASETS: list[str] = []  # Test dataset

# ---------------------------------------------------------------------------
# Excluded checkpoints
# List checkpoint directory names (basenames) to skip entirely.
# Supports both trained checkpoints ("checkpoint-500") and "raw_model".
# An empty list means no checkpoints are excluded.
# ---------------------------------------------------------------------------
EXCLUDED_CHECKPOINTS: list[str] = []  # Evaluate all checkpoints
# Example: EXCLUDED_CHECKPOINTS = ["checkpoint-500", "raw_model"]
