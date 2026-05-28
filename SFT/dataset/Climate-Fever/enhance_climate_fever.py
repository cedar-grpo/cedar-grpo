#!/usr/bin/env python3
"""
Generate rationale-enhanced CLIMATE-FEVER JSON files using OpenRouter.

Usage:
  for linux/mac:
  export OPENROUTER_API_KEY="your_openrouter_key"

  for windows:
  set OPENROUTER_API_KEY=your_key_here

  python enhance_climate_fever.py \
    --train ../../dataset/Climate-Fever/climate_fever_train.json \
    --val ../../dataset/Climate-Fever/climate_fever_val.json \
    --output-dir . \
    --threads 8 \
    --resume

Outputs:
  ./climate_fever_train_enhanced.json
  ./climate_fever_val_enhanced.json

Each output sample keeps the original fields and adds:
  "rationale": "...inner text from <think>...</think>..."

The script only accepts model outputs that exactly match:

<think>
...
</think>

<answer>
SUPPORTS
</answer>

or

<think>
...
</think>

<answer>
REFUTES
</answer>

or

<think>
...
</think>

<answer>
NOT ENOUGH INFO
</answer>
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Config defaults
# =========================

MODEL = "google/gemini-3-flash-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Can be overridden by CLI flags --resume / --no-resume.
RESUME_FROM_EXISTING_OUTPUTS = True

DEFAULT_THREADS = 8
DEFAULT_MAX_RETRIES = 6
DEFAULT_REQUEST_TIMEOUT_SECONDS = 90

TEMPERATURE = 0.2
TOP_P = 0.9
MAX_COMPLETION_TOKENS = 350

# Optional OpenRouter app-identifying headers.
# Leave as None unless you want OpenRouter analytics attribution.
HTTP_REFERER = None
X_TITLE = "climate-fever-rationale-generation"


SYSTEM_PROMPT_SYNTHETIC_CLIMATE_FEVER = """\
You are generating supervised fine-tuning data for an evidence-grounded fact-checking model.

You will receive:
1. A claim
2. A list of evidences
3. The gold answer

Write a concise rationale explaining why the evidence supports, refutes, is insufficient to evaluate, or disputes the claim.

The rationale should:
- Start from the claim.
- Identify the most relevant evidence.
- Explain how the evidence relates to the claim.
- For SUPPORTS, explain why the evidence makes the claim more likely true.
- For REFUTES, explain why the evidence contradicts the claim.
- For NOT ENOUGH INFO, explain what key information is missing.
- For DISPUTED, explain that the provided evidence is mixed, conflicting, or does not lead to a single clear verdict.
- Use only the provided evidence, not external knowledge.
- Avoid unrelated evidence and unsupported speculation.

Output exactly:

<think>
[A concise evidence-grounded rationale, usually 3–5 sentences.]
</think>

<answer>
[The gold answer only: SUPPORTS, REFUTES, NOT ENOUGH INFO, or DISPUTED.]
</answer>

Do not change the gold answer in the answer section.
Do not include any text, punctuation, or explanation in the answer section.
""".strip()


STRICT_RESPONSE_RE = re.compile(
    r"\A<think>\n(?P<rationale>.+?)\n</think>\n\n<answer>\n"
    r"(?P<answer>SUPPORTS|REFUTES|NOT ENOUGH INFO|DISPUTED)\n"
    r"</answer>\Z",
    re.DOTALL,
)


# =========================
# Prompt + validation
# =========================

def climate_fever_label_to_answer(label_raw: Any) -> str:
    """
    Standard CLIMATE-FEVER mapping:
      0 = SUPPORTS
      1 = REFUTES
      2 = NOT ENOUGH INFO
      3 = DISPUTED

    Also accepts string versions of these labels.
    """
    label_map = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO",
        3: "DISPUTED",
    }

    if isinstance(label_raw, bool):
        raise ValueError(f"Invalid boolean claim_label: {label_raw!r}")

    if isinstance(label_raw, int):
        if label_raw not in label_map:
            raise ValueError(f"Invalid numeric claim_label: {label_raw!r}")
        return label_map[label_raw]

    if isinstance(label_raw, str):
        stripped = label_raw.strip()

        if stripped in {"0", "1", "2", "3"}:
            return label_map[int(stripped)]

        normalized = stripped.upper().replace("_", " ")
        if normalized in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "DISPUTED"}:
            return normalized

    raise ValueError(f"Invalid claim_label: {label_raw!r}")


def create_synthetic_climate_fever_prompt(example: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns:
      system_prompt, user_prompt, gold_answer
    """
    system_prompt = SYSTEM_PROMPT_SYNTHETIC_CLIMATE_FEVER

    claim = example["claim"]

    evidence_objs = example.get("evidences", [])
    evidence_list = [e.get("evidence", "") for e in evidence_objs if isinstance(e, dict)]
    evidence_text = "\n".join(
        [f"Evidence {i + 1}: {txt}" for i, txt in enumerate(evidence_list) if txt]
    )

    gold_answer = climate_fever_label_to_answer(example["claim_label"])

    user_prompt = f"""\
Claim:
{claim}

Evidence:
{evidence_text}

Gold Answer:
{gold_answer}

Generate the SFT target.
""".strip()

    return system_prompt, user_prompt, gold_answer


def validate_and_extract_rationale(response_text: str, expected_answer: str) -> str:
    """
    Strictly validates the model response.

    No fuzzy parsing.
    No extraction from malformed text.
    The whole stripped response must exactly match the requested format.
    """
    normalized = response_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    match = STRICT_RESPONSE_RE.fullmatch(normalized)
    if not match:
        raise ValueError(f"Bad format. Raw response was:\n{response_text}")

    answer = match.group("answer")
    if answer != expected_answer:
        raise ValueError(
            f"Answer mismatch. Expected {expected_answer}, got {answer}. "
            f"Raw response was:\n{response_text}"
        )

    rationale = match.group("rationale").strip()
    if not rationale:
        raise ValueError("Empty rationale.")

    return rationale


# =========================
# JSON I/O
# =========================

def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {i} is not a JSON object.")

        required = {"claim", "evidences", "claim_label"}
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"{path} item {i} is missing fields: {sorted(missing)}")

        if not isinstance(item["claim"], str) or not item["claim"].strip():
            raise ValueError(f"{path} item {i} has invalid claim: {item['claim']!r}")

        if not isinstance(item["evidences"], list):
            raise ValueError(f"{path} item {i} field 'evidences' must be a list.")

        for j, evidence_obj in enumerate(item["evidences"]):
            if not isinstance(evidence_obj, dict):
                raise ValueError(f"{path} item {i} evidence {j} is not a JSON object.")

            if "evidence" not in evidence_obj:
                raise ValueError(f"{path} item {i} evidence {j} is missing 'evidence'.")

            if evidence_obj["evidence"] is not None and not isinstance(evidence_obj["evidence"], str):
                raise ValueError(
                    f"{path} item {i} evidence {j} has non-string evidence: "
                    f"{evidence_obj['evidence']!r}"
                )

        # Validates both numeric and string labels.
        climate_fever_label_to_answer(item["claim_label"])

    return data


def atomic_write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """
    Writes a complete valid JSON file atomically.

    This avoids leaving a half-written JSON file if the process is interrupted
    during the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_enhanced.json"


def evidence_signature(example: Dict[str, Any]) -> List[Tuple[Any, Any, Any]]:
    """
    Creates a stable lightweight signature for resume validation.

    CLIMATE-FEVER examples do not always have a top-level id, so resume
    validation compares claim, label, and evidence identity/text instead.
    """
    sig = []
    for evidence_obj in example.get("evidences", []):
        if not isinstance(evidence_obj, dict):
            continue
        sig.append(
            (
                evidence_obj.get("evidence_id"),
                evidence_obj.get("article"),
                evidence_obj.get("evidence"),
            )
        )
    return sig


def sample_signature(example: Dict[str, Any]) -> Tuple[Any, Any, Tuple[Tuple[Any, Any, Any], ...]]:
    return (
        example.get("claim"),
        example.get("claim_label"),
        tuple(evidence_signature(example)),
    )


def initialize_output_data(
    input_data: List[Dict[str, Any]],
    output_path: Path,
    resume: bool,
) -> List[Dict[str, Any]]:
    """
    If resume is enabled and an output file already exists, load and verify it.
    Otherwise, initialize from the input data.
    """
    if resume and output_path.exists():
        existing = load_json_list(output_path)

        if len(existing) != len(input_data):
            raise ValueError(
                f"Cannot resume from {output_path}: length mismatch. "
                f"Input has {len(input_data)} rows, output has {len(existing)} rows."
            )

        for i, (inp, out) in enumerate(zip(input_data, existing)):
            if sample_signature(inp) != sample_signature(out):
                raise ValueError(
                    f"Cannot resume from {output_path}: sample mismatch at row {i}. "
                    "The existing enhanced file does not match the input file."
                )

        return existing

    initialized = copy.deepcopy(input_data)
    atomic_write_json(output_path, initialized)
    return initialized


def has_completed_rationale(example: Dict[str, Any]) -> bool:
    return isinstance(example.get("rationale"), str) and bool(example["rationale"].strip())


def append_error_log(output_dir: Path, record: Dict[str, Any]) -> None:
    log_path = output_dir / "generation_errors.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# OpenRouter API
# =========================

def post_json_with_urllib(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_seconds: int,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {error_body}") from e

    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e}") from e


def call_openrouter_once(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if HTTP_REFERER:
        headers["HTTP-Referer"] = HTTP_REFERER
    if X_TITLE:
        headers["X-Title"] = X_TITLE

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
    }

    response_json = post_json_with_urllib(
        url=OPENROUTER_URL,
        payload=payload,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )

    try:
        content = response_json["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {response_json}") from e

    if not isinstance(content, str):
        raise RuntimeError(f"Expected string content, got: {type(content).__name__}")

    return content


def generate_rationale_with_retries(
    api_key: str,
    example: Dict[str, Any],
    max_retries: int,
    timeout_seconds: int,
) -> Tuple[str, str]:
    """
    Returns:
      rationale, raw_valid_response
    """
    system_prompt, user_prompt, expected_answer = create_synthetic_climate_fever_prompt(example)

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_openrouter_once(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=timeout_seconds,
            )
            rationale = validate_and_extract_rationale(raw_response, expected_answer)
            return rationale, raw_response

        except Exception as e:
            last_error = e

            if attempt == max_retries:
                break

            # Exponential backoff with jitter.
            sleep_seconds = min(60.0, 1.5 * (2 ** (attempt - 1)))
            sleep_seconds += random.uniform(0.0, 0.75)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


# =========================
# Worker orchestration
# =========================

@dataclass(frozen=True)
class Task:
    split_name: str
    index: int
    example: Dict[str, Any]


@dataclass
class SplitState:
    input_path: Path
    output_path: Path
    data: List[Dict[str, Any]]
    lock: threading.Lock


def process_task(
    task: Task,
    api_key: str,
    max_retries: int,
    timeout_seconds: int,
) -> Tuple[Task, str, str]:
    rationale, raw_response = generate_rationale_with_retries(
        api_key=api_key,
        example=task.example,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    return task, rationale, raw_response


def build_pending_tasks(split_name: str, data: List[Dict[str, Any]]) -> List[Task]:
    tasks: List[Task] = []
    for i, example in enumerate(data):
        if not has_completed_rationale(example):
            tasks.append(Task(split_name=split_name, index=i, example=copy.deepcopy(example)))
    return tasks


def run_generation(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    threads: int,
    resume: bool,
    max_retries: int,
    timeout_seconds: int,
) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_input = load_json_list(train_path)
    val_input = load_json_list(val_path)

    train_output_path = output_path_for(train_path, output_dir)
    val_output_path = output_path_for(val_path, output_dir)

    train_data = initialize_output_data(train_input, train_output_path, resume=resume)
    val_data = initialize_output_data(val_input, val_output_path, resume=resume)

    states: Dict[str, SplitState] = {
        "train": SplitState(
            input_path=train_path,
            output_path=train_output_path,
            data=train_data,
            lock=threading.Lock(),
        ),
        "val": SplitState(
            input_path=val_path,
            output_path=val_output_path,
            data=val_data,
            lock=threading.Lock(),
        ),
    }

    tasks = build_pending_tasks("train", train_data) + build_pending_tasks("val", val_data)

    total_rows = len(train_data) + len(val_data)
    already_done = total_rows - len(tasks)

    print(f"Train rows: {len(train_data)}")
    print(f"Val rows:   {len(val_data)}")
    print(f"Already completed: {already_done}/{total_rows}")
    print(f"Pending: {len(tasks)}")
    print(f"Output train: {train_output_path}")
    print(f"Output val:   {val_output_path}")
    print(f"Threads: {threads}")
    print("", flush=True)

    if not tasks:
        print("Nothing to do. All rows already have rationales.")
        return

    completed_now = 0
    failed: List[Dict[str, Any]] = []
    progress_lock = threading.Lock()
    error_log_lock = threading.Lock()

    with futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_task = {
            executor.submit(
                process_task,
                task,
                api_key,
                max_retries,
                timeout_seconds,
            ): task
            for task in tasks
        }

        for future in futures.as_completed(future_to_task):
            task = future_to_task[future]
            state = states[task.split_name]

            try:
                finished_task, rationale, raw_response = future.result()

                with state.lock:
                    row = copy.deepcopy(state.data[finished_task.index])
                    row["rationale"] = rationale

                    # Keep this disabled by default to avoid storing extra SFT fields.
                    # Uncomment if you want to audit the exact accepted response later.
                    # row["rationale_raw_response"] = raw_response

                    state.data[finished_task.index] = row

                    # Save after every single successful rationale.
                    atomic_write_json(state.output_path, state.data)

                with progress_lock:
                    completed_now += 1
                    overall_done = already_done + completed_now
                    print(
                        f"[OK] {task.split_name}[{task.index}] "
                        f"progress={overall_done}/{total_rows}",
                        flush=True,
                    )

            except Exception as e:
                error_record = {
                    "split": task.split_name,
                    "index": task.index,
                    "claim": task.example.get("claim"),
                    "error": str(e),
                }
                failed.append(error_record)

                with error_log_lock:
                    append_error_log(output_dir, error_record)

                with progress_lock:
                    print(
                        f"[FAILED] {task.split_name}[{task.index}] "
                        f"error={e}",
                        file=sys.stderr,
                        flush=True,
                    )

    print("", flush=True)
    print(f"Completed in this run: {completed_now}")
    print(f"Failed in this run:    {len(failed)}")

    if failed:
        print(f"Failure log: {output_dir / 'generation_errors.jsonl'}")
        print(
            "Rows that failed were left without a rationale, so rerunning with "
            "--resume will retry only those unfinished rows."
        )


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rationale-enhanced CLIMATE-FEVER JSON files."
    )

    parser.add_argument("--train", required=True, type=Path, help="Path to climate_fever_train.json")
    parser.add_argument("--val", required=True, type=Path, help="Path to climate_fever_val.json")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for enhanced JSON files")

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of simultaneous API calls. Default: {DEFAULT_THREADS}",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max attempts per example. Default: {DEFAULT_MAX_RETRIES}",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=f"HTTP request timeout. Default: {DEFAULT_REQUEST_TIMEOUT_SECONDS}",
    )

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        help="Resume from existing enhanced output files.",
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore existing enhanced output files and start fresh.",
    )
    parser.set_defaults(resume=RESUME_FROM_EXISTING_OUTPUTS)

    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be >= 1")

    return args


def main() -> None:
    args = parse_args()

    run_generation(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output_dir,
        threads=args.threads,
        resume=args.resume,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()