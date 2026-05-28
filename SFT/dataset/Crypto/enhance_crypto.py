#!/usr/bin/env python3
"""
Generate rationale-enhanced Crypto JSON files using OpenRouter.

Usage:
  for linux/mac:
  export OPENROUTER_API_KEY="your_openrouter_key"

  for windows:
  set OPENROUTER_API_KEY=your_key_here

  python enhance_crypto.py \
    --train "../../dataset/Crypto/crypto_train.json" \
    --val "../../dataset/Crypto/crypto_val.json" \
    --output-dir ./ \
    --threads 8 \
    --resume

Outputs:
  ./crypto_train_enhanced.json
  ./crypto_val_enhanced.json

Each output sample keeps the original fields and adds:
  "rationale": "...inner text from <think>...</think>..."

The script only accepts model outputs that exactly match:

<think>
...
</think>

<answer>
def transform(s):
    ...
</answer>

The generated code inside <answer> is accepted only if:
  1. It is valid Python.
  2. It defines exactly one function named transform.
  3. transform takes exactly one argument named s.
  4. It passes all 10 test input-output examples for that row.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as futures
import copy
import json
import multiprocessing as mp
import os
import queue
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
DEFAULT_CODE_TIMEOUT_SECONDS = 5

TEMPERATURE = 0.2
TOP_P = 0.9
MAX_COMPLETION_TOKENS = 900

# Optional OpenRouter app-identifying headers.
# Leave as None unless you want OpenRouter analytics attribution.
HTTP_REFERER = None
X_TITLE = "crypto-rationale-generation"


SYSTEM_PROMPT_SYNTHETIC_CRYPTO_FUNCTION = """\
You are generating supervised fine-tuning data for a rule-induction-to-code model.

You will receive:
1. Several training examples with input and output strings
2. A gold Python function that implements the intended hidden transformation

Write a concise rationale explaining why the gold transformation fits the training examples, then provide a Python implementation of the same rule.

The rationale should:
- Start from the input-output examples.
- Identify the hidden character-level string transformation.
- Explain how the rule accounts for the outputs across examples.
- Mention whether the rule is a Caesar shift, Atbash mapping, or another exact character mapping when relevant.
- Briefly rule out a simpler wrong pattern if helpful.
- Emphasize that the rule should generalize beyond the shown examples.
- Avoid unrelated speculation.

Output exactly:

<think>
[A concise rule-induction rationale, usually 3–5 sentences.]
</think>

<answer>
[Python code only.]
</answer>

Code requirements:
- Define exactly one function named transform.
- The function takes one argument: s.
- The function must return a string.
- Use the gold function only to infer the intended rule.
- Preserve the behavior implied by the examples for lowercase letters, uppercase letters, and any non-letter characters.
- Do not hardcode the training examples.
- Do not use imports, printing, input(), or randomness.
- Do not include markdown code blocks.
- Do not include any text outside the Python function in the answer section.
""".strip()


STRICT_RESPONSE_RE = re.compile(
    r"\A<think>\n(?P<rationale>.+?)\n</think>\n\n<answer>\n(?P<code>.+?)\n</answer>\Z",
    re.DOTALL,
)


# =========================
# Prompt + validation
# =========================

def create_synthetic_crypto_function_prompt(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns:
      system_prompt, user_prompt
    """
    system_prompt = SYSTEM_PROMPT_SYNTHETIC_CRYPTO_FUNCTION

    train_examples = example["train"]
    if isinstance(train_examples, dict):
        # Use only the normal training examples, matching your RL/SFT task setup.
        train_examples = train_examples.get("normal", [])

    train_examples = train_examples[:10]

    train_prompt = "\n".join([
        f"Example {i + 1}:\nInput: {repr(ex['input'])}\nOutput: {repr(ex['output'])}"
        for i, ex in enumerate(train_examples)
    ])

    gold_function = example["function"]

    # Optional but useful metadata. The model should not rely only on this;
    # it should still explain the rule from the examples and gold function.
    split = example.get("split", None)
    split_text = f"\nTransformation Type Hint: {split}" if split else ""

    user_prompt = f"""\
Training examples:
{train_prompt}
{split_text}

Gold Function:
{gold_function}

Generate the SFT target.
""".strip()

    return system_prompt, user_prompt


def validate_and_extract_rationale_and_code(
    response_text: str,
    example: Dict[str, Any],
    code_timeout_seconds: int,
) -> Tuple[str, str]:
    """
    Strictly validates the model response.

    No fuzzy parsing.
    No extraction from malformed text.
    The whole stripped response must exactly match the requested format.

    Then validates generated code by compiling and running transform(s)
    against the row's 10 test examples.
    """
    normalized = response_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    match = STRICT_RESPONSE_RE.fullmatch(normalized)
    if not match:
        raise ValueError(f"Bad format. Raw response was:\n{response_text}")

    rationale = match.group("rationale").strip()
    code = match.group("code").strip()

    if not rationale:
        raise ValueError("Empty rationale.")

    if not code:
        raise ValueError("Empty code in answer section.")

    verify_generated_code_on_tests(
        code=code,
        test_examples=example["test"],
        timeout_seconds=code_timeout_seconds,
    )

    return rationale, code


# =========================
# Generated-code verification
# =========================

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "enumerate": enumerate,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "ord": ord,
    "range": range,
    "reversed": reversed,
    "set": set,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def validate_transform_ast(tree: ast.Module) -> None:
    """
    Enforce that answer code is only a single top-level function definition:

      def transform(s):
          ...

    Comments and blank lines are fine because they are not AST nodes.
    """
    if len(tree.body) != 1:
        raise ValueError("Generated code must contain exactly one top-level statement.")

    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef):
        raise ValueError("Generated code must define exactly one function.")

    if fn.name != "transform":
        raise ValueError(f"Generated function must be named transform, got {fn.name!r}.")

    args = fn.args

    positional_args = list(args.posonlyargs) + list(args.args)
    if len(positional_args) != 1:
        raise ValueError("transform must take exactly one positional argument.")

    if positional_args[0].arg != "s":
        raise ValueError(f"transform argument must be named 's', got {positional_args[0].arg!r}.")

    if args.vararg is not None:
        raise ValueError("transform must not use *args.")

    if args.kwonlyargs:
        raise ValueError("transform must not use keyword-only arguments.")

    if args.kwarg is not None:
        raise ValueError("transform must not use **kwargs.")

    if args.defaults or args.kw_defaults:
        raise ValueError("transform must not use default argument values.")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Generated code must not use imports.")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"print", "input", "eval", "exec", "open", "__import__"}:
                raise ValueError(f"Generated code must not call {node.func.id}().")


def _verification_worker(
    code: str,
    test_pairs: List[Tuple[str, str]],
    result_queue: mp.Queue,
) -> None:
    try:
        tree = ast.parse(code, mode="exec")
        validate_transform_ast(tree)

        env: Dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
        }

        compiled = compile(tree, "<generated_crypto_transform>", "exec")
        exec(compiled, env, env)

        transform = env.get("transform")
        if not callable(transform):
            raise ValueError("transform was not callable after execution.")

        for i, (input_text, expected_output) in enumerate(test_pairs):
            got = transform(input_text)

            if not isinstance(got, str):
                raise ValueError(
                    f"Test {i + 1} returned {type(got).__name__}, expected str."
                )

            if got != expected_output:
                raise ValueError(
                    f"Test {i + 1} failed: input={input_text!r}, "
                    f"expected={expected_output!r}, got={got!r}."
                )

        result_queue.put({"ok": True})

    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})


def verify_generated_code_on_tests(
    code: str,
    test_examples: List[Dict[str, Any]],
    timeout_seconds: int,
) -> None:
    """
    Compile and run generated transform(s) on exactly the row's 10 test samples.
    Accept only if every output is correct.
    """
    if not isinstance(test_examples, list):
        raise ValueError("example['test'] must be a list.")

    if len(test_examples) != 10:
        raise ValueError(f"Expected exactly 10 test examples, got {len(test_examples)}.")

    test_pairs: List[Tuple[str, str]] = []
    for i, ex in enumerate(test_examples):
        if not isinstance(ex, dict):
            raise ValueError(f"Test example {i} is not a JSON object.")

        if "input" not in ex or "output" not in ex:
            raise ValueError(f"Test example {i} is missing input/output.")

        if not isinstance(ex["input"], str) or not isinstance(ex["output"], str):
            raise ValueError(f"Test example {i} input/output must be strings.")

        test_pairs.append((ex["input"], ex["output"]))

    result_queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_verification_worker,
        args=(code, test_pairs, result_queue),
    )

    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(
            f"Generated code verification timed out after {timeout_seconds} seconds."
        )

    try:
        result = result_queue.get(timeout=1.0)
    except queue.Empty:
        raise RuntimeError(
            f"Generated code verification process exited with code {proc.exitcode} "
            "without returning a result."
        )

    if not result.get("ok"):
        raise ValueError(result.get("error", "Generated code failed verification."))


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

        required = {"train", "test", "index", "split", "function"}
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"{path} item {i} is missing fields: {sorted(missing)}")

        validate_crypto_row(path=path, row_index=i, item=item)

    return data


def validate_example_pair(path: Path, row_index: int, section: str, pair_index: int, ex: Any) -> None:
    if not isinstance(ex, dict):
        raise ValueError(f"{path} item {row_index} {section}[{pair_index}] is not an object.")

    if "input" not in ex or "output" not in ex:
        raise ValueError(
            f"{path} item {row_index} {section}[{pair_index}] is missing input/output."
        )

    if not isinstance(ex["input"], str) or not isinstance(ex["output"], str):
        raise ValueError(
            f"{path} item {row_index} {section}[{pair_index}] input/output must be strings."
        )


def validate_crypto_row(path: Path, row_index: int, item: Dict[str, Any]) -> None:
    if not isinstance(item["function"], str) or not item["function"].strip():
        raise ValueError(f"{path} item {row_index} has invalid function.")

    if not isinstance(item["split"], str) or not item["split"].strip():
        raise ValueError(f"{path} item {row_index} has invalid split.")

    train_examples = item["train"]
    if isinstance(train_examples, dict):
        normal_examples = train_examples.get("normal", [])
        if not isinstance(normal_examples, list):
            raise ValueError(f"{path} item {row_index} train['normal'] must be a list.")
        train_examples_to_check = normal_examples

    elif isinstance(train_examples, list):
        train_examples_to_check = train_examples

    else:
        raise ValueError(f"{path} item {row_index} train must be a list or dict.")

    if len(train_examples_to_check) < 10:
        raise ValueError(
            f"{path} item {row_index} must have at least 10 normal train examples."
        )

    for j, ex in enumerate(train_examples_to_check[:10]):
        validate_example_pair(path, row_index, "train", j, ex)

    test_examples = item["test"]
    if not isinstance(test_examples, list):
        raise ValueError(f"{path} item {row_index} test must be a list.")

    if len(test_examples) != 10:
        raise ValueError(
            f"{path} item {row_index} must have exactly 10 test examples, got {len(test_examples)}."
        )

    for j, ex in enumerate(test_examples):
        validate_example_pair(path, row_index, "test", j, ex)


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


def row_key(example: Dict[str, Any]) -> str:
    return f"split={example.get('split')!r}, index={example.get('index')!r}"


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
            if inp.get("split") != out.get("split"):
                raise ValueError(
                    f"Cannot resume from {output_path}: split mismatch at row {i}. "
                    f"Input split={inp.get('split')!r}, output split={out.get('split')!r}."
                )

            if inp.get("index") != out.get("index"):
                raise ValueError(
                    f"Cannot resume from {output_path}: index mismatch at row {i}. "
                    f"Input index={inp.get('index')!r}, output index={out.get('index')!r}."
                )

            if inp.get("function") != out.get("function"):
                raise ValueError(
                    f"Cannot resume from {output_path}: function mismatch at row {i}."
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
    code_timeout_seconds: int,
) -> Tuple[str, str, str]:
    """
    Returns:
      rationale, generated_code, raw_valid_response
    """
    system_prompt, user_prompt = create_synthetic_crypto_function_prompt(example)

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_openrouter_once(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=timeout_seconds,
            )

            rationale, generated_code = validate_and_extract_rationale_and_code(
                response_text=raw_response,
                example=example,
                code_timeout_seconds=code_timeout_seconds,
            )

            return rationale, generated_code, raw_response

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
    code_timeout_seconds: int,
) -> Tuple[Task, str, str, str]:
    rationale, generated_code, raw_response = generate_rationale_with_retries(
        api_key=api_key,
        example=task.example,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        code_timeout_seconds=code_timeout_seconds,
    )
    return task, rationale, generated_code, raw_response


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
    code_timeout_seconds: int,
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
                code_timeout_seconds,
            ): task
            for task in tasks
        }

        for future in futures.as_completed(future_to_task):
            task = future_to_task[future]
            state = states[task.split_name]

            try:
                finished_task, rationale, generated_code, raw_response = future.result()

                with state.lock:
                    row = copy.deepcopy(state.data[finished_task.index])
                    row["rationale"] = rationale

                    # Keep these disabled by default to avoid storing extra SFT fields.
                    # The original gold code is already stored in row["function"].
                    # Uncomment if you want to audit the accepted generated code/response later.
                    # row["generated_function"] = generated_code
                    # row["rationale_raw_response"] = raw_response

                    state.data[finished_task.index] = row

                    # Save after every single successful rationale.
                    atomic_write_json(state.output_path, state.data)

                with progress_lock:
                    completed_now += 1
                    overall_done = already_done + completed_now
                    print(
                        f"[OK] {task.split_name}[{task.index}] "
                        f"{row_key(task.example)} "
                        f"progress={overall_done}/{total_rows}",
                        flush=True,
                    )

            except Exception as e:
                error_record = {
                    "split": task.split_name,
                    "row": task.index,
                    "crypto_split": task.example.get("split"),
                    "crypto_index": task.example.get("index"),
                    "error": str(e),
                }
                failed.append(error_record)

                with error_log_lock:
                    append_error_log(output_dir, error_record)

                with progress_lock:
                    print(
                        f"[FAILED] {task.split_name}[{task.index}] "
                        f"{row_key(task.example)} error={e}",
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
        description="Generate rationale-enhanced Crypto JSON files."
    )

    parser.add_argument("--train", required=True, type=Path, help="Path to crypto_train.json")
    parser.add_argument("--val", required=True, type=Path, help="Path to crypto_val.json")
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

    parser.add_argument(
        "--code-timeout-seconds",
        type=int,
        default=DEFAULT_CODE_TIMEOUT_SECONDS,
        help=f"Timeout for running generated code on the 10 test samples. Default: {DEFAULT_CODE_TIMEOUT_SECONDS}",
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
    if args.code_timeout_seconds < 1:
        parser.error("--code-timeout-seconds must be >= 1")

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
        code_timeout_seconds=args.code_timeout_seconds,
    )


if __name__ == "__main__":
    # Needed for multiprocessing compatibility, especially on Windows.
    mp.freeze_support()
    main()