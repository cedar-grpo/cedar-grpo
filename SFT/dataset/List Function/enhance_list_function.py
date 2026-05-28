#!/usr/bin/env python3
"""
Generate rationale-enhanced List Function JSON files using OpenRouter.

Usage:
  for linux/mac:
  export OPENROUTER_API_KEY="your_openrouter_key"

  for windows:
  set OPENROUTER_API_KEY=your_key_here

  python enhance_list_function.py \
    --train "../../dataset/List Function/list_function_train.json" \
    --val "../../dataset/List Function/list_function_val.json" \
    --output-dir ./ \
    --threads 8 \
    --resume

Outputs:
  ./list_function_train_enhanced.json
  ./list_function_val_enhanced.json

Each output sample keeps the original fields and adds:
  "rationale": "...inner text from <think>...</think>..."

The script only accepts model outputs that exactly match:

<think>
...
</think>

<answer>
def transform(lst):
    ...
</answer>

The answer code is accepted only if:
  1. it is valid Python code;
  2. it defines exactly one function named transform;
  3. transform takes exactly one argument named lst;
  4. it uses no imports, print(), input(), or randomness;
  5. it returns the correct output list for every test example in the sample.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as futures
import copy
import json
import multiprocessing as mp
import os
import random
import re
import sys
import tempfile
import threading
import time
import traceback
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
DEFAULT_CODE_EXECUTION_TIMEOUT_SECONDS = 3

TEMPERATURE = 0.2
TOP_P = 0.9
MAX_COMPLETION_TOKENS = 1000

# Optional OpenRouter app-identifying headers.
# Leave as None unless you want OpenRouter analytics attribution.
HTTP_REFERER = None
X_TITLE = "list-function-rationale-generation"


SYSTEM_PROMPT_SYNTHETIC_LIST_FUNCTION = """\
You are generating supervised fine-tuning data for a rule-induction-to-code model.

You will receive:
1. Several training examples with input and output lists
2. A gold Python function that implements the intended hidden transformation

Write a concise rationale explaining why the gold transformation fits the training examples, then provide a Python implementation of the same rule.

The rationale should:
- Start from the input-output examples.
- Identify the hidden list transformation.
- Explain how the rule accounts for the outputs.
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
- The function takes one argument: lst.
- The function must return a list of integers.
- Use the gold function only to infer the intended rule; adapt the function name and argument name to transform(lst).
- Do not hardcode the training examples.
- Do not use imports, printing, input(), or randomness.
- Do not include markdown code blocks.
- Do not include any text outside the Python function in the answer section.
""".strip()


STRICT_RESPONSE_RE = re.compile(
    r"\A<think>\n(?P<rationale>.+?)\n</think>\n\n<answer>\n(?P<code>.+?)\n</answer>\Z",
    re.DOTALL,
)

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

BANNED_CALL_NAMES = {"print", "input", "eval", "exec", "open", "compile", "__import__"}
BANNED_NAME_REFERENCES = {"random"}


# =========================
# Prompt + validation
# =========================

def create_synthetic_list_function_prompt(example: Dict[str, Any]) -> Tuple[str, str]:
    system_prompt = SYSTEM_PROMPT_SYNTHETIC_LIST_FUNCTION

    train_prompt = "\n".join([
        f"--- Example {i + 1} ---\nInput: {ex['input']}\nOutput: {ex['output']}"
        for i, ex in enumerate(example["train"])
    ])

    gold_function = example["function"]

    user_prompt = f"""\
Training examples:
{train_prompt}

Gold Function:
{gold_function}

Generate the SFT target.
""".strip()

    return system_prompt, user_prompt


def parse_int_list_literal(value: Any, context: str) -> List[int]:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string containing a Python list literal.")

    try:
        parsed = ast.literal_eval(value)
    except Exception as e:
        raise ValueError(f"{context} is not a valid Python literal: {value!r}") from e

    if not isinstance(parsed, list):
        raise ValueError(f"{context} must parse to a list, got {type(parsed).__name__}.")

    if not all(isinstance(x, int) and not isinstance(x, bool) for x in parsed):
        raise ValueError(f"{context} must be a list of integers, got {parsed!r}.")

    return parsed


def parse_io_pairs(examples: List[Dict[str, Any]], context: str) -> List[Tuple[List[int], List[int]]]:
    pairs: List[Tuple[List[int], List[int]]] = []
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            raise ValueError(f"{context}[{i}] is not a JSON object.")
        if "input" not in ex or "output" not in ex:
            raise ValueError(f"{context}[{i}] must contain 'input' and 'output'.")

        inp = parse_int_list_literal(ex["input"], f"{context}[{i}].input")
        out = parse_int_list_literal(ex["output"], f"{context}[{i}].output")
        pairs.append((inp, out))
    return pairs


def validate_candidate_code_static(code: str) -> None:
    if not code.strip():
        raise ValueError("Empty code in answer section.")
    if "```" in code:
        raise ValueError("Markdown code fence found in answer section.")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Generated code is not valid Python: {e}") from e

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Answer section must contain exactly one top-level function definition.")

    fn = tree.body[0]
    if fn.name != "transform":
        raise ValueError(f"Function must be named transform, got {fn.name!r}.")

    if fn.decorator_list:
        raise ValueError("Function must not use decorators.")

    args = fn.args
    if args.posonlyargs:
        raise ValueError("transform must not use positional-only arguments.")
    if len(args.args) != 1 or args.args[0].arg != "lst":
        raise ValueError("transform must take exactly one positional argument named lst.")
    if args.vararg is not None or args.kwarg is not None:
        raise ValueError("transform must not use *args or **kwargs.")
    if args.kwonlyargs:
        raise ValueError("transform must not use keyword-only arguments.")
    if args.defaults or args.kw_defaults:
        raise ValueError("transform must not define default argument values.")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Generated code must not use imports.")

        if isinstance(node, ast.Call):
            called_name: Optional[str] = None
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called_name = node.func.attr

            if called_name in BANNED_CALL_NAMES:
                raise ValueError(f"Generated code must not call {called_name}().")

        if isinstance(node, ast.Name) and node.id in BANNED_NAME_REFERENCES:
            raise ValueError(f"Generated code must not reference {node.id!r}.")


def _candidate_code_worker(
    code: str,
    test_pairs: List[Tuple[List[int], List[int]]],
    result_queue: Any,
) -> None:
    try:
        namespace: Dict[str, Any] = {}
        globals_dict = {"__builtins__": SAFE_BUILTINS}

        compiled = compile(code, "<generated_transform>", "exec")
        exec(compiled, globals_dict, namespace)

        transform = namespace.get("transform")
        if not callable(transform):
            result_queue.put((False, "transform was not defined as a callable."))
            return

        for i, (inp, expected) in enumerate(test_pairs):
            got = transform(copy.deepcopy(inp))
            if not isinstance(got, list):
                result_queue.put((False, f"Test {i + 1}: transform returned {type(got).__name__}, not list."))
                return
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in got):
                result_queue.put((False, f"Test {i + 1}: transform returned a non-integer list: {got!r}."))
                return
            if got != expected:
                result_queue.put((
                    False,
                    f"Test {i + 1}: input={inp!r}, expected={expected!r}, got={got!r}.",
                ))
                return

        result_queue.put((True, ""))

    except BaseException:
        result_queue.put((False, traceback.format_exc(limit=3)))


def verify_candidate_code_on_tests(
    code: str,
    test_pairs: List[Tuple[List[int], List[int]]],
    execution_timeout_seconds: int,
) -> None:
    """
    Runs generated transform(lst) against the sample's test examples.

    A separate process is used so malformed generated code cannot hang the main
    generation process forever.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_candidate_code_worker,
        args=(code, test_pairs, result_queue),
    )

    process.start()
    process.join(execution_timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(1)
        raise ValueError(
            f"Generated code timed out after {execution_timeout_seconds} seconds."
        )

    if result_queue.empty():
        raise ValueError(
            f"Generated code process exited without returning a validation result "
            f"(exitcode={process.exitcode})."
        )

    ok, message = result_queue.get()
    if not ok:
        raise ValueError(f"Generated code failed test verification: {message}")


def validate_and_extract_rationale(
    response_text: str,
    example: Dict[str, Any],
    execution_timeout_seconds: int,
) -> str:
    """
    Strictly validates the model response.

    No fuzzy parsing.
    No extraction from malformed text.
    The whole stripped response must exactly match the requested format.

    The answer code is then compiled and run against every test input/output pair
    for the current sample. The rationale is returned only if the code passes.
    """
    normalized = response_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    match = STRICT_RESPONSE_RE.fullmatch(normalized)
    if not match:
        raise ValueError(f"Bad format. Raw response was:\n{response_text}")

    rationale = match.group("rationale").strip()
    code = match.group("code").strip()

    if not rationale:
        raise ValueError("Empty rationale.")

    validate_candidate_code_static(code)
    test_pairs = parse_io_pairs(example["test"], "test")
    verify_candidate_code_on_tests(
        code=code,
        test_pairs=test_pairs,
        execution_timeout_seconds=execution_timeout_seconds,
    )

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

        required = {"idx", "id", "function", "train", "test"}
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"{path} item {i} is missing fields: {sorted(missing)}")

        if not isinstance(item["id"], str) or not item["id"].strip():
            raise ValueError(f"{path} item {i} has invalid id: {item['id']!r}")

        if not isinstance(item["function"], str) or not item["function"].strip():
            raise ValueError(f"{path} item {i} has invalid function field.")

        for split_name in ("train", "test"):
            if not isinstance(item[split_name], list):
                raise ValueError(f"{path} item {i} field {split_name!r} must be a list.")
            if len(item[split_name]) != 10:
                raise ValueError(
                    f"{path} item {i} field {split_name!r} must contain exactly 10 examples; "
                    f"got {len(item[split_name])}."
                )
            parse_io_pairs(item[split_name], f"{path} item {i} {split_name}")

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
            if inp.get("id") != out.get("id"):
                raise ValueError(
                    f"Cannot resume from {output_path}: id mismatch at row {i}. "
                    f"Input id={inp.get('id')!r}, output id={out.get('id')!r}."
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
    code_execution_timeout_seconds: int,
) -> Tuple[str, str]:
    """
    Returns:
      rationale, raw_valid_response
    """
    system_prompt, user_prompt = create_synthetic_list_function_prompt(example)

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = call_openrouter_once(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=timeout_seconds,
            )
            rationale = validate_and_extract_rationale(
                response_text=raw_response,
                example=example,
                execution_timeout_seconds=code_execution_timeout_seconds,
            )
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
    code_execution_timeout_seconds: int,
) -> Tuple[Task, str, str]:
    rationale, raw_response = generate_rationale_with_retries(
        api_key=api_key,
        example=task.example,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        code_execution_timeout_seconds=code_execution_timeout_seconds,
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
    code_execution_timeout_seconds: int,
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
    print(f"Code execution timeout: {code_execution_timeout_seconds}s")
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
                code_execution_timeout_seconds,
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
                        f"id={task.example.get('id')} "
                        f"progress={overall_done}/{total_rows}",
                        flush=True,
                    )

            except Exception as e:
                error_record = {
                    "split": task.split_name,
                    "index": task.index,
                    "id": task.example.get("id"),
                    "error": str(e),
                }
                failed.append(error_record)

                with error_log_lock:
                    append_error_log(output_dir, error_record)

                with progress_lock:
                    print(
                        f"[FAILED] {task.split_name}[{task.index}] "
                        f"id={task.example.get('id')} error={e}",
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
        description="Generate rationale-enhanced List Function JSON files."
    )

    parser.add_argument("--train", required=True, type=Path, help="Path to list_function_train.json")
    parser.add_argument("--val", required=True, type=Path, help="Path to list_function_val.json")
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
        "--code-execution-timeout-seconds",
        type=int,
        default=DEFAULT_CODE_EXECUTION_TIMEOUT_SECONDS,
        help=(
            "Max seconds allowed for running generated code against one sample's "
            f"test cases. Default: {DEFAULT_CODE_EXECUTION_TIMEOUT_SECONDS}"
        ),
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
    if args.code_execution_timeout_seconds < 1:
        parser.error("--code-execution-timeout-seconds must be >= 1")

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
        code_execution_timeout_seconds=args.code_execution_timeout_seconds,
    )


if __name__ == "__main__":
    main()
