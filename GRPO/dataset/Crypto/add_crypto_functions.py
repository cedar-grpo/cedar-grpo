#!/usr/bin/env python3

import argparse
import ast
import json
import os
import shutil
from collections import OrderedDict


DEFAULT_FILES = ["crypto_train.json", "crypto_val.json"]


def atbash_char(ch):
    if "a" <= ch <= "z":
        return chr(ord("z") - (ord(ch) - ord("a")))
    if "A" <= ch <= "Z":
        return chr(ord("Z") - (ord(ch) - ord("A")))
    return ch


def caesar_char(ch, shift):
    if "a" <= ch <= "z":
        return chr((ord(ch) - ord("a") + shift) % 26 + ord("a"))
    if "A" <= ch <= "Z":
        return chr((ord(ch) - ord("A") + shift) % 26 + ord("A"))
    return ch


def apply_atbash(s):
    return "".join(atbash_char(ch) for ch in s)


def apply_caesar(s, shift):
    return "".join(caesar_char(ch, shift) for ch in s)


def get_examples(entry, include_test=True, include_ood=True, include_noise=False):
    examples = []

    train = entry.get("train", {})
    examples.extend(train.get("normal", []))

    if include_ood:
        examples.extend(train.get("ood", []))

    if include_noise:
        examples.extend(train.get("noise", []))

    if include_test:
        examples.extend(entry.get("test", []))

    return examples


def examples_match_atbash(examples):
    for ex in examples:
        if apply_atbash(ex["input"]) != ex["output"]:
            return False
    return True


def examples_match_caesar(examples, shift):
    for ex in examples:
        if apply_caesar(ex["input"], shift) != ex["output"]:
            return False
    return True


def infer_caesar_shift(entry):
    inference_examples = get_examples(
        entry,
        include_test=False,
        include_ood=True,
        include_noise=False,
    )

    valid_shifts = []
    for shift in range(26):
        if examples_match_caesar(inference_examples, shift):
            valid_shifts.append(shift)

    if len(valid_shifts) != 1:
        index = entry.get("index", "<unknown>")
        raise ValueError(
            f"Entry index {index}: expected exactly one Caesar shift, "
            f"found {valid_shifts}"
        )

    return valid_shifts[0]


def make_atbash_function():
    return """def transform(s):
    result = ""
    for ch in s:
        if "a" <= ch <= "z":
            result += chr(ord("z") - (ord(ch) - ord("a")))
        elif "A" <= ch <= "Z":
            result += chr(ord("Z") - (ord(ch) - ord("A")))
        else:
            result += ch
    return result"""


def make_caesar_function(shift):
    return f"""def transform(s):
    result = ""
    shift = {shift}
    for ch in s:
        if "a" <= ch <= "z":
            result += chr((ord(ch) - ord("a") + shift) % 26 + ord("a"))
        elif "A" <= ch <= "Z":
            result += chr((ord(ch) - ord("A") + shift) % 26 + ord("A"))
        else:
            result += ch
    return result"""


def infer_function(entry):
    split = entry.get("split")

    if split == "atbash":
        inference_examples = get_examples(
            entry,
            include_test=False,
            include_ood=True,
            include_noise=False,
        )

        if not examples_match_atbash(inference_examples):
            index = entry.get("index", "<unknown>")
            raise ValueError(
                f"Entry index {index}: split is atbash, "
                f"but clean train examples do not match atbash"
            )

        return make_atbash_function()

    if split == "caesar":
        shift = infer_caesar_shift(entry)
        return make_caesar_function(shift)

    index = entry.get("index", "<unknown>")
    raise ValueError(f"Entry index {index}: unknown split {split!r}")


def validate_transform_source(source):
    """
    Static validation before execution.

    This enforces the same structural constraints expected by the prompt:
    - exactly one top-level function
    - function must be named transform
    - no imports
    - no class definitions
    - no async code
    - no global/nonlocal mutation
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Generated function has invalid Python syntax: {exc}") from exc

    if len(tree.body) != 1:
        raise ValueError("Generated source must contain exactly one top-level statement")

    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef):
        raise ValueError("Generated source must contain exactly one top-level function")

    if fn.name != "transform":
        raise ValueError("Generated function must be named transform")

    if len(fn.args.args) != 1 or fn.args.args[0].arg != "s":
        raise ValueError("Generated transform function must take exactly one argument named s")

    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.defaults or fn.args.kw_defaults:
        raise ValueError("Generated transform function must not use varargs, kwargs, or defaults")

    forbidden_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.Delete,
    )

    for node in ast.walk(tree):
        if isinstance(node, forbidden_nodes):
            raise ValueError(f"Forbidden Python construct in generated function: {type(node).__name__}")

    return tree


def compile_transform_in_sandbox(source):
    """
    Compile and execute the generated code in a restricted namespace.

    The generated templates only need chr() and ord().
    __import__ is intentionally unavailable.
    """
    tree = validate_transform_source(source)

    allowed_builtins = {
        "chr": chr,
        "ord": ord,
    }

    sandbox_globals = {
        "__builtins__": allowed_builtins,
    }
    sandbox_locals = {}

    compiled = compile(tree, filename="<generated_transform>", mode="exec")
    exec(compiled, sandbox_globals, sandbox_locals)

    transform = sandbox_locals.get("transform")
    if transform is None:
        transform = sandbox_globals.get("transform")

    if not callable(transform):
        raise ValueError("Generated source did not define a callable transform function")

    return transform


def verify_function_on_test_samples(entry, function_source):
    test_examples = entry.get("test", [])
    index = entry.get("index", "<unknown>")

    if len(test_examples) != 10:
        raise ValueError(
            f"Entry index {index}: expected exactly 10 test samples, "
            f"found {len(test_examples)}"
        )

    transform = compile_transform_in_sandbox(function_source)

    for sample_idx, ex in enumerate(test_examples):
        input_text = ex["input"]
        expected = ex["output"]

        try:
            actual = transform(input_text)
        except Exception as exc:
            raise RuntimeError(
                f"Entry index {index}, test sample {sample_idx}: "
                f"generated transform crashed on input {input_text!r}: {exc}"
            ) from exc

        if not isinstance(actual, str):
            raise TypeError(
                f"Entry index {index}, test sample {sample_idx}: "
                f"transform returned {type(actual).__name__}, expected str"
            )

        if actual != expected:
            raise ValueError(
                f"Entry index {index}, test sample {sample_idx}: "
                f"test verification failed. "
                f"input={input_text!r}, expected={expected!r}, actual={actual!r}"
            )


def validate_clean_examples(entry):
    """
    Optional deterministic sanity check against clean non-noise examples.

    Noise examples are intentionally excluded.
    """
    split = entry.get("split")
    clean_examples = get_examples(
        entry,
        include_test=True,
        include_ood=True,
        include_noise=False,
    )

    if split == "atbash":
        if not examples_match_atbash(clean_examples):
            index = entry.get("index", "<unknown>")
            raise ValueError(f"Entry index {index}: atbash rule fails on clean examples")
        return

    if split == "caesar":
        shift = infer_caesar_shift(entry)
        if not examples_match_caesar(clean_examples, shift):
            index = entry.get("index", "<unknown>")
            raise ValueError(f"Entry index {index}: Caesar shift {shift} fails on clean examples")
        return

    index = entry.get("index", "<unknown>")
    raise ValueError(f"Entry index {index}: unknown split {split!r}")


def prepare_dataset_with_verified_functions(data, source_name):
    if not isinstance(data, list):
        raise TypeError(f"{source_name}: expected top-level JSON value to be a list of entries")

    prepared = []

    for position, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise TypeError(f"{source_name}: entry at position {position} is not an object")

        validate_clean_examples(entry)

        function_source = infer_function(entry)

        # Final requested gate:
        # compile/run candidate code in a sandbox on all 10 test samples.
        # The field is only added after this passes.
        verify_function_on_test_samples(entry, function_source)

        entry["function"] = function_source
        prepared.append(entry)

    return prepared


def load_json_ordered(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def write_json_ordered(path, data, make_backup=True):
    if make_backup:
        backup_path = path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(path, backup_path)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    os.replace(tmp_path, path)


def process_all_files(paths, make_backup=True):
    """
    Two-phase update.

    Phase 1:
      Load every file, infer every function, sandbox-verify every function.

    Phase 2:
      Only if phase 1 succeeds completely, write all updated files.

    This prevents partially modified outputs if a later file or entry fails.
    """
    prepared_by_path = OrderedDict()

    for path in paths:
        data = load_json_ordered(path)
        prepared = prepare_dataset_with_verified_functions(data, source_name=path)
        prepared_by_path[path] = prepared

    for path, prepared in prepared_by_path.items():
        write_json_ordered(path, prepared, make_backup=make_backup)
        print(f"Updated {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="JSON files to update. Defaults to crypto_train.json crypto_val.json",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup files before overwriting",
    )
    args = parser.parse_args()

    process_all_files(args.files, make_backup=not args.no_backup)


if __name__ == "__main__":
    main()