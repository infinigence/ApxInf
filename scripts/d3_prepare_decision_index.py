#!/usr/bin/env python3
"""Prepare the public Decision Index suite for native ApxInf d3 evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from decision_index.suite.io import Suite, read_jsonl
from transformers import AutoTokenizer


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite-dir", type=Path)
    source.add_argument("--rows", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--edition", default="0.3")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    d3_format = load_module("d3_format", args.model_dir / "d3_format.py")
    decision = json.loads((args.model_dir / "decision_config.json").read_text())
    codes = decision["codes"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    rows = (
        read_jsonl(args.rows)
        if args.rows is not None
        else Suite(args.suite_dir, args.edition).rows(apply_exclusions=True)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    request_count = 0
    question_count = 0
    with args.out.open("w", encoding="utf-8") as output:
        for row in rows:
            questions = []
            for key, question in row["questions"].items():
                candidate_keys = d3_format.options(question)[0]
                prompt = d3_format.render(tokenizer, row["state"], question, codes)
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                questions.append(
                    {
                        "key": key,
                        "type": question["type"],
                        "candidate_keys": candidate_keys,
                        "token_ids": token_ids,
                    }
                )
                question_count += 1
            record = {**row["_evaluation"], "questions": questions}
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            request_count += 1
            if request_count % 1_000 == 0:
                print(f"progress={request_count}", flush=True)
            if args.limit is not None and request_count >= args.limit:
                break
    print(
        f"wrote {request_count} requests / {question_count} questions to {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
