#!/usr/bin/env python3
"""Prepare the public 1,050-row JEV-Benchmark for ApxInf d3 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer


DATASETS = ("banking77", "clinc150", "sst5", "boolq", "agnews")
EXPECTED_SHA256 = {
    "banking77": "0862f30acbb35c6b4429a8cf76f2a86884df39959a751090308982fe0140ed80",
    "clinc150": "aaac80ce9be550f522d4163ba1e18130e06e1ce1cd28c7d1caa5dca090083b97",
    "sst5": "74592bb801733e1b940854d91cff9fb07b2fd5bbe9d153ca4353b83dd1777282",
    "boolq": "6743afe1653eed902c7521d912f8c942f069fd94f5a31aeea30c31039027f709",
    "agnews": "b9d554701d436af901ffd9240503c3ff8ce05efd0ed4c0fba2ae53fe1a5b9cf8",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_question(question: dict) -> tuple[dict, list[str]]:
    kind = question["type"]
    if kind == "choice":
        return question, list(question["criteria"])
    if kind == "noul":
        return question, ["false", "true"]
    if kind == "score":
        levels = [str(value) for value in question["criteria"]]
        return {
            "type": "choice",
            "instructions": question.get("instructions"),
            "criteria": {level: None for level in levels},
        }, [str(index) for index in range(len(levels))]
    raise ValueError(f"unsupported question type {kind!r}")


def request(sample: dict) -> dict:
    task_type = sample["task_type"]
    if task_type == "choice":
        descriptions = sample.get("label_descriptions") or {}
        criteria = {
            label: descriptions.get(label, label.replace("_", " "))
            for label in sample["labels"]
        }
        question = {
            "type": "choice",
            "instructions": "Which option best describes the text?",
            "criteria": criteria,
        }
    elif task_type == "score":
        descriptions = sample.get("level_descriptions") or {}
        levels = sample.get("levels") or [0, 1, 2, 3, 4]
        question = {
            "type": "score",
            "instructions": "Rate the sentiment / intensity level of the text.",
            "criteria": [descriptions.get(str(level), str(level)) for level in levels],
        }
    elif task_type == "noul":
        question = {
            "type": "noul",
            "instructions": "Is the answer to the question in the text true?",
            "criteria": {
                "true": "Yes / affirmative",
                "false": "No / negative",
            },
        }
    else:
        raise ValueError(f"unsupported task type {task_type!r}")
    return {"state": sample["text"], "questions": {"q": question}}


def expected_index(sample: dict, keys: list[str]) -> int:
    if sample["task_type"] == "noul":
        label = sample["label"]
        if isinstance(label, bool):
            return int(label)
        normalized = str(label).lower()
        if normalized not in ("false", "true"):
            raise ValueError(f"invalid noul label {label!r}")
        return int(normalized == "true")
    label = str(sample["label"])
    if label not in keys:
        raise ValueError(f"label {label!r} is not in {keys!r}")
    return keys.index(label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    d3_format = load_module("d3_format", args.model_dir / "d3_format.py")
    decision = json.loads((args.model_dir / "decision_config.json").read_text())
    codes = decision["codes"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8") as output:
        for dataset in DATASETS:
            source = args.benchmark_dir / "datasets" / f"{dataset}.jsonl"
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != EXPECTED_SHA256[dataset]:
                raise ValueError(
                    f"{source} has sha256 {digest}, expected {EXPECTED_SHA256[dataset]}"
                )
            with source.open(encoding="utf-8") as rows:
                for line in rows:
                    sample_dict = json.loads(line)
                    prepared = request(sample_dict)
                    question, keys = normalize_question(prepared["questions"]["q"])
                    prompt = d3_format.render(
                        tokenizer, prepared["state"], question, codes
                    )
                    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                    record = {
                        "id": sample_dict["id"],
                        "dataset": dataset,
                        "task_type": sample_dict["task_type"],
                        "token_ids": token_ids,
                        "candidate_count": len(keys),
                        "expected_index": expected_index(sample_dict, keys),
                        "oos_index": keys.index("oos") if "oos" in keys else None,
                    }
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
                    total += 1
    if total != 1050:
        raise ValueError(f"expected 1050 records, wrote {total}")
    print(f"wrote {total} records to {args.out}")


if __name__ == "__main__":
    main()
