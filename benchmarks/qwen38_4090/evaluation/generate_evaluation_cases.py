#!/usr/bin/env python3
"""Generate exact-token public, context, and qualitative evaluation cases.

The contract is the workload source of truth. The generated JSONL contains
pretokenized input IDs so every backend receives identical model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

CASE_SCHEMA = "apxinf.qwen38_27b.evaluation_case.v1"
MANIFEST_SCHEMA = "apxinf.qwen38_27b.dataset_manifest.v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def load_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required to generate tokenized datasets"
        ) from error
    return AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )


def chat_token_ids(tokenizer, content: str) -> list[int]:
    messages = [{"role": "user", "content": content}]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    try:
        token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("preserve_thinking", None)
        try:
            token_ids = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    return [int(token_id) for token_id in token_ids]


def extract_corpus(path: Path, expected_sha256: str) -> str:
    payload = path.read_bytes()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path}: corpus SHA-256 mismatch; expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    text = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
    end = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
    if not start or not end or end.start() <= start.end():
        raise ValueError(f"{path}: Project Gutenberg body markers not found")
    body = text[start.end() : end.start()]
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) < 100_000:
        raise ValueError(f"{path}: extracted corpus is unexpectedly short")
    return body


def cyclic_text(text: str, offset: int, length: int) -> str:
    if length <= 0:
        return ""
    offset %= len(text)
    first = text[offset : offset + length]
    if len(first) == length:
        return first
    remaining = length - len(first)
    repeats, tail = divmod(remaining, len(text))
    return first + text * repeats + text[:tail]


def insert_records(body: str, records: Sequence[tuple[float, str]]) -> str:
    pieces: list[str] = []
    cursor = 0
    for fraction, record in sorted(records, key=lambda item: item[0]):
        point = min(len(body), max(cursor, int(len(body) * fraction)))
        pieces.append(body[cursor:point])
        pieces.append(f"\n\n{record}\n\n")
        cursor = point
    pieces.append(body[cursor:])
    return "".join(pieces)


def fit_exact_prompt(
    token_ids: Callable[[str], list[int]],
    target_tokens: int,
    corpus: str,
    corpus_offset: int,
    header: str,
    records: Sequence[tuple[float, str]],
    question: str,
) -> tuple[str, list[int]]:
    """Fit a semantic prompt to an exact chat-template token count."""

    def compose(body_chars: int, calibration: str = "") -> str:
        body = cyclic_text(corpus, corpus_offset, body_chars)
        body = insert_records(body, records)
        return f"{header}\n\n{body}{calibration}\n\n{question}"

    shell_ids = token_ids(compose(0))
    if len(shell_ids) > target_tokens:
        raise ValueError(
            f"target {target_tokens} is below semantic shell length {len(shell_ids)}"
        )

    low = 0
    high = max(1024, target_tokens * 3)
    while len(token_ids(compose(high))) <= target_tokens:
        low = high
        high *= 2
        if high > len(corpus) * 2:
            raise ValueError("corpus cannot fill the requested token length")
    while low + 1 < high:
        middle = (low + high) // 2
        if len(token_ids(compose(middle))) <= target_tokens:
            low = middle
        else:
            high = middle

    calibration_atoms = (" x", " 的", "。", " 0", "\n", "a", "〇")
    for body_chars in range(low, max(-1, low - 128), -1):
        calibration = ""
        current_ids = token_ids(compose(body_chars, calibration))
        for _ in range(512):
            if len(current_ids) == target_tokens:
                return compose(body_chars, calibration), current_ids
            candidates: list[tuple[int, str, list[int]]] = []
            for atom in calibration_atoms:
                candidate = calibration + atom
                candidate_ids = token_ids(compose(body_chars, candidate))
                increment = len(candidate_ids) - len(current_ids)
                if 0 < increment <= target_tokens - len(current_ids):
                    candidates.append((increment, candidate, candidate_ids))
            if not candidates:
                break
            _, calibration, current_ids = max(candidates, key=lambda item: item[0])
    raise ValueError(f"could not fit prompt to exactly {target_tokens} tokens")


def private_record_header() -> str:
    return (
        "下面是一份课程长文档。正文来自古典小说《红楼梦》，其中穿插了以"
        "【课程私有档案】标记的虚构记录。回答时只依据这些私有档案；正文中的"
        "人物、数字和情节都是背景或干扰项。不要猜测，不要补充解释。"
    )


def deterministic_token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randrange(100000, 999999)}"


def retrieval_semantics(
    rng: random.Random,
    position: float,
    tag: str,
) -> tuple[str, list[tuple[float, str]], str, str]:
    answer = deterministic_token(rng, f"KEY-{tag}")
    records = [
        (
            position,
            f"【课程私有档案】唯一目标字段 TARGET_KEY 的值为 {answer}。",
        )
    ]
    question = "问题：TARGET_KEY 的值是什么？只输出完整字段值。"
    return private_record_header(), records, question, answer


def multihop_semantics(
    rng: random.Random,
) -> tuple[str, list[tuple[float, str]], str, str]:
    answer = deterministic_token(rng, "MH")
    records = [
        (0.12, "【课程私有档案】青砚站的下一节点是暮港站。"),
        (0.46, "【课程私有档案】松岚站的下一节点是镜湖站。"),
        (0.71, f"【课程私有档案】暮港站当前有效的验证令牌是 {answer}。"),
        (0.88, "【课程私有档案】镜湖站当前有效的验证令牌是 DISTRACTOR-7731。"),
    ]
    question = (
        "问题：从青砚站沿“下一节点”走一步，再读取该节点当前有效的验证令牌。"
        "只输出令牌。"
    )
    return private_record_header(), records, question, answer


def revision_semantics(
    rng: random.Random,
) -> tuple[str, list[tuple[float, str]], str, str]:
    old = deterministic_token(rng, "OLD")
    revoked = deterministic_token(rng, "REVOKED")
    answer = deterministic_token(rng, "REV")
    records = [
        (0.14, f"【课程私有档案】配置 ORCHID，版本 1，值 {old}，状态：有效。"),
        (0.43, f"【课程私有档案】配置 ORCHID，版本 2，值 {revoked}，状态：有效。"),
        (0.67, "【课程私有档案】配置 ORCHID，版本 2，状态变更：已撤销。"),
        (0.86, f"【课程私有档案】配置 ORCHID，版本 3，值 {answer}，状态：有效。"),
    ]
    question = "问题：按版本和状态处理全部记录后，ORCHID 最终有效值是什么？只输出值。"
    return private_record_header(), records, question, answer


def aggregate_semantics(
    rng: random.Random,
    record_count: int = 8,
) -> tuple[str, list[tuple[float, str]], str, str]:
    if record_count < 2:
        raise ValueError("aggregate task requires at least two records")
    values = [rng.randrange(3, 20) for _ in range(record_count)]
    records = [
        (
            0.08 + index * 0.84 / (record_count - 1),
            f"【课程私有档案】批次 B{index + 1} 的合格件数为 {value}。",
        )
        for index, value in enumerate(values)
    ]
    question = (
        f"问题：B1 到 B{record_count} 的合格件数总和是多少？只输出十进制整数。"
    )
    return private_record_header(), records, question, str(sum(values))


def distractor_semantics(
    rng: random.Random,
) -> tuple[str, list[tuple[float, str]], str, str]:
    answer = deterministic_token(rng, "VALID")
    records = [
        (0.18, "【课程私有档案】示例文本：TARGET_KEY=EXAMPLE-000000，此行为格式样例，不是记录。"),
        (0.51, f"【课程私有档案】命名空间 COURSE、字段 TARGET_KEY 的有效值为 {answer}。"),
        (0.82, "【课程私有档案】命名空间 ARCHIVE、字段 TARGET_KEY 的值为 WRONG-8142。"),
    ]
    question = "问题：命名空间 COURSE 中 TARGET_KEY 的有效值是什么？只输出值。"
    return private_record_header(), records, question, answer


def make_case(
    *,
    tokenizer,
    corpus: str,
    rng: random.Random,
    case_id: str,
    split: str,
    suite: str,
    category: str,
    target_tokens: int,
    semantics: tuple[str, list[tuple[float, str]], str, str | None],
    max_new_tokens: int,
    ignore_eos: bool,
    validation: str,
    include_prompt: bool,
    roles: Sequence[str] = (),
) -> dict[str, Any]:
    header, records, question, expected = semantics
    corpus_offset = rng.randrange(0, len(corpus))
    prompt, input_ids = fit_exact_prompt(
        lambda value: chat_token_ids(tokenizer, value),
        target_tokens,
        corpus,
        corpus_offset,
        header,
        records,
        question,
    )
    case: dict[str, Any] = {
        "schema": CASE_SCHEMA,
        "id": case_id,
        "split": split,
        "suite": suite,
        "roles": list(roles),
        "category": category,
        "target_prompt_tokens": target_tokens,
        "actual_prompt_tokens": len(input_ids),
        "input_ids": input_ids,
        "input_ids_sha256": sha256_bytes(canonical_json(input_ids)),
        "expected": expected,
        "validation": validation,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": ignore_eos,
        "temperature": 0.0,
    }
    if include_prompt:
        case["prompt"] = prompt
    case["case_sha256"] = sha256_bytes(canonical_json(case))
    return case


def public_cases(
    contract: dict[str, Any],
    tokenizer,
    corpus: str,
    rng: random.Random,
    include_prompt: bool,
) -> list[dict[str, Any]]:
    workload = contract["correctness_workload"]
    output_budget = int(workload["public_functional_suite"]["output_tokens_per_case"])
    cases: list[dict[str, Any]] = []
    for position, tag in ((0.10, "P10"), (0.50, "P50"), (0.90, "P90")):
        cases.append(
            make_case(
                tokenizer=tokenizer,
                corpus=corpus,
                rng=rng,
                case_id=f"text-niah-1024-{tag.lower()}",
                split="public",
                suite="functional",
                category="exact_retrieval",
                target_tokens=1024,
                semantics=retrieval_semantics(rng, position, tag),
                max_new_tokens=output_budget,
                ignore_eos=False,
                validation="normalized_exact",
                include_prompt=include_prompt,
            )
        )
    for case_id, category, factory in (
        ("longdoc-multihop-8192", "multi_hop", multihop_semantics),
        ("longdoc-revision-8192", "revision_resolution", revision_semantics),
        (
            "longdoc-aggregate-8192",
            "aggregate",
            lambda value: aggregate_semantics(value, record_count=2),
        ),
    ):
        cases.append(
            make_case(
                tokenizer=tokenizer,
                corpus=corpus,
                rng=rng,
                case_id=case_id,
                split="public",
                suite="functional",
                category=category,
                target_tokens=8192,
                semantics=factory(rng),
                max_new_tokens=output_budget,
                ignore_eos=False,
                validation="normalized_exact",
                include_prompt=include_prompt,
            )
        )

    trajectory_ids = set(workload["public_token_trajectory_suite"]["case_ids"])
    performance_cells = {
        item["id"]: item
        for item in contract["performance_scoring"]["ttft_cells"]
    }
    for case_id, definition in performance_cells.items():
        semantics = (
            "下面是《红楼梦》长文本性能负载。阅读后按要求继续输出。",
            [(0.50, "【课程私有档案】性能标记 APXINF-QWEN38-4090。")],
            "任务：先输出 APXINF-QWEN38-4090，然后持续输出词语 benchmark。",
            None,
        )
        roles = ["performance"]
        if case_id in trajectory_ids:
            roles.append("public_trajectory")
        cases.append(
            make_case(
                tokenizer=tokenizer,
                corpus=corpus,
                rng=rng,
                case_id=case_id,
                split="public",
                suite="performance",
                category="fixed_decode_budget",
                target_tokens=int(definition["prompt_tokens"]),
                semantics=semantics,
                max_new_tokens=int(definition["output_tokens"]),
                ignore_eos=True,
                validation="nonempty",
                include_prompt=include_prompt,
                roles=roles,
            )
        )

    multi_header, multi_records, multi_question, multi_expected = retrieval_semantics(
        rng, 0.50, "MULTI"
    )
    cases.append(
        make_case(
            tokenizer=tokenizer,
            corpus=corpus,
            rng=rng,
            case_id="multi-functional-1024",
            split="public",
            suite="multi_request",
            category="exact_retrieval",
            target_tokens=1024,
            semantics=(
                multi_header,
                multi_records,
                multi_question
                + f" 输出必须以 {multi_expected} 开头，随后持续输出词语 multi 直到达到输出预算。",
                multi_expected,
            ),
            max_new_tokens=128,
            ignore_eos=True,
            validation="normalized_prefix",
            include_prompt=include_prompt,
            roles=("multi",),
        )
    )
    return cases


def context_cases(
    contract: dict[str, Any],
    tokenizer,
    corpus: str,
    rng: random.Random,
    include_prompt: bool,
    lengths: Sequence[int],
) -> list[dict[str, Any]]:
    bonus = contract["context_bonus"]
    output_budget = int(bonus["required_output_tokens"])
    cases: list[dict[str, Any]] = []
    factories: list[tuple[str, str, Callable[[], tuple[str, list[tuple[float, str]], str, str]]]] = [
        ("retrieval-early", "retrieval_early", lambda: retrieval_semantics(rng, 0.10, "EARLY")),
        ("retrieval-middle", "retrieval_middle", lambda: retrieval_semantics(rng, 0.50, "MIDDLE")),
        ("retrieval-late", "retrieval_late", lambda: retrieval_semantics(rng, 0.90, "LATE")),
        ("multihop", "multi_hop", lambda: multihop_semantics(rng)),
        ("revision", "revision_resolution", lambda: revision_semantics(rng)),
        ("aggregate", "aggregate", lambda: aggregate_semantics(rng)),
    ]
    for target_tokens in lengths:
        for suffix, category, factory in factories:
            header, records, question, expected = factory()
            semantics = (
                header,
                records,
                question
                + f" 输出必须以 {expected} 开头，随后持续输出词语 context 直到达到输出预算。",
                expected,
            )
            cases.append(
                make_case(
                    tokenizer=tokenizer,
                    corpus=corpus,
                    rng=rng,
                    case_id=f"context-{target_tokens}-{suffix}",
                    split="public",
                    suite="context",
                    category=category,
                    target_tokens=target_tokens,
                    semantics=semantics,
                    max_new_tokens=output_budget,
                    ignore_eos=True,
                    validation="normalized_prefix",
                    include_prompt=include_prompt,
                )
            )
    return cases


def qualitative_cases(
    tokenizer,
    corpus: str,
    rng: random.Random,
    include_prompt: bool,
) -> list[dict[str, Any]]:
    semantics = (
        "下面是《红楼梦》原文节选。这是定性演示任务，不进入机器 correctness 分数。",
        [],
        (
            "请分析节选中贾宝玉的性格矛盾。给出一个中心判断，并引用至少两处原文"
            "作为证据；明确区分原文事实和你的解释。"
        ),
        None,
    )
    return [
        make_case(
            tokenizer=tokenizer,
            corpus=corpus,
            rng=rng,
            case_id="qualitative-hlm-analysis-8192",
            split="public",
            suite="qualitative",
            category="literary_analysis_unscored",
            target_tokens=8192,
            semantics=semantics,
            max_new_tokens=512,
            ignore_eos=False,
            validation="qualitative_unscored",
            include_prompt=include_prompt,
        )
    ]


def tokenizer_fingerprint(model_dir: Path) -> dict[str, Any]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    )
    files = {
        name: sha256_file(model_dir / name)
        for name in names
        if (model_dir / name).is_file()
    }
    if not files:
        raise ValueError(f"{model_dir}: no tokenizer/config files found")
    return {
        "files": files,
        "combined_sha256": sha256_bytes(canonical_json(files)),
    }


def write_dataset(
    output_dir: Path,
    cases: Sequence[dict[str, Any]],
    *,
    contract_path: Path,
    contract: dict[str, Any],
    corpus_path: Path,
    model_dir: Path,
    split: str,
    seed: int,
    suite: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    rendered = b"".join(canonical_json(case) + b"\n" for case in cases)
    cases_path.write_bytes(rendered)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "contract_schema": contract["schema"],
        "contract_sha256": sha256_file(contract_path),
        "model_repo_id": contract["model"]["repo_id"],
        "model_revision": contract["model"]["revision"],
        "tokenizer": tokenizer_fingerprint(model_dir),
        "corpus": {
            "id": contract["data_generation"]["corpus"]["id"],
            "sha256": sha256_file(corpus_path),
        },
        "split": split,
        "suite": suite,
        "seed_sha256": sha256_bytes(str(seed).encode("ascii")),
        "case_count": len(cases),
        "case_ids": [case["id"] for case in cases],
        "case_hashes": {case["id"]: case["case_sha256"] for case in cases},
        "cases_jsonl": cases_path.name,
        "cases_jsonl_sha256": sha256_bytes(rendered),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def parse_lengths(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers") from error
    if not result or any(length <= 0 for length in result):
        raise argparse.ArgumentTypeError("lengths must contain positive integers")
    return result


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=here / "contract-v1.json")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=here / ".cache" / "pg24264.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("public", "context", "qualitative"),
        default="public",
    )
    parser.add_argument("--lengths", type=parse_lengths)
    parser.add_argument("--include-prompt", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_json(args.contract)
    corpus_spec = contract["data_generation"]["corpus"]
    corpus = extract_corpus(args.corpus, corpus_spec["sha256"])
    tokenizer = load_tokenizer(args.model_dir)
    seed = int(contract["data_generation"]["public_seed"])
    split = "public"
    rng = random.Random(seed)

    if args.suite == "public":
        cases = public_cases(contract, tokenizer, corpus, rng, args.include_prompt)
    elif args.suite == "context":
        lengths = args.lengths or [
            int(value) for value in contract["context_bonus"]["context_staircase_prompt_tokens"]
        ]
        maximum = int(contract["context_bonus"]["full_bonus_prompt_tokens"])
        if any(length > maximum for length in lengths):
            raise ValueError(f"context length exceeds scoring ceiling {maximum}")
        cases = context_cases(
            contract,
            tokenizer,
            corpus,
            rng,
            args.include_prompt,
            lengths,
        )
    elif args.suite == "qualitative":
        cases = qualitative_cases(tokenizer, corpus, rng, args.include_prompt)

    manifest = write_dataset(
        args.output_dir,
        cases,
        contract_path=args.contract,
        contract=contract,
        corpus_path=args.corpus,
        model_dir=args.model_dir,
        split=split,
        seed=seed,
        suite=args.suite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
