#!/usr/bin/env python3
"""Validate Codex review evidence and render one stable GitHub comment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote


HUNK = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)
PRIORITIES = {"P0", "P1", "P2"}
SIDES = {"LEFT", "RIGHT"}
VERDICTS = {"BLOCK", "NEEDS_WORK", "LGTM"}
REVIEW_STATUSES = {"COMPLETE", "INCOMPLETE"}
MARKER = "<!-- apxinf-ai-review -->"


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def clean_markdown(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("@", "@\u200b")
    )


def changed_line(
    merge_base: str,
    head: str,
    paths: list[str],
    side: str,
    line: int,
) -> bool:
    diff = git(
        "diff",
        "--unified=0",
        "--no-color",
        "--no-ext-diff",
        "-M",
        merge_base,
        head,
        "--",
        *paths,
    )
    for raw_line in diff.splitlines():
        match = HUNK.match(raw_line)
        if not match:
            continue
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        new_start = int(match.group(3))
        new_count = int(match.group(4) or "1")
        if side == "LEFT" and old_count and old_start <= line < old_start + old_count:
            return True
        if side == "RIGHT" and new_count and new_start <= line < new_start + new_count:
            return True
    return False


def expected_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "repository": context["repository"],
        "pr_number": context["pr_number"],
        "base_sha": context["base_sha"],
        "head_sha": context["head_sha"],
        "merge_base_sha": context["merge_base_sha"],
        "changed_files": context["changed_files"],
        "additions": context["additions"],
        "deletions": context["deletions"],
    }


def validate(context: dict[str, Any], review: dict[str, Any]) -> None:
    required_top = {
        "snapshot",
        "review_status",
        "verdict",
        "summary",
        "overall_impact",
        "coverage_notes",
        "findings",
    }
    require(set(review) == required_top, "review has unexpected or missing top-level keys")
    require(review["snapshot"] == expected_snapshot(context), "snapshot does not match")
    require(review["review_status"] in REVIEW_STATUSES, "invalid review status")
    require(review["verdict"] in VERDICTS, "invalid verdict")
    require(isinstance(review["summary"], str) and review["summary"].strip(), "empty summary")
    require(
        isinstance(review["overall_impact"], str) and review["overall_impact"].strip(),
        "empty overall impact",
    )
    require(isinstance(review["coverage_notes"], list), "coverage_notes must be a list")
    require(isinstance(review["findings"], list), "findings must be a list")
    require(len(review["findings"]) <= 20, "too many findings")

    by_right_path = {changed["path"]: changed for changed in context["files"]}
    by_left_path = {
        (changed["previous_path"] or changed["path"]): changed
        for changed in context["files"]
    }
    seen: set[tuple[str, int, str, str]] = set()

    for index, finding in enumerate(review["findings"], start=1):
        prefix = f"finding {index}"
        required_finding = {
            "priority",
            "title",
            "body",
            "path",
            "line",
            "side",
            "trigger",
            "impact",
            "suggestion",
        }
        require(set(finding) == required_finding, f"{prefix} has unexpected or missing keys")
        require(finding["priority"] in PRIORITIES, f"{prefix} has invalid priority")
        require(finding["side"] in SIDES, f"{prefix} has invalid side")
        require(isinstance(finding["line"], int) and finding["line"] > 0, f"{prefix} has invalid line")
        for field in ("title", "body", "path", "trigger", "impact", "suggestion"):
            require(
                isinstance(finding[field], str) and finding[field].strip(),
                f"{prefix} has an empty {field}",
            )

        changed = (
            by_left_path.get(finding["path"])
            if finding["side"] == "LEFT"
            else by_right_path.get(finding["path"])
        )
        require(changed is not None, f"{prefix} path/side is not in the comparison")
        paths = [changed["path"]]
        if changed["previous_path"] and changed["previous_path"] not in paths:
            paths.insert(0, changed["previous_path"])
        require(
            changed_line(
                context["merge_base_sha"],
                context["head_sha"],
                paths,
                finding["side"],
                finding["line"],
            ),
            f"{prefix} is not anchored to a changed line",
        )

        identity = (
            finding["path"],
            finding["line"],
            finding["side"],
            finding["title"].casefold(),
        )
        require(identity not in seen, f"{prefix} duplicates another finding")
        seen.add(identity)

    if review["review_status"] == "INCOMPLETE":
        require(not review["findings"], "an incomplete review cannot report findings")
        require(review["verdict"] == "NEEDS_WORK", "an incomplete review must be NEEDS_WORK")
        require(review["coverage_notes"], "an incomplete review needs coverage notes")
    elif review["findings"]:
        require(review["verdict"] != "LGTM", "LGTM cannot contain findings")
    else:
        require(review["verdict"] == "LGTM", "a finding-free review must be LGTM")


def render(context: dict[str, Any], review: dict[str, Any]) -> str:
    verdict_badge = {
        "BLOCK": "🔴 BLOCK",
        "NEEDS_WORK": "⚠️ NEEDS WORK",
        "LGTM": "✅ LGTM",
    }[review["verdict"]]
    short_head = context["head_sha"][:12]
    lines = [
        MARKER,
        "## ApxInf automated code review",
        "",
        f"**Verdict:** {verdict_badge}  ",
        f"**Review status:** `{review['review_status']}`  ",
        f"**Reviewed head:** `{short_head}`  ",
        (
            f"**Comparison:** `{context['merge_base_sha'][:12]}..{short_head}` · "
            f"{context['changed_files']} files · +{context['additions']} / -{context['deletions']}"
        ),
        "",
        "### Summary",
        "",
        clean_markdown(review["summary"]),
        "",
        "### Overall impact",
        "",
        clean_markdown(review["overall_impact"]),
    ]

    if review["findings"]:
        lines.extend(["", "### Findings"])
        for finding in review["findings"]:
            revision = (
                context["merge_base_sha"]
                if finding["side"] == "LEFT"
                else context["head_sha"]
            )
            encoded_path = quote(finding["path"], safe="/")
            source = (
                f"https://github.com/{context['repository']}/blob/{revision}/"
                f"{encoded_path}#L{finding['line']}"
            )
            lines.extend(
                [
                    "",
                    f"#### [{finding['priority']}] {clean_markdown(finding['title'])}",
                    "",
                    f"[`{clean_markdown(finding['path'])}:{finding['line']}`]({source})",
                    "",
                    clean_markdown(finding["body"]),
                    "",
                    f"- **Trigger:** {clean_markdown(finding['trigger'])}",
                    f"- **Impact:** {clean_markdown(finding['impact'])}",
                    f"- **Direction:** {clean_markdown(finding['suggestion'])}",
                ]
            )
    else:
        lines.extend(["", "### Findings", "", "No actionable P0-P2 defects found."])

    if review["coverage_notes"]:
        lines.extend(["", "### Coverage notes", ""])
        lines.extend(f"- {clean_markdown(note)}" for note in review["coverage_notes"])

    lines.extend(
        [
            "",
            "---",
            "Generated by the repository-pinned `apxinf-code-review` skill. "
            "This is review input, not an approval or merge decision.",
            "",
        ]
    )
    body = "\n".join(lines)
    require(len(body.encode("utf-8")) <= 60_000, "rendered review is too large")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()

    context = json.loads(args.context.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    validate(context, review)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render(context, review), encoding="utf-8")


if __name__ == "__main__":
    main()
