#!/usr/bin/env python3
"""Run quick checks before Claude invokes git commit; exit 2 blocks the tool."""

import json
from pathlib import Path
import shlex
import subprocess
import sys


def may_commit(command):
    """Conservatively match git commit, including global flags and shell chains.

    This is a convenience hook, not a shell security boundary. The Git hook
    remains the common enforcement point for aliases and other clients.
    """
    try:
        tokens = list(shlex.shlex(command, posix=True, punctuation_chars=True))
    except ValueError:
        return True
    git_seen = False
    for token in tokens:
        if token in (";", "&&", "||", "|", "&", "(", ")"):
            git_seen = False
        elif Path(token).name == "git":
            git_seen = True
        elif git_seen and token == "commit":
            return True
    return False


def main():
    try:
        event = json.load(sys.stdin)
        if event.get("tool_name") != "Bash":
            return 0
        if not may_commit(event["tool_input"]["command"]):
            return 0
        result = subprocess.run(
            ["just", "pre-commit"],
            cwd=Path(__file__).resolve().parents[2],
            # Hook stdout is a control channel; send check output to stderr.
            stdout=sys.stderr,
            stderr=sys.stderr,
        )
        if result.returncode:
            print("Commit blocked: fix just pre-commit before retrying.", file=sys.stderr)
            return 2
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Commit guard failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
