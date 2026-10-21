#!/usr/bin/env python3
"""Opt-in acceptance check at Stop; one corrective turn, never an endless loop."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_acceptance import inspect


def main():
    plan = os.environ.get("APXINF_AGENT_ACCEPTANCE_PLAN")
    if not plan:
        return 0
    try:
        event = json.load(sys.stdin)
        failures, blocked = inspect(Path(plan).resolve())
    except Exception as error:
        event = locals().get("event", {})
        failures, blocked = [str(error)], []
    if not failures and not blocked:
        return 0
    reason = "Acceptance INCOMPLETE:\n" + "\n".join(failures + blocked)
    if failures and not event.get("stop_hook_active"):
        print(json.dumps({"decision": "block", "reason": reason +
              "\nFix the failed/missing checks or record each concrete blocker with "
              "just acceptance block. Report blocked items as unverified, not complete. "
              "Do not retry a prohibited board operation."}))
    else:
        # A bounded stop is not success. No pass marker is written; the CLI
        # still exits nonzero until every required case actually passes.
        print(reason + "\nStopping with unresolved acceptance; not a PASS.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
