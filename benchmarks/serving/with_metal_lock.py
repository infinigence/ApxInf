#!/usr/bin/env python3
"""Run one command while this process owns an existing Metal measurement lock."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--wait-seconds", type=float, default=900)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.wait_seconds < 0:
        parser.error("Supply a command and a non-negative wait limit.")
    descriptor = os.open(args.lock, os.O_RDONLY)
    deadline = time.monotonic() + args.wait_seconds
    child = None
    try:
        print("Waiting for the existing Metal measurement lock.", flush=True)
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SystemExit("The Metal lock wait expired. No command started.")
                time.sleep(0.25)
        print("Metal measurement lock acquired.", flush=True)
        child = subprocess.Popen(command, start_new_session=True)
        print(f"Child process: {child.pid}", flush=True)
        return child.wait()
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        return 130
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    sys.exit(main())
