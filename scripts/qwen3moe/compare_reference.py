#!/usr/bin/env python3
"""Compare an ApxInf logits dump against the torch-CPU Qwen3-MoE reference.

Consumes the two artifacts produced by the pair

    qwen3moe_awq_reference.py --out DIR --case NAME      -> DIR/NAME.npz
    cargo run --example qwen3moe_verify -- --out OUT     -> OUT.json + OUT.bin

and answers the only question that matters when the two continuations differ:
is this floating-point noise around a near-tie, or a real kernel bug?

The tell is the *margin* at the diverging step. If the reference's top-1 and
top-2 logits differ by less than the observed logit error, BF16 accumulation
alone explains the flip and the continuation is not evidence of a bug. If the
margin is comfortably larger than the error, something is wrong upstream, and
the per-step error trend says whether it compounds (a decode/KV problem) or is
already present at the prefill (a weight, layout or attention problem).

Usage:
  python compare_reference.py --reference /opt/data/dev/ref/raw0.npz \
      --apxinf /opt/data/dev/ref/raw0.apxinf.json [--model DIR] [--top 5]
"""
import argparse
import json
import os

import numpy as np


def load_apxinf(meta_path):
    meta = json.load(open(meta_path))
    bin_path = os.path.join(os.path.dirname(os.path.abspath(meta_path)), meta["logits"])
    rows, vocab = meta["rows"], meta["vocab_size"]
    logits = np.fromfile(bin_path, dtype="<f4")
    if logits.size != rows * vocab:
        raise SystemExit(f"{bin_path}: {logits.size} floats, expected {rows} x {vocab}")
    return meta, logits.reshape(rows, vocab)


def decoder(model_dir):
    """Return a token -> text function, or a repr fallback without a tokenizer."""
    if model_dir:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_dir)
            return lambda t: repr(tok.decode([int(t)]))
        except Exception as error:  # tokenizer is a convenience, not a requirement
            print(f"(no tokenizer: {error})")
    return lambda t: f"<{int(t)}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="<case>.npz from the torch reference")
    ap.add_argument("--apxinf", required=True, help="<out>.json from the qwen3moe_verify example")
    ap.add_argument("--model", default=None, help="checkpoint dir, only to decode token ids")
    ap.add_argument("--top", type=int, default=5, help="top-k overlap width")
    ap.add_argument("--strict", action="store_true", help="require complete teacher-forced finite dumps and bounded error")
    ap.add_argument("--max-error", type=float, default=1.1, help="strict absolute error ceiling (baseline max 1.00)")
    ap.add_argument("--allow-tie-step", type=int, action="append", default=[], help="documented reference near-tie exempt from the margin gate")
    args = ap.parse_args()

    ref = np.load(args.reference)
    meta, mine = load_apxinf(args.apxinf)
    show = decoder(args.model)

    ref_ids = ref["token_ids"].astype(np.int64)
    my_ids = np.array(meta["token_ids"], dtype=np.int64)
    if not np.array_equal(ref_ids, my_ids):
        raise SystemExit(
            f"prompt mismatch: reference {ref_ids.tolist()} vs apxinf {my_ids.tolist()}"
        )

    ref_steps = ref["step_logits"].astype(np.float32)  # [steps, vocab]
    ref_greedy = ref["greedy_tokens"].astype(np.int64)
    my_greedy = np.array(meta["greedy_tokens"], dtype=np.int64)
    forced = bool(meta.get("teacher_forced", False))
    steps = min(len(ref_steps), len(mine), len(ref_greedy), len(my_greedy))
    if args.strict:
        if not forced or steps == 0 or steps != len(ref_steps) or steps != len(ref_greedy):
            raise SystemExit("acceptance requires all reference steps under teacher forcing")
        if meta.get("reference_greedy_tokens") != ref_greedy.tolist():
            raise SystemExit("teacher-forcing tokens differ from reference")
        if mine.shape[1] != ref_steps.shape[1] or not np.isfinite(mine).all() or not np.isfinite(ref_steps).all():
            raise SystemExit("invalid vocabulary shape or non-finite logits")
        if not np.array_equal(np.argmax(mine[:steps], axis=1), my_greedy[:steps]):
            raise SystemExit("recorded greedy tokens disagree with logits")
    if not forced:
        # Free-running, the two runs decode different sentences the moment they
        # disagree, so only the steps up to the first divergence share a context.
        agree = int(np.argmax(ref_greedy[:steps] != my_greedy[:steps])) \
            if (ref_greedy[:steps] != my_greedy[:steps]).any() else steps
        print("WARNING: dump is free-running; only the first "
              f"{agree + 1} rows share a context with the reference. "
              "Re-run the example without --free-running for a clean comparison.")

    print(f"prompt: {len(ref_ids)} tokens, vocab {mine.shape[1]}, comparing {steps} steps"
          f" ({'teacher-forced' if forced else 'free-running'})")
    print(f"reference greedy: {ref_greedy[:steps].tolist()}")
    print(f"apxinf    greedy: {my_greedy[:steps].tolist()}")
    print()
    header = f"{'step':>4} {'ref':>7} {'apx':>7} {'ok':>3} {'max|d|':>9} {'mean|d|':>9} " \
             f"{'corr':>7} {'top%d' % args.top:>6} {'margin':>9} {'verdict':>10}"
    print(header)
    print("-" * len(header))

    first_bad = None
    for s in range(steps):
        r, m = ref_steps[s], mine[s]
        # Logits are only defined up to a per-row shift for argmax purposes,
        # but the runtime does not shift them, so compare them directly.
        d = np.abs(r - m)
        max_d, mean_d = float(d.max()), float(d.mean())
        corr = float(np.corrcoef(r, m)[0, 1])
        rt = set(np.argpartition(-r, args.top)[: args.top].tolist())
        mt = set(np.argpartition(-m, args.top)[: args.top].tolist())
        overlap = len(rt & mt)
        # Reference margin between its own top-1 and top-2: how much error the
        # step can absorb before the argmax flips.
        top2 = np.partition(-r, 1)[:2]
        margin = float(-top2[0] - (-top2[1]))
        agree = int(ref_greedy[s]) == int(my_greedy[s])
        if args.strict and s not in args.allow_tie_step and (not agree or max_d >= margin):
            first_bad = s if first_bad is None else first_bad
        if agree:
            verdict = "match"
        elif margin < max_d:
            verdict = "near-tie"
        else:
            verdict = "SUSPECT"
            if first_bad is None:
                first_bad = s
        print(
            f"{s:>4} {int(ref_greedy[s]):>7} {int(my_greedy[s]):>7} {'y' if agree else 'n':>3} "
            f"{max_d:>9.4f} {mean_d:>9.4f} {corr:>7.5f} {overlap:>4}/{args.top} "
            f"{margin:>9.4f} {verdict:>10}"
        )

    print()
    for s in range(steps):
        if int(ref_greedy[s]) != int(my_greedy[s]):
            r, m = ref_steps[s], mine[s]
            print(f"step {s} detail — reference top {args.top}:")
            for t in np.argsort(-r)[: args.top]:
                print(f"  {int(t):>7} {show(t):<20} ref {r[t]:>9.4f}   apx {m[t]:>9.4f}")
            print(f"step {s} detail — apxinf top {args.top}:")
            for t in np.argsort(-m)[: args.top]:
                print(f"  {int(t):>7} {show(t):<20} ref {r[t]:>9.4f}   apx {m[t]:>9.4f}")
            break
    else:
        print("no divergence in the compared steps")

    # Error growth across steps separates a decode/KV defect (error rises with
    # the step index) from a prefill defect (error is already large at step 0).
    # Only meaningful under teacher forcing, where every step shares a context.
    errs = [float(np.abs(ref_steps[s] - mine[s]).max()) for s in range(steps)]
    print()
    print("max|d| by step:", " ".join(f"{e:.3f}" for e in errs))
    if not forced:
        print("-> free-running dump: the trend past the first divergence is noise, not signal")
    elif errs[0] > 0 and errs[-1] / errs[0] > 3.0:
        print("-> error grows with the step index; suspect the decode path or the KV cache")
    else:
        print("-> error is flat across steps; whatever is off is already off at the prefill")

    if first_bad is not None:
        raise SystemExit(f"divergence at step {first_bad} exceeds the reference margin")
    if args.strict and (max(errs) > args.max_error or errs[-1] > 3 * max(errs[0], 1e-6)):
        raise SystemExit("logit error exceeds the baseline envelope or grows across steps")


if __name__ == "__main__":
    main()
