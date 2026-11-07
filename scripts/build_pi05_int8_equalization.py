#!/usr/bin/env python3
"""Build per-channel equalization vectors from the row-level diagnostic.

Reads `row_levels.jsonl` (produced by `APXINF_PI05_INT8_STATS` runs, one record
per `report_row_levels` call with a per-channel `channel_max`), aggregates each
site's channel amax across every call, and emits the SmoothQuant-style
equalization vector `c_k = max(amax_k, floor)**alpha` as JSON:

    { "<site name>": [c_0, c_1, ...], ... }

The model folds `c` into the INT8 weight (column k scales by c_k) and `1/c`
into the activation (see `musa_pi05/weights/int8_dynamic.rs`'s
`APXINF_INT8_EQUALIZATION` knob), which leaves the math unchanged and moves
quantization difficulty from the activation's per-row amax onto the weight's
per-output-channel amax.

The alpha selection checked in as
`configs/pi05_int8_equalization_q50_g60_d60.json` (qkv=0.5 / gate_up=0.6 /
down=0.6) is the one the MUSA INT8 route was measured with;
`doc/musa-backend.md` carries the latency and cosine of the runs it belongs to.
Regenerate an equivalent table with

    python3 scripts/build_pi05_int8_equalization.py \
        row_levels.jsonl out.json \
        --alpha 0.5 --site-alpha lang.gate_up_in=0.6 --site-alpha lang.down_in=0.6

after collecting `row_levels.jsonl` from a
`APXINF_PI05_INT8_STATS=1 APXINF_PI05_INT8_CHANNELS=<path>` bench run.

Provenance of the checked-in table: collected on a `--model-variant
int8_dynamic` bench of the PI0.5 LIBERO checkpoint, aggregating every call's
per-channel maximum, with `--floor-ratio` left at its default (1e-4). That
aggregate is a property of the *schedule*, not of the checkpoint alone: a
different flow-step count, batch or view count changes which rows are seen, so
the table has to be rebuilt when the schedule changes.

One recorded site has no consumer: `lang.o_proj_in` is collected like the
others, but `equalization_site()` in `musa_pi05/weights/int8_dynamic.rs` folds
only `lang.qkv_in`, `lang.gate_up_in` and `lang.down_in` -- the attention output
is a convex combination that already quantizes cleanly -- so the checked-in file
carries that vector as a collection byproduct.
"""

import argparse
import json
import math
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="row_levels.jsonl path")
    parser.add_argument("output", help="equalization JSON output path")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="difficulty shift exponent (0 = off, 1 = all on the weight)")
    parser.add_argument("--site-alpha", action="append", default=[],
                        metavar="SITE=ALPHA",
                        help="per-site alpha override, e.g. --site-alpha lang.down_in=0.65")
    parser.add_argument("--floor-ratio", type=float, default=1.0e-4,
                        help="per-site amax floor, as a fraction of that site's max")
    args = parser.parse_args()

    site_alpha = {}
    for override in args.site_alpha:
        name, _, value = override.partition("=")
        site_alpha[name] = float(value)

    amax = {}
    with open(args.input) as handle:
        for line in handle:
            record = json.loads(line)
            name = record["name"]
            channels = record["channel_max"]
            if name in amax:
                current = amax[name]
                assert len(current) == len(channels), f"{name}: width changed"
                amax[name] = [max(a, b) for a, b in zip(current, channels)]
            else:
                amax[name] = list(channels)

    table = {}
    for name, channels in sorted(amax.items()):
        alpha = site_alpha.get(name, args.alpha)
        peak = max(channels)
        floor = peak * args.floor_ratio
        vector = [max(value, floor) ** alpha for value in channels]
        table[name] = vector

        ordered = sorted(channels)
        median = ordered[len(ordered) // 2]
        # The same `max(value, floor) ** alpha` the vector above carries, so the
        # printed levels are what the model gets rather than an idealized alpha:
        # a channel the floor lifts is scaled by `floor ** alpha`, not by its
        # own amax.
        median_factor = max(median, floor) ** alpha
        peak_factor = max(peak, floor) ** alpha
        ratio_before = peak / max(median, 1.0e-30)
        ratio_after = (peak / peak_factor) / max(median / median_factor, 1.0e-30)
        # A row whose typical element is `median` keeps about
        # 127 * median / amax levels; equalization divides both by their c_k.
        levels_before = 127.0 * median / peak
        levels_after = 127.0 * (median / median_factor) / (peak / peak_factor)
        print(f"{name}: cols={len(channels)} top={peak:.3f} median={median:.3f} "
              f"alpha={alpha} top/median {ratio_before:.1f}x -> {ratio_after:.1f}x, "
              f"typical levels {levels_before:.2f} -> {levels_after:.2f}")

    with open(args.output, "w") as handle:
        json.dump(table, handle)
    print(f"wrote {args.output} (alpha={args.alpha}, floor_ratio={args.floor_ratio})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
