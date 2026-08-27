#!/usr/bin/env python3
import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def merged_duration(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)

    commits = connection.execute(
        """
        SELECT k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.shortName
        WHERE s.value = 'argmax_pair_final_kernel'
        ORDER BY k.start
        """
    ).fetchall()
    if len(commits) < 7:
        raise RuntimeError(f"expected at least 7 token commits, got {len(commits)}")
    commits = commits[-7:]

    windows = []
    kernel_totals = defaultdict(lambda: [0, 0])
    runtime_totals = defaultdict(lambda: [0, 0])
    for index in range(1, len(commits)):
        start = commits[index - 1][1]
        end = commits[index][1]
        kernels = connection.execute(
            """
            SELECT k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.shortName
            WHERE k.start >= ? AND k.end <= ?
            ORDER BY k.start
            """,
            (start, end),
        ).fetchall()
        runtimes = connection.execute(
            """
            SELECT r.start, r.end, s.value
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s ON s.id = r.nameId
            WHERE r.start >= ? AND r.end <= ?
            ORDER BY r.start
            """,
            (start, end),
        ).fetchall()
        for item_start, item_end, name in kernels:
            kernel_totals[name][0] += item_end - item_start
            kernel_totals[name][1] += 1
        for item_start, item_end, name in runtimes:
            runtime_totals[name][0] += item_end - item_start
            runtime_totals[name][1] += 1
        busy = merged_duration([(row[0], row[1]) for row in kernels])
        windows.append(
            {
                "step": index + 1,
                "start_ns": start,
                "end_ns": end,
                "envelope_ns": end - start,
                "gpu_busy_ns": busy,
                "gpu_gap_ns": end - start - busy,
                "kernel_instances": len(kernels),
                "runtime_calls": len(runtimes),
            }
        )

    def top_rows(values, limit=30):
        return [
            {"name": name, "total_ns": total, "count": count}
            for name, (total, count) in sorted(
                values.items(), key=lambda item: item[1][0], reverse=True
            )[:limit]
        ]

    report = {
        "schema": "apxinf.qwen25_omni.decode_window_analysis.v1",
        "source": str(args.database),
        "boundary": "six complete intervals between the last seven GPU argmax token commits",
        "windows": windows,
        "summary": {
            "steps": len(windows),
            "mean_envelope_ns": sum(row["envelope_ns"] for row in windows) / len(windows),
            "mean_gpu_busy_ns": sum(row["gpu_busy_ns"] for row in windows) / len(windows),
            "mean_gpu_gap_ns": sum(row["gpu_gap_ns"] for row in windows) / len(windows),
            "kernel_instances": sum(row["kernel_instances"] for row in windows),
            "runtime_calls": sum(row["runtime_calls"] for row in windows),
        },
        "top_kernels": top_rows(kernel_totals),
        "top_runtime_calls": top_rows(runtime_totals),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
