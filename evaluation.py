"""Offline 25-request intent benchmark for the T10 parser criteria."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from agent import local_analysis
from topology import validate_config


CASES = [
    ("Run TCP on dumbbell, 20 ms, queue depth 10", "TCP", "dumbbell", 10, 10, 10),
    ("UDP throughput on dumbbell from 10 to 100 packets, 20 ms", "UDP", "dumbbell", 10, 100, 10),
    ("TCP on star from 5 to 50 with step 5", "TCP", "star", 5, 50, 5),
    ("UDP on tree, queue depth 20, 10 ms", "UDP", "tree", 20, 20, 10),
    ("TCP dumbbell, queue between 1 and 16, 5 trials", "TCP", "dumbbell", 1, 16, 10),
    ("UDP star, 40 ms, from 10 to 40, step 10", "UDP", "star", 10, 40, 10),
    ("TCP tree 30 ms queue 8", "TCP", "tree", 8, 8, 10),
    ("UDP dumbbell 15 ms from 20 to 80 step 20", "UDP", "dumbbell", 20, 80, 20),
    ("TCP star queue 12", "TCP", "star", 12, 12, 10),
    ("UDP tree from 5 to 25 step 5", "UDP", "tree", 5, 25, 5),
    ("TCP dumbbell delay 5 ms, queue 30", "TCP", "dumbbell", 30, 30, 10),
    ("UDP star delay 25ms, queues from 10 to 50 step 10", "UDP", "star", 10, 50, 10),
    ("TCP tree from 10 to 30 step 10", "TCP", "tree", 10, 30, 10),
    ("UDP over dumbbell queue depth 100", "UDP", "dumbbell", 100, 100, 10),
    ("TCP star 3 trials for 4 seconds", "TCP", "star", 10, 10, 10),
    ("UDP tree 50 ms from 15 to 45 step 15", "UDP", "tree", 15, 45, 15),
    ("TCP dumbbell queue 2 to 20 step 2", "TCP", "dumbbell", 2, 20, 2),
    ("UDP star from 16 to 64 step 16", "UDP", "star", 16, 64, 16),
    ("TCP tree 2 trials", "TCP", "tree", 10, 10, 10),
    ("UDP dumbbell 1 ms queue from 10 to 30", "UDP", "dumbbell", 10, 30, 10),
    ("TCP star from 3 to 9 step 3", "TCP", "star", 3, 9, 3),
    ("UDP tree queue 7", "UDP", "tree", 7, 7, 10),
    ("TCP dumbbell 100 ms, 10 seconds", "TCP", "dumbbell", 10, 10, 10),
    ("UDP star from 25 to 75 step 25", "UDP", "star", 25, 75, 25),
    ("TCP tree queue depth from 4 to 12 step 4", "TCP", "tree", 4, 12, 4),
]


def evaluate(output_dir="results/evaluation"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    total_time = 0.0
    errors = 0
    for index, (request, protocol, topology, qstart, qend, qstep) in enumerate(CASES, 1):
        start = time.perf_counter()
        try:
            config = validate_config(local_analysis(request))
            parameter_error = any(config[key] != expected for key, expected in (
                ("protocol", protocol), ("topology", topology), ("queue_depth_start", qstart),
                ("queue_depth_end", qend), ("queue_depth_step", qstep)))
            error = "" if not parameter_error else "Parsed parameters differ from expected values"
        except Exception as exc:
            config, parameter_error, error = {}, True, f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        total_time += elapsed
        errors += int(parameter_error)
        rows.append({"case": index, "request": request, "expected_protocol": protocol,
                     "expected_topology": topology, "expected_queue_start": qstart,
                     "expected_queue_end": qend, "expected_queue_step": qstep,
                     "actual_config": json.dumps(config, sort_keys=True), "parameter_error": parameter_error,
                     "elapsed_seconds": round(elapsed, 6), "error": error})
    with (out / "intent_evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"evaluation_type": "offline intent parsing; does not execute Mininet",
               "tasks": len(rows), "completed": len(rows) - errors,
               "task_completion_percent": 100 * (len(rows) - errors) / len(rows),
               "tool_parameter_error_percent": 100 * errors / len(rows),
               "mean_execution_seconds": total_time / len(rows)}
    (out / "intent_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/evaluation")
    evaluate(parser.parse_args().output)
