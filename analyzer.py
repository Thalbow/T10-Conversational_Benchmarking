"""Statistical summaries and Matplotlib figures for benchmark CSV rows."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


def analyze_results(rows, output_dir):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "success":
            groups[int(row["queue_depth"])].append(row)
    summary = []
    outliers = []
    for depth in sorted(groups):
        items = groups[depth]
        throughput = [float(x["throughput_mbps"]) for x in items]
        losses = [float(x["packet_loss_percent"]) for x in items]
        if len(throughput) >= 4:
            ordered = sorted(throughput)

            def percentile(p):
                position = (len(ordered) - 1) * p
                low, high = math.floor(position), math.ceil(position)
                if low == high:
                    return ordered[low]
                return ordered[low] * (high - position) + ordered[high] * (position - low)

            q1, q3 = percentile(0.25), percentile(0.75)
            fence_low, fence_high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            for row in items:
                value = float(row["throughput_mbps"])
                if value < fence_low or value > fence_high:
                    outliers.append({"queue_depth": depth, "trial": int(row["trial"]),
                                     "throughput_mbps": value, "iqr_low_fence": fence_low,
                                     "iqr_high_fence": fence_high})
        summary.append({"queue_depth": depth, "successful_trials": len(items),
                        "mean_throughput_mbps": mean(throughput),
                        "median_throughput_mbps": median(throughput),
                        "stddev_throughput_mbps": stdev(throughput) if len(throughput) > 1 else 0.0,
                        "min_throughput_mbps": min(throughput), "max_throughput_mbps": max(throughput),
                        "mean_packet_loss_percent": mean(losses)})
    successful = sum(row.get("status") == "success" for row in rows)
    metrics = {
        "total_tasks": len(rows), "successful_tasks": successful,
        "task_completion_percent": 100 * successful / len(rows) if rows else 0.0,
        "tool_calls": len(rows), "successful_tool_calls": successful,
        "tool_success_rate_percent": 100 * successful / len(rows) if rows else 0.0,
        "failed_tasks": len(rows) - successful,
        "error_recovery_rate_percent": 100 * sum(row.get("attempt", 1) > 1 and row.get("status") == "success" for row in rows) / max(1, sum(row.get("attempt", 1) > 1 for row in rows)),
        "mean_execution_seconds": mean(float(row.get("runtime_seconds", 0)) for row in rows) if rows else 0.0,
        "outlier_count": len(outliers), "outliers": outliers,
        "queue_depth_summary": summary,
    }
    out = Path(output_dir)
    (out / "statistics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if summary:
        with (out / "statistics_by_queue.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return metrics


def generate_graphs(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "success":
            groups[int(row["queue_depth"])].append(float(row["throughput_mbps"]))
    if not groups:
        return []
    x = sorted(groups)
    means = [mean(groups[k]) for k in x]
    errors = [stdev(groups[k]) if len(groups[k]) > 1 else 0 for k in x]
    paths = []
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, means, yerr=errors, marker="o", capsize=4)
    ax.set(title="Throughput vs Queue Depth", xlabel="Queue depth (packets)", ylabel="Throughput (Mbps)")
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    p = out / "throughput_vs_queue_depth.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p.name)
    loss_groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "success":
            loss_groups[int(row["queue_depth"])].append(float(row["packet_loss_percent"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted(loss_groups), [mean(loss_groups[k]) for k in sorted(loss_groups)], marker="o")
    ax.set(title="Packet Loss vs Queue Depth", xlabel="Queue depth (packets)", ylabel="Packet loss (%)")
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    p = out / "packet_loss_vs_queue_depth.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p.name)
    return paths
