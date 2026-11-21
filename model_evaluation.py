"""Run the T10 3 prompt styles x 3 provider models x 8 passes study.

This evaluates intent translation only; it does not start Mininet. Every API
failure, malformed response, and local validation error remains a failed row
instead of silently becoming a local-fallback result.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from llm import DEFAULT_MAX_TOKENS, GEMINI_EVALUATION_MODELS, chat_with_retry, extract_json_object, message_text, provider_settings
from topology import validate_config


STYLES = ("concise", "ambiguous", "detailed")
FIELDS = ["style", "model", "pass", "scenario", "request", "expected_config",
          "actual_config", "task_completed", "parameter_error", "api_error",
          "latency_seconds", "prompt_tokens", "completion_tokens", "attempts", "finish_reason"]

SCENARIOS = [
    {"id": "S01", "protocol": "UDP", "topology": "dumbbell", "delay_ms": 20, "queue_depth": 10},
    {"id": "S02", "protocol": "TCP", "topology": "star", "delay_ms": 5, "queue_depth": 30},
    {"id": "S03", "protocol": "UDP", "topology": "tree", "delay_ms": 15, "queue_depth": 40},
    {"id": "S04", "protocol": "TCP", "topology": "dumbbell", "delay_ms": 30, "queue_depth": 50},
    {"id": "S05", "protocol": "UDP", "topology": "star", "delay_ms": 10, "queue_depth": 60},
    {"id": "S06", "protocol": "TCP", "topology": "tree", "delay_ms": 25, "queue_depth": 70},
    {"id": "S07", "protocol": "UDP", "topology": "dumbbell", "delay_ms": 5, "queue_depth": 80},
    {"id": "S08", "protocol": "TCP", "topology": "star", "delay_ms": 20, "queue_depth": 90},
]

STYLE_INSTRUCTIONS = {
    "concise": "The operator writes briefly. Parse the compact request; return only the config JSON.",
    "ambiguous": "The operator uses informal wording and imprecise phrasing. Infer the stated protocol, topology, delay, queue depth, and trial count; use documented defaults for all other fields. Return only JSON.",
    "detailed": "The operator gives a detailed specification. Extract every stated benchmark parameter carefully and return only the config JSON.",
}


def render_request(scenario, style):
    p, t, d, q = scenario["protocol"], scenario["topology"], scenario["delay_ms"], scenario["queue_depth"]
    if style == "concise":
        return f"{p} {t}; queue {q}; delay {d} ms; 1 trial; 2 seconds."
    if style == "ambiguous":
        return f"Can you check {p} performance on a {t} setup? Keep the buffer around {q} packets and link delay about {d} ms; use exactly one trial lasting 2 seconds."
    return (f"Please run a {p} throughput benchmark using the {t} Mininet topology. "
            f"Set the queue depth to exactly {q} packets, queue_depth_step to 10 packets, and link delay to {d} ms. "
            "Use one trial lasting 2 seconds. Return protocol, topology, delay_ms, "
            "queue_depth_start, queue_depth_end, queue_depth_step, trials, duration, "
            "bandwidth_mbps, and udp_rate_mbps in the JSON result.")


def expected_config(scenario):
    q = scenario["queue_depth"]
    return {"protocol": scenario["protocol"], "topology": scenario["topology"],
            "delay_ms": scenario["delay_ms"], "queue_depth_start": q,
            "queue_depth_end": q, "queue_depth_step": 10, "trials": 1,
            "duration": 2, "bandwidth_mbps": 10, "udp_rate_mbps": 8}


def discover_free_models():
    request = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        items = json.load(response).get("data", [])
    free = []
    for item in items:
        model_id = item.get("id", "")
        pricing = item.get("pricing", {}) or {}
        is_free = model_id.endswith(":free") or (pricing.get("prompt") == "0" and pricing.get("completion") == "0")
        if model_id and is_free:
            free.append(model_id)
    # The endpoint order is retained because it generally reflects provider
    # ranking; keep the selected IDs in the output for reproducibility.
    unique = list(dict.fromkeys(free))
    if len(unique) < 3:
        raise RuntimeError("OpenRouter returned fewer than three free models. Pass three available model IDs with --models.")
    return unique[:3]


def parse_json_response(content):
    return extract_json_object(content)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, models):
    output = {"total_runs": len(rows), "models": models, "prompt_styles": list(STYLES), "passes_per_cell": 8,
              "successful_runs": sum(row["task_completed"] for row in rows),
              "api_failures": sum(bool(row["api_error"]) for row in rows), "by_model_style": []}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["style"])].append(row)
    for (model, style), items in sorted(groups.items()):
        output["by_model_style"].append({
            "model": model, "style": style, "runs": len(items),
            "task_completion_percent": 100 * sum(row["task_completed"] for row in items) / len(items),
            "parameter_error_percent": 100 * sum(row["parameter_error"] for row in items) / len(items),
            "api_failure_count": sum(bool(row["api_error"]) for row in items),
            "mean_latency_seconds": sum(float(row["latency_seconds"]) for row in items) / len(items),
            "prompt_tokens": sum(int(row["prompt_tokens"] or 0) for row in items),
            "completion_tokens": sum(int(row["completion_tokens"] or 0) for row in items),
        })
    return output


def _read_saved_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["pass"] = int(row["pass"])
        row["task_completed"] = str(row["task_completed"]).lower() == "true"
        row["parameter_error"] = str(row["parameter_error"]).lower() == "true"
        row["latency_seconds"] = float(row["latency_seconds"] or 0)
        row["attempts"] = int(row["attempts"] or 0)
    return rows


def run(models=None, output="results/model_evaluation", delay_seconds=5.0, retry_from=None):
    load_dotenv(Path(__file__).resolve().parent / ".env")
    provider, api_key, base_url, _default_model = provider_settings()
    if not api_key:
        raise RuntimeError(f"API key for {provider} is missing from the project .env file")

    retry_path = None
    saved_rows = []
    if retry_from:
        retry_path = Path(retry_from).expanduser()
        if retry_path.is_dir():
            retry_path = retry_path / "model_comparison.csv"
        if not retry_path.is_file():
            raise FileNotFoundError(f"Cannot find saved model_comparison.csv: {retry_path}")
        saved_rows = _read_saved_rows(retry_path)
        out = retry_path.parent
        if models is None:
            models = list(dict.fromkeys(row["model"] for row in saved_rows))
    else:
        out = Path(output) / datetime.now().strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=False)

    if models is None:
        models = list(GEMINI_EVALUATION_MODELS) if provider == "gemini" else discover_free_models()
    if len(models) != 3 or len(set(models)) != 3:
        raise ValueError("Provide exactly three distinct model IDs")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=90.0, max_retries=0)
    existing = {(row["style"], row["model"], row["pass"]): row for row in saved_rows}
    total = len(STYLES) * len(models) * len(SCENARIOS)
    targets = []
    for style in STYLES:
        for model in models:
            for pass_number, scenario in enumerate(SCENARIOS, 1):
                key = (style, model, pass_number)
                if key not in existing or not existing[key]["task_completed"]:
                    targets.append((key, scenario))

    if retry_path:
        print(f"[INFO] Resume mode: keeping {len(existing) - len(targets)} successful rows; retrying {len(targets)} failed or missing rows.")
    else:
        print(f"[INFO] Starting {total} model/style/scenario requests.")

    for run_index, (key, scenario) in enumerate(targets, 1):
        style, model, pass_number = key
        request_text = render_request(scenario, style)
        expected = expected_config(scenario)
        start = time.monotonic()
        actual, api_error, prompt_tokens, completion_tokens = {}, "", "", ""
        attempts, finish_reason = 0, ""
        parameter_error = False
        try:
            response, attempts = chat_with_retry(
                client,
                attempts=2,
                base_delay=2.0,
                max_delay=20.0,
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Convert a Mininet benchmark request to JSON only. Required keys: protocol, topology, "
                        "delay_ms, queue_depth_start, queue_depth_end, queue_depth_step, trials, duration, "
                        "bandwidth_mbps, udp_rate_mbps. Defaults: TCP, dumbbell, 20, 10, 10, 10, 3, 5, 10, 8. "
                        + STYLE_INSTRUCTIONS[style]
                    )},
                    {"role": "user", "content": request_text},
                ],
                temperature=0,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", "") or ""
            completion_tokens = getattr(usage, "completion_tokens", "") or ""
            try:
                text, finish_reason = message_text(response)
                actual = validate_config(parse_json_response(text))
            except Exception as exc:
                parameter_error = True
                api_error = f"Invalid model output: {type(exc).__name__}: {exc} (finish_reason={finish_reason})"
            if not api_error:
                parameter_error = any(actual.get(field) != value for field, value in expected.items())
            if parameter_error:
                api_error = api_error or "Parsed config differs from expected parameters"
        except Exception as exc:
            api_error = f"{type(exc).__name__}: {exc}"

        elapsed = round(time.monotonic() - start, 4)
        existing[key] = {"style": style, "model": model, "pass": pass_number,
                         "scenario": scenario["id"], "request": request_text,
                         "expected_config": json.dumps(expected, sort_keys=True),
                         "actual_config": json.dumps(actual, sort_keys=True),
                         "task_completed": not parameter_error and not api_error,
                         "parameter_error": parameter_error, "api_error": api_error,
                         "latency_seconds": elapsed, "prompt_tokens": prompt_tokens,
                         "completion_tokens": completion_tokens, "attempts": attempts,
                         "finish_reason": finish_reason}
        ordered_rows = [existing[(style_name, model_name, pass_no)]
                        for style_name in STYLES for model_name in models for pass_no in range(1, 9)
                        if (style_name, model_name, pass_no) in existing]
        write_rows(out / "model_comparison.csv", ordered_rows)
        label = "RETRY" if retry_path else "RUN"
        print(f"[{run_index}/{len(targets)}] {label} {style} | {model} | pass {pass_number}: "
              f"{'OK' if existing[key]['task_completed'] else 'FAILED'} ({elapsed:.2f}s)")
        if run_index < len(targets) and delay_seconds > 0:
            time.sleep(delay_seconds)

    ordered_rows = [existing[(style_name, model_name, pass_no)]
                    for style_name in STYLES for model_name in models for pass_no in range(1, 9)
                    if (style_name, model_name, pass_no) in existing]
    write_rows(out / "model_comparison.csv", ordered_rows)
    summary = summarize(ordered_rows, models)
    (out / "model_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results saved to {out}")
    print(json.dumps(summary, indent=2))
    return out, summary

def main():
    parser = argparse.ArgumentParser(description="T10 provider 3x3x8 prompt/model comparison")
    parser.add_argument("--models", nargs=3, help="Three distinct model IDs; defaults to three Gemini models or three discovered free OpenRouter models")
    parser.add_argument("--output", default="results/model_evaluation")
    parser.add_argument("--retry-incomplete", metavar="CSV_OR_RUN_DIR", help="Keep successful rows and rerun only failed or missing cases from an earlier run")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between API calls; default 5 seconds")
    args = parser.parse_args()
    if args.delay < 0 or args.delay > 60:
        parser.error("--delay must be between 0 and 60 seconds")
    try:
        run(args.models, args.output, args.delay, retry_from=args.retry_incomplete)
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Completed rows remain in model_comparison.csv.")


if __name__ == "__main__":
    main()
