"""Conversational entry point and orchestration for T10 benchmarking."""
from __future__ import annotations

import argparse
import json
import os
import pwd
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Under sudo, Python disables the invoking user's site-packages. Re-add only
# that user's standard per-user package directory so the documented --user
# installation remains available to this root-run Mininet process.
if os.environ.get("SUDO_USER"):
    try:
        invoking_home = Path(pwd.getpwnam(os.environ["SUDO_USER"]).pw_dir)
        invoking_site = invoking_home / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if invoking_site.is_dir():
            sys.path.insert(0, str(invoking_site))
    except (KeyError, OSError):
        pass

from dotenv import load_dotenv

from analyzer import analyze_results, generate_graphs
from benchmark import run_benchmark
from llm import DEFAULT_MAX_TOKENS, chat_with_retry, extract_json_object, message_text, provider_settings
from reporter import write_report
from topology import validate_config


def local_analysis(request: str) -> dict:
    """Parse the supported request vocabulary without a network call."""
    import re

    text = request.lower()
    config = {
        "protocol": "UDP" if re.search(r"\budp\b", text) else "TCP",
        "topology": next((t for t in ("dumbbell", "star", "tree") if re.search(rf"\b{t}\b", text)), "dumbbell"),
        "delay_ms": 20,
        "queue_depth_start": 10,
        "queue_depth_end": 10,
        "queue_depth_step": 10,
        "trials": 3,
        "duration": 5,
        "bandwidth_mbps": 10,
        "udp_rate_mbps": 8,
    }
    match = re.search(r"(\d+)\s*ms", text)
    if match:
        config["delay_ms"] = int(match.group(1))
    match = re.search(r"\bqueue(?: depth)?\s+(?:from\s+)?(\d+)\s+(?:to|and)\s+(\d+)", text)
    if not match:
        match = re.search(r"(?:from|between)\s+(\d+)\s+(?:to|and)\s+(\d+)", text)
    if match:
        config["queue_depth_start"], config["queue_depth_end"] = map(int, match.groups())
    match = re.search(r"(?:step|increment|interval)\s+(?:of\s+)?(\d+)", text)
    if match:
        config["queue_depth_step"] = int(match.group(1))
    elif re.search(r"\bqueue(?: depth)?\s+(?:of\s+)?\d+\b", text):
        config["queue_depth_start"] = config["queue_depth_end"] = int(
            re.search(r"\bqueue(?: depth)?\s+(?:of\s+)?(\d+)\b", text).group(1)
        )
    match = re.search(r"(\d+)\s*(?:trials?|runs?|repetitions?)", text)
    if match:
        config["trials"] = int(match.group(1))
    match = re.search(r"(\d+)\s*(?:seconds?|secs?)", text)
    if match:
        config["duration"] = int(match.group(1))
    return config


def analyze_intent(request: str) -> tuple[dict, str, dict]:
    """Use OpenRouter when configured; validate every model response locally."""
    load_dotenv(Path(__file__).resolve().parent / ".env")
    try:
        provider, api_key, base_url, model = provider_settings()
    except Exception as exc:
        print(f"[WARN] LLM provider configuration failed; using local parser ({type(exc).__name__}).")
        provider, api_key, base_url, model = "gemini", None, "", "gemini-3.8-flash"
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=90.0, max_retries=0)
            response, _attempts = chat_with_retry(
                client, attempts=4,
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Convert benchmark requests to one JSON object only. Fields: protocol (TCP/UDP), "
                        "topology (dumbbell/star/tree), delay_ms, queue_depth_start, queue_depth_end, "
                        "queue_depth_step, trials, duration, bandwidth_mbps, udp_rate_mbps. "
                        "Defaults: TCP, dumbbell, 20, 10, 10, 10, 3, 5, 10, 8. "
                        "Do not include commands or additional fields."
                    )},
                    {"role": "user", "content": request},
                ],
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            answer, _finish = message_text(response)
            parsed = extract_json_object(answer)
            parsed = validate_config(parsed)
            usage = getattr(response, "usage", None)
            return parsed, f"{provider.title()} AI", {
                "model": model,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        except Exception as exc:
            print(f"[WARN] {provider.title()} intent parsing failed; using local parser ({type(exc).__name__}).")
            config = validate_config(local_analysis(request))
            return config, "Local Fallback", {"model": model, "fallback_reason": f"{type(exc).__name__}: {exc}"}
    config = validate_config(local_analysis(request))
    return config, "Local Fallback", {"fallback_reason": f"{provider.upper()} API key is not configured"}


def execute_request(request: str, output_root: Path, max_retries: int = 1) -> Path:
    config, parser, api_telemetry = analyze_intent(request)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = output_root / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "request": request,
        "config": config,
        "intent_parser": parser,
        "api_telemetry": api_telemetry,
        "started_at": datetime.now().astimezone().isoformat(),
        "max_retries": max_retries,
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] Intent parser: {parser}\n[INFO] Validated config: {json.dumps(config)}")
    started = time.monotonic()
    results, recovery = run_benchmark(config, outdir, retries=max_retries)
    stats = analyze_results(results, outdir)
    figures = generate_graphs(results, outdir)
    metadata.update({
        "finished_at": datetime.now().astimezone().isoformat(),
        "runtime_seconds": round(time.monotonic() - started, 3),
        "recovery": recovery,
        "reasoning_iterations": 1 + len(recovery),
        "task_completed": bool(results) and all(row["status"] == "success" for row in results),
        "successful_tests": sum(row["status"] == "success" for row in results),
        "total_tests": len(results),
    })
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(metadata, stats, figures, outdir)
    # Mininet must run as root, but result files should remain manageable by
    # the account that invoked sudo. Limit ownership changes to the selected
    # output directory inside this project tree.
    if os.geteuid() == 0 and os.environ.get("SUDO_UID") and os.environ.get("SUDO_GID"):
        project_root = Path(__file__).resolve().parent
        resolved_output = output_root.resolve()
        if project_root in resolved_output.parents:
            uid, gid = int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
            try:
                for path in outdir.rglob("*"):
                    os.chown(path, uid, gid)
                os.chown(outdir, uid, gid)
                os.chown(resolved_output, uid, gid)
            except OSError as exc:
                print(f"[WARN] Could not restore artifact ownership: {type(exc).__name__}")
    print(f"\n[INFO] Artifacts: {outdir}")
    print(f"[INFO] Task completion: {metadata['successful_tests']}/{metadata['total_tests']}")
    return outdir


def main() -> int:
    parser = argparse.ArgumentParser(description="T10 Conversational Benchmarking Agent")
    parser.add_argument("--request", help="Plain-English benchmark request")
    parser.add_argument("--output", default="results/runs", help="Artifact root directory")
    parser.add_argument("--retries", type=int, default=1, help="Reruns per failed test")
    args = parser.parse_args()
    request = args.request or input("Enter benchmark request: ").strip()
    if not request:
        print("[ERROR] Request cannot be empty.", file=sys.stderr)
        return 2
    if args.retries < 0 or args.retries > 3:
        print("[ERROR] retries must be between 0 and 3.", file=sys.stderr)
        return 2
    try:
        execute_request(request, Path(args.output), args.retries)
        return 0
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Benchmark stopped by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
