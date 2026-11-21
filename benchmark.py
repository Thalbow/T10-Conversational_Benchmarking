"""Execute repeatable Mininet/iperf3 benchmark trials and preserve raw CSV."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from topology import create_topology, validate_config

FIELDS = ["protocol", "topology", "delay_ms", "queue_depth", "trial", "attempt",
          "throughput_mbps", "packet_loss_percent", "packets_received", "packets_lost",
          "runtime_seconds", "status", "error"]


def _extract_stats(data, protocol):
    end = data.get("end", {})
    candidates = [end.get("sum_received"), end.get("sum"), end.get("sum_sent")]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        bps = candidate.get("bits_per_second", 0)
        if bps <= 0:
            continue
        return {
            "throughput_mbps": bps / 1_000_000,
            "packet_loss_percent": candidate.get("lost_percent", 0.0) if protocol == "UDP" else 0.0,
            "packets_received": candidate.get("packets", 0),
            "packets_lost": candidate.get("lost_packets", 0),
        }
    raise RuntimeError("iperf3 returned no positive receiver throughput statistic")


def run_iperf_test(net, config, queue_depth, trial, attempt):
    hosts = {host.name: host for host in net.hosts}
    source, target = hosts["h1"], hosts["h3" if config["topology"] == "dumbbell" else "h4"]
    server = target.popen(["iperf3", "-s", "-1", "-J"], stdout=-1, stderr=-1, text=True)
    start = time.monotonic()
    try:
        # Inspect the listening socket without connecting. A TCP health probe
        # would consume the sole session of iperf3's -1 (one-off) server.
        ready = False
        for _ in range(30):
            if server.poll() is not None:
                raise RuntimeError("iperf3 server exited before accepting a client")
            listener = target.cmd("ss -ltn 'sport = :5201' | grep -q LISTEN; echo $?").strip()
            if listener.endswith("0"):
                ready = True
                break
            time.sleep(0.1)
        if not ready:
            raise RuntimeError("iperf3 server did not become ready within 3 seconds")
        command = ["iperf3", "-c", target.IP(), "-t", str(config["duration"]), "-J"]
        if config["protocol"] == "UDP":
            command += ["-u", "-b", f"{config['udp_rate_mbps']}M"]
        raw = source.popen(command, stdout=-1, stderr=-1, text=True).communicate(timeout=config["duration"] + 15)
        stdout, stderr = raw
        if server.poll() is None:
            server.wait(timeout=5)
        server_out, server_err = server.communicate(timeout=5)
        if server.returncode not in (0, None):
            raise RuntimeError(f"iperf3 server failed: {(server_err or '').strip()}")
        try:
            client_data = json.loads(stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"iperf3 client emitted invalid JSON: {(stderr or stdout or '')[-500:]}") from exc
        if "error" in client_data:
            raise RuntimeError(f"iperf3 client: {client_data['error']}")
        # Depending on iperf3 version, the client either embeds receiver
        # statistics in server_output_json or the one-off server emits its
        # own authoritative receiver summary on stdout.
        try:
            stats = _extract_stats(client_data, config["protocol"])
        except RuntimeError as client_error:
            try:
                server_data = json.loads(server_out or "{}")
                stats = _extract_stats(server_data, config["protocol"])
            except (RuntimeError, json.JSONDecodeError) as server_error:
                end = client_data.get("end", {})
                summary = {key: end.get(key) for key in ("sum", "sum_sent", "sum_received", "streams")}
                raise RuntimeError(
                    f"No positive receiver throughput in client or server output; "
                    f"client_end={json.dumps(summary)[:1000]}; "
                    f"server_stderr={(server_err or '').strip()[-400:]}"
                ) from server_error
        stats["runtime_seconds"] = round(time.monotonic() - start, 4)
        return stats
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=2)
            except Exception:
                server.kill()


def run_benchmark(config, output_dir, retries=1):
    config = validate_config(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "benchmark_results.csv"
    rows, recovery = [], []
    depths = range(config["queue_depth_start"], config["queue_depth_end"] + 1, config["queue_depth_step"])
    total = len(depths) * config["trials"]
    index = 0
    for depth in depths:
        for trial in range(1, config["trials"] + 1):
            index += 1
            last_error = ""
            for attempt in range(1, retries + 2):
                net = None
                started = time.monotonic()
                try:
                    print(f"[TEST {index}/{total}] {config['protocol']} {config['topology']} queue={depth} trial={trial} attempt={attempt}")
                    trial_config = dict(config, queue_depth_start=depth, queue_depth_end=depth)
                    net = create_topology(trial_config)
                    net.start()
                    time.sleep(0.4)
                    loss = net.ping([net.get("h1"), net.get("h3" if config["topology"] == "dumbbell" else "h4")])
                    if loss > 0:
                        raise RuntimeError(f"connectivity smoke test reported {loss}% packet loss")
                    stats = run_iperf_test(net, config, depth, trial, attempt)
                    row = dict(protocol=config["protocol"], topology=config["topology"], delay_ms=config["delay_ms"],
                               queue_depth=depth, trial=trial, attempt=attempt, **stats, status="success", error="")
                    rows.append(row)
                    print(f"[OK] throughput={stats['throughput_mbps']:.3f} Mbps, loss={stats['packet_loss_percent']:.3f}%")
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    print(f"[WARN] Attempt {attempt} failed: {last_error}")
                    if attempt <= retries:
                        recovery.append({"queue_depth": depth, "trial": trial, "attempt": attempt,
                                         "error": last_error, "action": "stop network, wait, rebuild topology, retry"})
                        time.sleep(min(attempt, 2))
                finally:
                    if net is not None:
                        try:
                            net.stop()
                        except Exception as exc:
                            print(f"[WARN] Mininet cleanup: {type(exc).__name__}")
                if len(rows) < index or (rows and rows[-1].get("queue_depth") != depth):
                    continue
                if rows and rows[-1].get("queue_depth") == depth and rows[-1].get("trial") == trial:
                    break
            else:
                rows.append(dict(protocol=config["protocol"], topology=config["topology"], delay_ms=config["delay_ms"],
                                 queue_depth=depth, trial=trial, attempt=retries + 1, throughput_mbps=0.0,
                                 packet_loss_percent=0.0, packets_received=0, packets_lost=0,
                                 runtime_seconds=round(time.monotonic() - started, 4), status="failed", error=last_error))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (out / "recovery_trace.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    return rows, recovery
