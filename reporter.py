"""Create a self-contained HTML and PDF benchmark report."""
from __future__ import annotations

import html
from pathlib import Path


def write_report(metadata, stats, figures, output_dir):
    out = Path(output_dir)
    config = metadata["config"]
    rows = stats["queue_depth_summary"]
    table_rows = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["queue_depth"], row["successful_trials"], f"{row['mean_throughput_mbps']:.4f}",
            f"{row['stddev_throughput_mbps']:.4f}", f"{row['mean_packet_loss_percent']:.4f}%")) + "</tr>"
        for row in rows
    ) or '<tr><td colspan="5">No successful trials. See recovery_trace.json and benchmark_results.csv.</td></tr>'
    images = "\n".join(f'<figure><img src="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in figures)
    request = html.escape(metadata.get("request", ""))
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>T10 Benchmark Report</title>
<style>body{{font:15px system-ui;max-width:960px;margin:36px auto;padding:0 20px;color:#202938}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3dd;padding:8px;text-align:left}}th{{background:#eff3f8}}img{{max-width:100%}}.metric{{display:inline-block;margin:8px 18px 8px 0}}</style></head><body>
<h1>T10 Conversational Benchmarking Report</h1><p><b>Request:</b> {request}</p>
<h2>Run configuration</h2><pre>{html.escape(str(config))}</pre><p>Intent parser: {html.escape(metadata.get('intent_parser',''))}</p>
<h2>Evaluation</h2><p><span class="metric">Task completion: {stats['task_completion_percent']:.1f}%</span><span class="metric">Tool success: {stats['tool_success_rate_percent']:.1f}% ({stats['successful_tool_calls']}/{stats['tool_calls']})</span><span class="metric">Mean trial time: {stats['mean_execution_seconds']:.2f}s</span><span class="metric">Error recovery rate: {stats['error_recovery_rate_percent']:.1f}%</span><span class="metric">Agent iterations: {metadata.get('reasoning_iterations',1)}</span></p>
<p>OpenRouter tokens — prompt: {html.escape(str(metadata.get('api_telemetry',{}).get('prompt_tokens','unavailable')))}, completion: {html.escape(str(metadata.get('api_telemetry',{}).get('completion_tokens','unavailable')))}.</p>
<h2>Queue-depth statistics</h2><table><thead><tr><th>Queue depth</th><th>Successful trials</th><th>Mean throughput (Mbps)</th><th>Std dev (Mbps)</th><th>Mean packet loss</th></tr></thead><tbody>{table_rows}</tbody></table>
<p>Outliers flagged by the 1.5×IQR rule: {stats['outlier_count']}. Failed tests: {stats['failed_tasks']}. Review recovery_trace.json and benchmark_results.csv for failed attempts and causes.</p>
<h2>Figures</h2>{images}<p>Runtime: {metadata.get('runtime_seconds',0):.2f}s. Raw records: benchmark_results.csv. Recovery events: recovery_trace.json.</p></body></html>'''
    (out / "report.html").write_text(page, encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(out / "report.pdf") as pdf:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            lines = ["T10 Conversational Benchmarking Report", "", f"Request: {metadata.get('request','')}",
                     f"Parser: {metadata.get('intent_parser','')}", f"Configuration: {config}", "",
                     f"Task completion: {stats['task_completion_percent']:.1f}% ({stats['successful_tasks']}/{stats['total_tasks']})",
                     f"Tool success rate: {stats['tool_success_rate_percent']:.1f}% ({stats['successful_tool_calls']}/{stats['tool_calls']})",
                     f"Error recovery rate: {stats['error_recovery_rate_percent']:.1f}%",
                     f"Outliers flagged (1.5x IQR): {stats['outlier_count']}",
                     f"Agent reasoning/recovery iterations: {metadata.get('reasoning_iterations', 1)}",
                     f"OpenRouter prompt/completion tokens: {metadata.get('api_telemetry', {}).get('prompt_tokens', 'unavailable')}/{metadata.get('api_telemetry', {}).get('completion_tokens', 'unavailable')}",
                     f"Mean trial runtime: {stats['mean_execution_seconds']:.2f} seconds", "", "Per queue depth:"]
            for row in rows:
                lines.append(f"  {row['queue_depth']} packets: n={row['successful_trials']}, throughput={row['mean_throughput_mbps']:.4f} ± {row['stddev_throughput_mbps']:.4f} Mbps, loss={row['mean_packet_loss_percent']:.4f}%")
            if not rows:
                lines.append("  No successful benchmark records.")
            ax.text(.06, .95, "\n".join(lines), va="top", wrap=True, fontsize=10, family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            for image in figures:
                fig, ax = plt.subplots(figsize=(8.27, 5.8))
                ax.imshow(plt.imread(out / image))
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    except Exception as exc:
        (out / "pdf_report_error.txt").write_text(f"PDF generation failed: {type(exc).__name__}: {exc}\nHTML report remains available.\n", encoding="utf-8")
