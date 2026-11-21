# T10 Conversational Benchmarking

This project turns a plain-English network test request into a validated Mininet benchmark, records raw per-trial results, retries failed trials after cleaning up their network, and writes plots plus HTML/PDF statistical reports.

## Requirements

- Ubuntu 22.04 or compatible Linux, Python 3.10+
- Mininet with Open vSwitch, `iperf3`, `iproute2`/`tc`
- Python packages in `requirements.txt`
- Gemini is the default provider. Set `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...`, and `MODEL_NAME=gemini-3.8-flash` in `.env`. The code uses Google Gemini API's OpenAI-compatible endpoint. OpenRouter remains supported with `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY=...`, and an OpenRouter model name. The parser uses deterministic local fallback if the API is unavailable.

Install the Python dependencies for the normal account before running the network as root:

```bash
python3 -m pip install --user -r requirements.txt
```

The agent adds the invoking account's per-user Python packages when launched through `sudo`.

## Run

From the project directory, request elevated privileges for Mininet (network namespaces and qdiscs require root):

```bash
sudo python3 agent.py --request "Test how UDP throughput varies as bottleneck queue depth changes from 10 to 100 packets under 20 ms delay, step 10, 3 trials"
```

To use TCP or another topology, say `TCP` or `UDP` and `dumbbell`, `star`, or `tree` in the request. The supported queue sweep limit is 101 points; trials are limited to 100 and each run to 300 seconds. A failed trial is retried once by default. Set `--retries 0..3` to change that. Results are placed in a timestamped folder under `results/runs/` and do not overwrite earlier runs.

Each run folder contains `run_metadata.json`, `benchmark_results.csv`, `recovery_trace.json`, `statistics.json`, `statistics_by_queue.csv`, PNG plots, `report.html`, and `report.pdf` (when Matplotlib PDF support is available).

## Offline parser evaluation

```bash
python3 evaluation.py
```

This runs a 25-request deterministic intent-parsing battery and writes completion and parameter-error metrics to `results/evaluation/`. It does not create a Mininet network. Actual end-to-end benchmark completion is measured separately in each run's metadata and report.

## Model and prompt-style comparison (T10 point 2)

Run the 72-case comparison from the project directory as the normal user (Mininet/root is not needed):

```bash
python3 model_evaluation.py
```

With Gemini selected, the runner tests three configured Gemini models (`gemini-3.5-flash-lite`, `gemini-3.6-flash`, and `gemini-3.1-flash-lite`). With OpenRouter selected, it discovers three currently free OpenRouter models. It tests concise, ambiguous, and detailed request styles over eight benchmark scenarios per model/style. It writes a checkpointed `model_comparison.csv` after every API call and a `model_comparison_summary.json` at completion. API errors and invalid parameter outputs count as failures; there is no local fallback in this evaluator. To pin three model IDs explicitly, use `python3 model_evaluation.py --models model-id-one model-id-two model-id-three`. The model IDs used are stored in the summary.

## Validation boundaries

The configuration is validated locally before network setup. Benchmark trials check host reachability, use receiver-side iperf3 JSON statistics, write explicit success/failure rows, and record each retry. The model response is never executed as shell input. A run that cannot create Mininet namespaces or start OVS is recorded as failed rather than reported as a successful benchmark.
