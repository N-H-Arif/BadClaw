# BadClaw

This repository contains NeurIPS submission code for BadClaw: Benchmarking Dimensional Trigger Attacks in LLM Agentic Systems. It studies cases where trigger evidence is distributed across time, system locations, and persistent context, then combines during routing, planning, or tool execution.

Tasks are generated from curated OpenClaw skill records, used as Skillcard, descriptions of benign tool behavior and task context. The setting additionally uses AgentCard execution metadata to model modular routing and component-mediated tool use.

The code supports two main evaluation settings:

- `baseline`: matched 1D prior baselines such as AgentPoison, ASB-PoT, and AMA.
- `badclaw`: BadClaw 1D, 2D, and 3D attacks in a agent pipeline.


## Setup

Create and activate an environment:

```bash
conda create -n badclaw python=3.11 -y
conda activate badclaw
python -m pip install -U pip
python -m pip install -r requirements.txt
```

For closed-source OpenAI API runs:

```bash
python -m pip install -r closed_source_api_eval/requirements-openai.txt
```

On Windows CMD, run from the repository root:

```cmd
conda activate badclaw
set PYTHONPATH=%CD%
```

On PowerShell:

```powershell
conda activate badclaw
$env:PYTHONPATH = $PWD
```

## Data

The repository includes the main dataset:

```text
badclaw_tasks_v2.json
```

It also includes the SkillCard source files used to build the dataset:

```text
openclaw_skill_catalog.json
curated_openclaw_skills.json
```

To use the included skill data, no extra step is needed. The benchmark commands below load `badclaw_tasks_v2.json` directly.

To regenerate it:

```bash
python run_benchmark.py \
  --bootstrap_skills \
  --regenerate_dataset \
  --dataset badclaw_tasks_v2.json \
  --tasks_per_family 60 \
  --skills_per_family 80 \
  --models echo
```

To refresh and recurate the skill catalog before regenerating tasks:

```bash
python run_benchmark.py \
  --refresh_skill_catalog \
  --regenerate_dataset \
  --dataset badclaw_tasks_v2.json \
  --tasks_per_family 60 \
  --skills_per_family 80 \
  --models echo
```

## Quick Smoke Test

```bash
python run_benchmark.py \
  --dataset badclaw_tasks_v2.json \
  --models echo \
  --platforms openclaw_like baseline_single_agent \
  --defenses none execution_policy suite \
  --max_tasks 30 \
  --outdir results/echo_smoke \
  --execution_mode real \
  --save_task_records
```

## Local Hugging Face Run

```bash
python run_benchmark.py \
  --dataset badclaw_tasks_v2.json \
  --models meta-llama/Llama-3.2-3B-Instruct \
  --platforms openclaw_like baseline_single_agent \
  --defenses none execution_policy suite \
  --max_tasks 30 \
  --outdir results/llama32_3b \
  --execution_mode real \
  --save_task_records
```

Use `--max_tasks 0` or omit `--max_tasks` for the full dataset.

## Closed-Source API Run

Windows CMD:

```cmd
set OPENAI_API_KEY=<your_api_key>
set PYTHONPATH=%CD%
python closed_source_api_eval\run_openai_benchmark.py ^
  --dataset badclaw_tasks_v2.json ^
  --models gpt-5.4-mini ^
  --platforms openclaw_like baseline_single_agent ^
  --defenses none execution_policy suite ^
  --max_tasks 30 ^
  --outdir closed_source_api_eval\results ^
  --save_task_records
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "<your_api_key>"
$env:PYTHONPATH = $PWD
python closed_source_api_eval\run_openai_benchmark.py `
  --dataset badclaw_tasks_v2.json `
  --models gpt-5.4-mini `
  --platforms openclaw_like baseline_single_agent `
  --defenses none execution_policy suite `
  --max_tasks 30 `
  --outdir closed_source_api_eval\results `
  --save_task_records
```

Replace `gpt-5.4-mini` with any available API model ID.

## Common Flags

- `--models`: one or more model names.
- `--platforms`: usually `openclaw_like baseline_single_agent`.
- `--defenses`: e.g. `none execution_policy suite`.
- `--max_tasks`: set to `30` for a quick run; omit for full run.
- `--outdir`: output folder.
- `--execution_mode real`: model-driven planning.
- `--save_task_records`: writes per-task JSONL traces.

Available defenses:

```text
none proof_guardrail execution_policy tool_anomaly capability_isolation least_privilege trigger_sensitivity causal_tracing 
```

## Outputs

Each run writes:

```text
summary.json
<model>__<platform>__<defense>.json
<model>__<platform>__<defense>.paper.json
<model>__<defense>.paper_comparison.json
<model>__<platform>__<defense>.tasks.jsonl
```

The main comparison file is:

```text
<model>__<defense>.paper_comparison.json
```

## Cluster

Cluster files are in:

```text
cluster/
```

Create a slurm file.

Set tokens through environment variables before submitting:

```bash
export HF_TOKEN=<your_hf_token>
sbatch run_badclaw.slurm
```

Do not commit API keys, Hugging Face tokens, model caches, large result folders, or cluster logs.
