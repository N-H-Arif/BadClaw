from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from badclaw.datasets.loader import load_tasks
from badclaw.defenses.registry import get_defense
from badclaw.evaluation.metrics import score
from badclaw.platforms.baseline_single_agent import BaselineSingleAgentPlatform
from badclaw.platforms.openclaw_like import OpenClawLikePlatform
from closed_source_api_eval.openai_api_responder import OpenAIResponsesResponder


def make_platform(name: str, responder, defense):
    if name == "openclaw_like":
        return OpenClawLikePlatform(responder, defense, execution_mode="real")
    if name == "baseline_single_agent":
        return BaselineSingleAgentPlatform(responder, defense, execution_mode="real")
    raise ValueError(f"Unsupported platform for closed-source API eval: {name}")


def stratified_limit(tasks, max_tasks: int, seed: int = 13):
    rnd = random.Random(seed)
    buckets = defaultdict(list)

    grouped_sequences = defaultdict(list)
    ungrouped = []
    for task in tasks:
        if task.category == "attack_sequence" and task.seed_group:
            grouped_sequences[task.seed_group].append(task)
        else:
            ungrouped.append([task])

    units = list(ungrouped)
    for group_tasks in grouped_sequences.values():
        group_tasks.sort(key=lambda item: 0 if item.sequence_role == "seed" else 1)
        units.append(group_tasks)

    for unit in units:
        task = next((item for item in unit if item.sequence_role == "fire"), unit[0])
        if task.category == "clean":
            key = ("clean", "0d", "clean")
        elif task.category == "false_activation":
            key = ("false_activation", "0d", "near_miss")
        else:
            key = (
                task.category,
                str(task.metadata.get("dimension_count", 0)),
                str(task.metadata.get("attack_family", task.attack_surface or "unknown")),
            )
        buckets[key].append(unit)

    keys = list(buckets.keys())
    rnd.shuffle(keys)
    for key in keys:
        rnd.shuffle(buckets[key])

    selected = []
    while len(selected) < max_tasks:
        progressed = False
        for key in keys:
            if not buckets[key]:
                continue
            unit = buckets[key][-1]
            if len(selected) + len(unit) <= max_tasks:
                selected.extend(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def describe_slice(tasks):
    counts = defaultdict(int)
    for task in tasks:
        if task.category in {"clean", "false_activation"}:
            label = task.category
        else:
            label = f"{task.category}:{task.metadata.get('dimension_count', 0)}d"
        counts[label] += 1
    return dict(sorted(counts.items()))


def tasks_for_platform(tasks, platform_name: str):
    out = []
    for task in tasks:
        group = task.comparison_group or ""
        if task.category in {"clean", "false_activation"}:
            out.append(task)
        elif platform_name == "openclaw_like":
            if group.startswith("badclaw_"):
                out.append(task)
        elif platform_name == "baseline_single_agent":
            if group == "prior_1d":
                out.append(task)
    return out


def cross_platform_paper_view(platform_metrics: dict) -> dict:
    openclaw = platform_metrics.get("openclaw_like", {})
    baseline = platform_metrics.get("baseline_single_agent", {})
    openclaw_surface = openclaw.get("attack_success_by_group_surface", {})
    baseline_surface = baseline.get("attack_success_by_group_surface", {})

    badclaw_1d = {
        key.split(":", 1)[1]: value
        for key, value in openclaw_surface.items()
        if key.startswith("badclaw_1d:")
    }
    prior_1d = {
        key.split(":", 1)[1]: value
        for key, value in baseline_surface.items()
        if key.startswith("prior_1d:")
    }
    surfaces = sorted(set(badclaw_1d) | set(prior_1d))
    delta = {
        surface: round(badclaw_1d.get(surface, 0.0) - prior_1d.get(surface, 0.0), 4)
        for surface in surfaces
    }
    return {
        "comparison_design": "single_agent_sota_1d_vs_openclaw_badclaw_dimensions",
        "single_agent_sota_1d_by_surface": prior_1d,
        "openclaw_badclaw_1d_by_surface": badclaw_1d,
        "openclaw_badclaw_1d_minus_single_agent_sota_by_surface": delta,
        "openclaw_badclaw_2d_asr": openclaw.get("badclaw_2d_asr", 0.0),
        "openclaw_badclaw_3d_asr": openclaw.get("badclaw_3d_asr", 0.0),
        "openclaw_badclaw_dimension_table": openclaw.get("badclaw_attack_success_by_dimension_count", {}),
        "single_agent_sota_method_table": baseline.get("attack_success_by_method", {}),
        "openclaw_badclaw_method_table": openclaw.get("attack_success_by_method", {}),
    }


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="badclaw_tasks_v2.json")
    parser.add_argument("--models", nargs="+", default=["gpt-5-mini"])
    parser.add_argument("--platforms", nargs="+", default=["openclaw_like", "baseline_single_agent"])
    parser.add_argument("--defenses", nargs="+", default=["none"])
    parser.add_argument("--outdir", type=str, default="closed_source_api_eval/results")
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save_task_records", action="store_true")
    parser.add_argument("--reasoning_effort", type=str, default="")
    parser.add_argument("--max_output_tokens", type=int, default=400)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    print(f"[DATA] Loading tasks from {dataset_path}")
    tasks = load_tasks(str(dataset_path))
    print(f"[DATA] Loaded {len(tasks)} tasks before platform filtering")
    print(f"[DATA] Full slice mix: {describe_slice(tasks)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for model_name in tqdm(args.models, desc="Models"):
        responder = OpenAIResponsesResponder(
            model_name=model_name,
            reasoning_effort=args.reasoning_effort or None,
            max_output_tokens=args.max_output_tokens,
        )
        model_slug = safe_filename(model_name)
        summary[model_name] = {}
        for defense_name in tqdm(args.defenses, desc="Defenses", leave=False):
            defense = get_defense(defense_name)
            summary[model_name][defense_name] = {}
            for platform_name in tqdm(args.platforms, desc=f"Platforms [{defense_name}]", leave=False):
                platform = make_platform(platform_name, responder, defense)
                platform_tasks = tasks_for_platform(tasks, platform_name)
                if args.max_tasks and args.max_tasks > 0:
                    platform_tasks = stratified_limit(platform_tasks, args.max_tasks, seed=args.seed)
                print(f"\n[DATA] [{platform_name}] using {len(platform_tasks)} applicable tasks; slice mix: {describe_slice(platform_tasks)}")
                results = []
                for task in tqdm(platform_tasks, desc=f"Tasks [{model_name} | {platform_name} | {defense_name}]", leave=False):
                    results.append(platform.run_task(task))
                metrics = score(platform_tasks, results)
                summary[model_name][defense_name][platform_name] = metrics

                save_path = outdir / f"{model_slug}__{platform_name}__{defense_name}.json"
                write_json(save_path, metrics)

                if args.save_task_records:
                    records_path = outdir / f"{model_slug}__{platform_name}__{defense_name}.tasks.jsonl"
                    records_path.parent.mkdir(parents=True, exist_ok=True)
                    with records_path.open("w", encoding="utf-8") as fh:
                        for task, result in zip(platform_tasks, results):
                            fh.write(json.dumps({
                                "task_id": task.task_id,
                                "category": task.category,
                                "comparison_group": task.comparison_group,
                                "attack_method": task.attack_method,
                                "attack_surface": task.attack_surface,
                                "input_text": task.input_text,
                                "planned_actions": [asdict(action) for action in result.planned_actions],
                                "raw_model_output": result.raw_model_output,
                                "diagnostics": result.diagnostics,
                                "trace": [asdict(item) for item in result.trace],
                            }, ensure_ascii=False) + "\n")

                paper_view = {
                    "comparison_1d_by_surface": metrics.get("comparison_1d_by_surface", {}),
                    "badclaw_1d_minus_prior_by_surface": metrics.get("badclaw_1d_minus_prior_by_surface", {}),
                    "badclaw_2d_asr": metrics.get("badclaw_2d_asr", 0.0),
                    "badclaw_3d_asr": metrics.get("badclaw_3d_asr", 0.0),
                    "mean_attack_execution_harm": metrics.get("mean_attack_execution_harm", 0.0),
                    "mean_attack_harm_score": metrics.get("mean_attack_harm_score", 0.0),
                    "total_attack_harm_score": metrics.get("total_attack_harm_score", 0),
                    "max_attack_harm_score": metrics.get("max_attack_harm_score", 0),
                    "mean_attack_stealthiness": metrics.get("mean_attack_stealthiness", 0.0),
                    "surface_table_all": metrics.get("attack_success_by_group_surface", {}),
                    "method_table": metrics.get("attack_success_by_method", {}),
                    "group_table": metrics.get("attack_success_by_comparison_group", {}),
                    "badclaw_dimension_table": metrics.get("badclaw_attack_success_by_dimension_count", {}),
                    "badclaw_combo_table": metrics.get("badclaw_attack_success_by_dimension_combo", {}),
                }
                paper_path = outdir / f"{model_slug}__{platform_name}__{defense_name}.paper.json"
                write_json(paper_path, paper_view)
                print(f"\n[{model_name}] [{platform_name}] [{defense_name}] saved -> {save_path}")

            comparison_view = cross_platform_paper_view(summary[model_name][defense_name])
            comparison_path = outdir / f"{model_slug}__{defense_name}.paper_comparison.json"
            write_json(comparison_path, comparison_view)

    summary_path = outdir / "summary.json"
    write_json(summary_path, summary)
    print(f"\n[DONE] Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
