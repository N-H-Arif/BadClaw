from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from badclaw.datasets.curate_openclaw import ensure_curated_catalog
from badclaw.datasets.generate import generate_dataset
from badclaw.datasets.generate_from_openclaw import generate_dimensional_dataset_from_curated_catalog
from badclaw.datasets.loader import load_tasks
from badclaw.evaluation.metrics import score
from badclaw.models.factory import create_model
from badclaw.platforms.baseline_single_agent import BaselineSingleAgentPlatform
from badclaw.platforms.openclaw_like import OpenClawLikePlatform
from badclaw.platforms.smolagents_style import SmolagentsStylePlatform
from badclaw.defenses.registry import get_defense


def make_platform(name: str, responder, defense, execution_mode: str):
    if name == 'openclaw_like':
        return OpenClawLikePlatform(responder, defense, execution_mode=execution_mode)
    if name == 'baseline_single_agent':
        return BaselineSingleAgentPlatform(responder, defense, execution_mode=execution_mode)
    if name == 'smolagents_style':
        return SmolagentsStylePlatform(responder)
    raise ValueError(f'Unknown platform: {name}')


def stratified_limit(tasks, max_tasks: int, seed: int = 13):
    rnd = random.Random(seed)
    buckets = defaultdict(list)

    grouped_sequences = defaultdict(list)
    ungrouped = []
    for t in tasks:
        if t.category == 'attack_sequence' and t.seed_group:
            grouped_sequences[t.seed_group].append(t)
        else:
            ungrouped.append([t])

    units = list(ungrouped)
    for group_tasks in grouped_sequences.values():
        group_tasks.sort(key=lambda t: 0 if t.sequence_role == 'seed' else 1)
        units.append(group_tasks)

    for unit in units:
        t = next((item for item in unit if item.sequence_role == 'fire'), unit[0])
        if t.category == 'clean':
            key = ('clean', '0d', 'clean')
        elif t.category == 'false_activation':
            key = ('false_activation', '0d', 'near_miss')
        else:
            key = (
                t.category,
                str(t.metadata.get('dimension_count', 0)),
                str(t.metadata.get('attack_family', t.attack_surface or 'unknown')),
            )
        buckets[key].append(unit)
    keys = list(buckets.keys())
    rnd.shuffle(keys)
    for k in keys:
        rnd.shuffle(buckets[k])
    selected = []
    while len(selected) < max_tasks:
        progressed = False
        for k in keys:
            if not buckets[k]:
                continue
            unit = buckets[k][-1]
            if len(selected) + len(unit) <= max_tasks:
                selected.extend(buckets[k].pop())
                progressed = True
        if not progressed:
            break
    return selected


def describe_slice(tasks):
    counts = defaultdict(int)
    for t in tasks:
        if t.category in {'clean', 'false_activation'}:
            label = t.category
        else:
            label = f"{t.category}:{t.metadata.get('dimension_count', 0)}d"
        counts[label] += 1
    return dict(sorted(counts.items()))


def tasks_for_platform(tasks, platform_name: str):
    out = []
    for task in tasks:
        group = task.comparison_group or ''
        if task.category in {'clean', 'false_activation'}:
            out.append(task)
        elif platform_name == 'openclaw_like':
            if group.startswith('badclaw_'):
                out.append(task)
        elif platform_name == 'baseline_single_agent':
            if group == 'prior_1d':
                out.append(task)
        else:
            out.append(task)
    return out


def cross_platform_paper_view(platform_metrics: dict) -> dict:
    openclaw = platform_metrics.get('openclaw_like', {})
    baseline = platform_metrics.get('baseline_single_agent', {})
    openclaw_surface = openclaw.get('attack_success_by_group_surface', {})
    baseline_surface = baseline.get('attack_success_by_group_surface', {})

    badclaw_1d = {
        key.split(':', 1)[1]: value
        for key, value in openclaw_surface.items()
        if key.startswith('badclaw_1d:')
    }
    prior_1d = {
        key.split(':', 1)[1]: value
        for key, value in baseline_surface.items()
        if key.startswith('prior_1d:')
    }
    surfaces = sorted(set(badclaw_1d) | set(prior_1d))
    delta = {
        surface: round(badclaw_1d.get(surface, 0.0) - prior_1d.get(surface, 0.0), 4)
        for surface in surfaces
    }
    return {
        'comparison_design': 'single_agent_sota_1d_vs_openclaw_badclaw_dimensions',
        'single_agent_sota_1d_by_surface': prior_1d,
        'openclaw_badclaw_1d_by_surface': badclaw_1d,
        'openclaw_badclaw_1d_minus_single_agent_sota_by_surface': delta,
        'openclaw_badclaw_2d_asr': openclaw.get('badclaw_2d_asr', 0.0),
        'openclaw_badclaw_3d_asr': openclaw.get('badclaw_3d_asr', 0.0),
        'openclaw_badclaw_dimension_table': openclaw.get('badclaw_attack_success_by_dimension_count', {}),
        'single_agent_sota_method_table': baseline.get('attack_success_by_method', {}),
        'openclaw_badclaw_method_table': openclaw.get('attack_success_by_method', {}),
    }


def safe_filename(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('_') or 'model'


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='badclaw_tasks_v2.json')
    p.add_argument('--generate_if_missing', action='store_true')
    p.add_argument('--regenerate_dataset', action='store_true')
    p.add_argument('--max_tasks', type=int, default=0)
    p.add_argument('--tasks_per_family', type=int, default=60)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--skill_source', choices=['openclaw', 'synthetic'], default='openclaw')
    p.add_argument('--catalog_path', type=str, default='openclaw_skill_catalog.json')
    p.add_argument('--curated_catalog_path', type=str, default='curated_openclaw_skills.json')
    p.add_argument('--skills_per_family', type=int, default=80)
    p.add_argument('--bootstrap_skills', action='store_true')
    p.add_argument('--refresh_skill_catalog', action='store_true')
    p.add_argument('--models', nargs='+', default=['echo'])
    p.add_argument('--platforms', nargs='+', default=['openclaw_like', 'baseline_single_agent'])
    p.add_argument('--defenses', nargs='+', default=['none'])
    p.add_argument('--outdir', type=str, default='results')
    p.add_argument('--execution_mode', choices=['real', 'simulated'], default='real')
    p.add_argument('--save_task_records', action='store_true')
    args = p.parse_args()

    ds = Path(args.dataset)
    need_catalog = args.bootstrap_skills or args.refresh_skill_catalog or (
        args.skill_source == 'openclaw' and (not Path(args.catalog_path).exists() or not Path(args.curated_catalog_path).exists())
    )
    if need_catalog:
        curated = ensure_curated_catalog(
            curated_path=args.curated_catalog_path,
            catalog_path=args.catalog_path,
            target_per_family=args.skills_per_family,
            seed=args.seed,
            refresh_catalog=args.refresh_skill_catalog or args.bootstrap_skills,
            force_recurate=args.refresh_skill_catalog or args.bootstrap_skills,
        )
        print(f'[CATALOG] Curated {len(curated)} local OpenClaw skill records')

    if args.regenerate_dataset or (args.generate_if_missing and not ds.exists()):
        print(f'[DATA] Generating dataset -> {ds}')
        if args.skill_source == 'openclaw':
            generate_dimensional_dataset_from_curated_catalog(
                dataset_out=str(ds),
                curated_catalog_path=args.curated_catalog_path,
                tasks_per_family=args.tasks_per_family,
                seed=args.seed,
            )
        else:
            generate_dataset(str(ds), tasks_per_family=args.tasks_per_family, seed=args.seed)

    print(f'[DATA] Loading tasks from {ds}')
    tasks = load_tasks(str(ds))
    print(f'[DATA] Loaded {len(tasks)} tasks before platform filtering')
    print(f'[DATA] Full slice mix: {describe_slice(tasks)}')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for model_name in tqdm(args.models, desc='Models'):
        responder = create_model(model_name)
        model_slug = safe_filename(model_name)
        summary[model_name] = {}
        for defense_name in tqdm(args.defenses, desc='Defenses', leave=False):
            defense = get_defense(defense_name)
            summary[model_name][defense_name] = {}
            for platform_name in tqdm(args.platforms, desc=f'Platforms [{defense_name}]', leave=False):
                platform = make_platform(platform_name, responder, defense, args.execution_mode)
                platform_tasks = tasks_for_platform(tasks, platform_name)
                if args.max_tasks and args.max_tasks > 0:
                    platform_tasks = stratified_limit(platform_tasks, args.max_tasks, seed=args.seed)
                print(f'\n[DATA] [{platform_name}] using {len(platform_tasks)} applicable tasks; slice mix: {describe_slice(platform_tasks)}')
                results = []
                for task in tqdm(platform_tasks, desc=f'Tasks [{model_name} | {platform_name} | {defense_name}]', leave=False):
                    results.append(platform.run_task(task))
                metrics = score(platform_tasks, results)
                summary[model_name][defense_name][platform_name] = metrics
                save_path = outdir / f"{model_slug}__{platform_name}__{defense_name}.json"
                write_json(save_path, metrics)
                if args.save_task_records:
                    records_path = outdir / f"{model_slug}__{platform_name}__{defense_name}.tasks.jsonl"
                    records_path.parent.mkdir(parents=True, exist_ok=True)
                    with records_path.open('w', encoding='utf-8') as fh:
                        for task, result in zip(platform_tasks, results):
                            fh.write(json.dumps({
                                'task_id': task.task_id,
                                'category': task.category,
                                'comparison_group': task.comparison_group,
                                'attack_method': task.attack_method,
                                'attack_surface': task.attack_surface,
                                'input_text': task.input_text,
                                'planned_actions': [asdict(a) for a in result.planned_actions],
                                'raw_model_output': result.raw_model_output,
                                'diagnostics': result.diagnostics,
                                'trace': [asdict(t) for t in result.trace],
                            }, ensure_ascii=False) + '\n')
                paper_view = {
                    'comparison_1d_by_surface': metrics.get('comparison_1d_by_surface', {}),
                    'badclaw_1d_minus_prior_by_surface': metrics.get('badclaw_1d_minus_prior_by_surface', {}),
                    'badclaw_2d_asr': metrics.get('badclaw_2d_asr', 0.0),
                    'badclaw_3d_asr': metrics.get('badclaw_3d_asr', 0.0),
                    'mean_attack_execution_harm': metrics.get('mean_attack_execution_harm', 0.0),
                    'mean_attack_harm_score': metrics.get('mean_attack_harm_score', 0.0),
                    'total_attack_harm_score': metrics.get('total_attack_harm_score', 0),
                    'max_attack_harm_score': metrics.get('max_attack_harm_score', 0),
                    'mean_attack_stealthiness': metrics.get('mean_attack_stealthiness', 0.0),
                    'surface_table_all': metrics.get('attack_success_by_group_surface', {}),
                    'method_table': metrics.get('attack_success_by_method', {}),
                    'group_table': metrics.get('attack_success_by_comparison_group', {}),
                    'badclaw_dimension_table': metrics.get('badclaw_attack_success_by_dimension_count', {}),
                    'badclaw_combo_table': metrics.get('badclaw_attack_success_by_dimension_combo', {}),
                }
                paper_path = outdir / f"{model_slug}__{platform_name}__{defense_name}.paper.json"
                write_json(paper_path, paper_view)
                print(f'\n[{model_name}] [{platform_name}] [{defense_name}] saved -> {save_path}')
            comparison_view = cross_platform_paper_view(summary[model_name][defense_name])
            comparison_path = outdir / f"{model_slug}__{defense_name}.paper_comparison.json"
            write_json(comparison_path, comparison_view)

    summary_path = outdir / 'summary.json'
    write_json(summary_path, summary)
    print(f'\n[DONE] Summary saved -> {summary_path}')


if __name__ == '__main__':
    main()
