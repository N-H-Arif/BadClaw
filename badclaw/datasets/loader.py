from __future__ import annotations

import json
from pathlib import Path

from badclaw.core.types import EvalMessage, TaskSpec


DEFAULT_GROUP_BY_CATEGORY = {
    "clean": "clean",
    "false_activation": "false_activation",
    "control_attack": "badclaw_1d",
    "attack_sequence": "badclaw_1d",
}


def load_tasks(path: str) -> list[TaskSpec]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks: list[TaskSpec] = []

    for item in raw:
        msgs = [EvalMessage(**m) for m in item["messages"]]

        # Backward compatibility with older dataset schema
        category = item.get("category", "clean")
        trigger_mechanism = item.get("trigger_mechanism")
        attack_surface = item.get("attack_surface")
        sequence_role = item.get("sequence_role", "none")
        seed_group = item.get("seed_group")

        # Old schema fallback mapping
        if trigger_mechanism is None:
            old_trigger = item.get("trigger_type", "clean")

            if old_trigger == "clean":
                trigger_mechanism = "clean"
                attack_surface = None
                sequence_role = "none"

            elif old_trigger == "router":
                trigger_mechanism = "lexical"
                attack_surface = "router"
                category = "control_attack"

            elif old_trigger == "planner":
                trigger_mechanism = "lexical"
                attack_surface = "planner"
                category = "control_attack"

            elif old_trigger == "memory":
                trigger_mechanism = "lexical"
                attack_surface = "memory"
                category = "control_attack"

            elif old_trigger == "multi_stage_seed":
                trigger_mechanism = "multi_stage_latent"
                attack_surface = "memory"
                category = "attack_sequence"
                sequence_role = "seed"

            elif old_trigger == "multi_stage_fire":
                trigger_mechanism = "multi_stage_latent"
                attack_surface = "memory"
                category = "attack_sequence"
                sequence_role = "fire"

            elif old_trigger == "near_miss":
                trigger_mechanism = "near_miss"
                attack_surface = None
                category = "false_activation"

            else:
                trigger_mechanism = "clean"
                attack_surface = None
                sequence_role = "none"

        attack_method = item.get("attack_method")
        if attack_method is None:
            if category == "clean":
                attack_method = "benign"
            elif category == "false_activation":
                attack_method = "benign"
            else:
                attack_method = "badclaw"

        comparison_group = item.get("comparison_group") or DEFAULT_GROUP_BY_CATEGORY.get(category, "badclaw")

        tasks.append(
            TaskSpec(
                task_id=item["task_id"],
                name=item["name"],
                category=category,
                trigger_mechanism=trigger_mechanism,
                attack_surface=attack_surface,
                sequence_role=sequence_role,
                seed_group=seed_group,
                input_text=item["input_text"],
                messages=msgs,
                expected_tool=item.get("expected_tool"),
                expected_route=item.get("expected_route"),
                expected_payload=item.get("expected_payload"),
                expected_harm_min=item.get("expected_harm_min", 0),
                should_trigger=item.get("should_trigger", False),
                metadata=item.get("metadata", {}),
                attack_method=attack_method,
                comparison_group=comparison_group,
            )
        )

    return tasks
