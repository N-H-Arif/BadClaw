from __future__ import annotations

import json
from typing import Dict, List

from badclaw.core.types import PlannedAction, TaskSpec
from badclaw.models.base import ModelResponder
from badclaw.models.planning import parse_planner_output, parse_route


class EchoResponder(ModelResponder):
    def plan_actions(
        self,
        *,
        task: TaskSpec,
        platform_name: str,
        selected_route: str,
        allowed_tools: List[str],
        context: str,
    ) -> tuple[str, List[PlannedAction], str]:
        if task.expected_tool and task.expected_tool in allowed_tools:
            payload = {"route": selected_route, "actions": [{"name": task.expected_tool, "args": {}}]}
        else:
            payload = {"route": selected_route, "actions": [{"name": "mock_summarize", "args": {"text": task.input_text[:60]}}]}
        raw = json.dumps(payload)
        return parse_route(raw, selected_route), parse_planner_output(raw, task=task, allowed_tools=allowed_tools), raw

    def respond(
        self,
        *,
        user_text: str,
        selected_route: str,
        actions: List[PlannedAction],
        triggered: Dict[str, bool],
        rendered_output: str,
    ) -> str:
        return (
            f"Benchmark result. Route={selected_route}. "
            f"Actions={[a.name for a in actions]}. Triggered={triggered}. "
        )
