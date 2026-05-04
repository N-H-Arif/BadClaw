from __future__ import annotations

from badclaw.attacks.baselines.base import PriorBaselineAdapter
from badclaw.attacks.engine import prior_planner_score
from badclaw.core.types import TaskSpec


class ASBPoTPlannerBaseline(PriorBaselineAdapter):
    name = "asb_pot"
    surface = "planner"
    trace_label = "ASB-PoT-style 1D"

    def should_consider(self, task: TaskSpec, text: str, state) -> bool:
        if task.attack_method != self.name or task.attack_surface != self.surface:
            return False
        phrase = str(task.metadata.get("planner_trigger_phrase", "")).strip().lower()
        return bool(task.should_trigger and phrase and phrase in text.lower())

    def activation_score(self, task: TaskSpec, text: str, state, platform_name: str) -> tuple[float, dict]:
        s = prior_planner_score(task, text, state, platform_name)
        return s.score, s.details

    def action_mode(self) -> str:
        return "replace_action"
