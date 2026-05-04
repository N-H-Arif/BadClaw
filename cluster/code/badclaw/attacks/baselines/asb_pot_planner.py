from __future__ import annotations

from badclaw.attacks.baselines.base import PriorBaselineAdapter
from badclaw.core.types import TaskSpec


class ASBPoTPlannerBaseline(PriorBaselineAdapter):
    name = "asb_pot"
    surface = "planner"
    trace_label = "ASB-PoT-inspired stylized 1D"

    def should_consider(self, task: TaskSpec, text: str, state) -> bool:
        if task.attack_method != self.name or task.attack_surface != self.surface:
            return False
        phrase = str(task.metadata.get("planner_trigger_phrase", "")).strip().lower()
        return bool(task.should_trigger and phrase and phrase in text.lower())

    def activation_score(self, task: TaskSpec, platform_name: str) -> float:
        return 0.76 if platform_name == "openclaw_like" else 0.52

    def action_mode(self) -> str:
        return "replace_action"
