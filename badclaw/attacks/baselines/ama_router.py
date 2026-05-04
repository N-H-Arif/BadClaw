from __future__ import annotations

from badclaw.attacks.baselines.base import PriorBaselineAdapter
from badclaw.attacks.engine import prior_router_score
from badclaw.core.types import TaskSpec


class AMARouterBaseline(PriorBaselineAdapter):
    name = "ama"
    surface = "router"
    trace_label = "AMA-style 1D"

    def should_consider(self, task: TaskSpec, text: str, state) -> bool:
        if task.attack_method != self.name or task.attack_surface != self.surface:
            return False
        score = float(task.metadata.get("metadata_attractiveness", 0.0))
        bait = str(task.metadata.get("metadata_bait_phrase", "")).strip().lower()
        return bool(task.should_trigger and score >= 0.65 and bait and any(tok in text.lower() for tok in bait.split()))

    def activation_score(self, task: TaskSpec, text: str, state, platform_name: str) -> tuple[float, dict]:
        s = prior_router_score(task, text, state, platform_name)
        return s.score, s.details

    def action_mode(self) -> str:
        return "route_and_append"
