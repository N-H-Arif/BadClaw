from __future__ import annotations

from badclaw.attacks.baselines.base import PriorBaselineAdapter
from badclaw.attacks.engine import prior_memory_score
from badclaw.core.types import TaskSpec


class AgentPoisonMemoryBaseline(PriorBaselineAdapter):
    name = "agentpoison"
    surface = "memory"
    trace_label = "AgentPoison-style 1D"

    def should_consider(self, task: TaskSpec, text: str, state) -> bool:
        if task.attack_method != self.name or task.attack_surface != self.surface:
            return False
        token = str(task.metadata.get("single_trigger_token", "")).strip().lower()
        memory_ready = bool(task.metadata.get("poisoned_memory_entry", True))
        return bool(task.should_trigger and memory_ready and token and token in text.lower())

    def activation_score(self, task: TaskSpec, text: str, state, platform_name: str) -> tuple[float, dict]:
        s = prior_memory_score(task, text, state, platform_name)
        return s.score, s.details

    def action_mode(self) -> str:
        return "memory_append"
