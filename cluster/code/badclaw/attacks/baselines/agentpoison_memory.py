from __future__ import annotations

from badclaw.attacks.baselines.base import PriorBaselineAdapter
from badclaw.core.types import TaskSpec


class AgentPoisonMemoryBaseline(PriorBaselineAdapter):
    name = "agentpoison"
    surface = "memory"
    trace_label = "AgentPoison-inspired stylized 1D"

    def should_consider(self, task: TaskSpec, text: str, state) -> bool:
        if task.attack_method != self.name or task.attack_surface != self.surface:
            return False
        token = str(task.metadata.get("single_trigger_token", "")).strip().lower()
        memory_ready = bool(task.metadata.get("poisoned_memory_entry", True))
        return bool(task.should_trigger and memory_ready and token and token in text.lower())

    def activation_score(self, task: TaskSpec, platform_name: str) -> float:
        return 0.80 if platform_name == "openclaw_like" else 0.56

    def action_mode(self) -> str:
        return "memory_append"
