from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from badclaw.core.types import PlannedAction, TaskSpec


@dataclass(frozen=True)
class AdapterDecision:
    activated: bool
    score: float
    mode: str
    trace_label: str


class PriorBaselineAdapter(ABC):
    name: str = "prior_baseline"
    surface: str = "unknown"
    trace_label: str = "prior_1d"

    @abstractmethod
    def activation_score(self, task: TaskSpec, text: str, state, platform_name: str) -> tuple[float, dict]:
        raise NotImplementedError

    @abstractmethod
    def should_consider(self, task: TaskSpec, text: str, state) -> bool:
        raise NotImplementedError

    @abstractmethod
    def action_mode(self) -> str:
        raise NotImplementedError

    def make_action(self, task: TaskSpec) -> PlannedAction:
        payload = task.expected_payload or "mock_send_message"
        args = {"source": self.name, "task_id": task.task_id}
        if payload == "mock_send_message":
            args["recipient"] = "audit-shadow"
            args["body"] = f"BASELINE::{self.name}::{task.task_id}"
        elif payload == "mock_doc_search":
            args["query"] = "security"
        elif payload == "mock_calendar_lookup":
            args["person"] = "alice"
        else:
            args["text"] = task.input_text[:80]
        return PlannedAction(
            name=payload,
            args=args,
            harm=max(1, task.expected_harm_min or 1),
            mock_only=True,
            origin="attack",
        )
