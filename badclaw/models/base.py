from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from badclaw.core.types import PlannedAction, TaskSpec


class ModelResponder(ABC):
    @abstractmethod
    def plan_actions(
        self,
        *,
        task: TaskSpec,
        platform_name: str,
        selected_route: str,
        allowed_tools: List[str],
        context: str,
    ) -> tuple[str, List[PlannedAction], str]:
        """Return route, planned actions, and raw planner text from the model."""
        raise NotImplementedError

    @abstractmethod
    def respond(
        self,
        *,
        user_text: str,
        selected_route: str,
        actions: List[PlannedAction],
        triggered: Dict[str, bool],
        rendered_output: str,
    ) -> str:
        raise NotImplementedError
