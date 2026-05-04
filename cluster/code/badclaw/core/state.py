from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SessionState:
    flags: Dict[str, bool] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    observations: List[Dict[str, Any]] = field(default_factory=list)

    def set_flag(self, key: str, value: bool = True) -> None:
        self.flags[key] = value

    def get_flag(self, key: str) -> bool:
        return bool(self.flags.get(key, False))

    def put(self, key: str, value: Any) -> None:
        self.memory[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.memory.get(key, default)

    def add_observation(self, *, kind: str, data: Dict[str, Any]) -> None:
        self.observations.append({"kind": kind, "data": data})

    def seen_observation(self, kind: str, key: str, value: Any) -> bool:
        for row in self.observations:
            if row.get("kind") != kind:
                continue
            if row.get("data", {}).get(key) == value:
                return True
        return False
