from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalMessage:
    role: str
    content: str
    channel: str = "chat"


@dataclass
class PlannedAction:
    name: str
    args: Dict[str, Any]
    harm: int = 0
    mock_only: bool = True
    origin: str = "benign"  # benign | attack


@dataclass
class EvalTrace:
    stage: str
    data: Dict[str, Any]


@dataclass
class EvalResult:
    platform: str
    selected_route: str
    planned_actions: List[PlannedAction] = field(default_factory=list)
    triggered: Dict[str, bool] = field(default_factory=dict)
    trace: List[EvalTrace] = field(default_factory=list)
    rendered_output: str = ""
    raw_model_output: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    task_id: str
    name: str
    category: str
    trigger_mechanism: str
    attack_surface: Optional[str]
    sequence_role: str
    seed_group: Optional[str]
    input_text: str
    messages: List[EvalMessage]
    expected_tool: Optional[str] = None
    expected_route: Optional[str] = None
    expected_payload: Optional[str] = None
    expected_harm_min: int = 0
    should_trigger: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    attack_method: str = "badclaw"      # badclaw | agentpoison | asb_pot | ama | benign
    comparison_group: str = "badclaw"   # prior_1d | badclaw_1d | badclaw_2d | badclaw_3d | clean | false_activation
