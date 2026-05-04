from badclaw.attacks.baselines.agentpoison_memory import AgentPoisonMemoryBaseline
from badclaw.attacks.baselines.asb_pot_planner import ASBPoTPlannerBaseline
from badclaw.attacks.baselines.ama_router import AMARouterBaseline

BASELINE_ATTACKS = {
    "agentpoison": AgentPoisonMemoryBaseline(),
    "asb_pot": ASBPoTPlannerBaseline(),
    "ama": AMARouterBaseline(),
}

__all__ = ["BASELINE_ATTACKS"]
