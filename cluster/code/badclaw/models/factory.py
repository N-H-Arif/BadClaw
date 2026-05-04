from __future__ import annotations

from badclaw.models.base import ModelResponder
from badclaw.models.echo import EchoResponder


def create_model(name: str) -> ModelResponder:
    if name == "echo":
        return EchoResponder()
    from badclaw.models.hf_local import LocalHFResponder
    return LocalHFResponder(model_name=name)
