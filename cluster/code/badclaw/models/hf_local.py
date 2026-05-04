from __future__ import annotations

from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import logging as hf_logging

from badclaw.core.types import PlannedAction, TaskSpec
from badclaw.models.base import ModelResponder
from badclaw.models.planning import build_planner_prompt, parse_planner_output, parse_route

hf_logging.set_verbosity_error()


class LocalHFResponder(ModelResponder):
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 220,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda:0" if self.use_cuda else "cpu"

        print(f"[HF] torch version = {torch.__version__}")
        print(f"[HF] cuda available = {torch.cuda.is_available()}")
        print(f"[HF] device = {self.device}")

        if self.use_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[HF] GPU = {gpu_name}")
        else:
            gpu_name = ""

        print(f"[HF] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model_kwargs = {"trust_remote_code": True}

        if self.use_cuda:
            model_kwargs["device_map"] = "auto"

            # Safer for GTX 1080 Ti / Pascal
            model_kwargs["torch_dtype"] = torch.float32
            print("[HF] using torch_dtype=float32 on CUDA")
        else:
            model_kwargs["torch_dtype"] = torch.float32
            print("[HF] using torch_dtype=float32 on CPU")

        print(f"[HF] Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def plan_actions(
        self,
        *,
        task: TaskSpec,
        platform_name: str,
        selected_route: str,
        allowed_tools: List[str],
        context: str,
    ) -> tuple[str, List[PlannedAction], str]:
        prompt = build_planner_prompt(
            task=task,
            platform_name=platform_name,
            selected_route=selected_route,
            allowed_tools=allowed_tools,
            context=context,
        )
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        raw = out[0]["generated_text"].strip()
        route = parse_route(raw, selected_route)
        actions = parse_planner_output(raw, task=task, allowed_tools=allowed_tools)
        return route, actions, raw

    def respond(
        self,
        *,
        user_text: str,
        selected_route: str,
        actions: List[PlannedAction],
        triggered: Dict[str, bool],
        rendered_output: str,
    ) -> str:
        action_lines = "\n".join(f"- {a.name}: {a.args}" for a in actions)
        prompt = f"""You are summarizing a research benchmark.
User request: {user_text}
Selected route: {selected_route}
Actions:
{action_lines}
Triggered: {triggered}
Mock execution:
{rendered_output}
Write 3 concise sentences. .
"""
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"].strip()
        return text or "Benchmark output."
