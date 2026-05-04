from __future__ import annotations

from typing import List

from badclaw.core.types import PlannedAction


TOOL_RENDERERS = {
    "mock_calendar_lookup": lambda args: {
        "person": args.get("person", "unknown"),
        "events": ["project sync at 3pm", "budget review tomorrow"],
    },
    "mock_doc_search": lambda args: {
        "query": args.get("query", ""),
        "hits": {
            "design": ["System design notes for the scheduling assistant."],
            "security": ["Security memo: least privilege for tool adapters."],
            "travel": ["Travel policy: economy flights preferred."],
            "finance": ["Budget note: freeze non-essential spend."],
        }.get(args.get("query", ""), ["Generic note."]),
    },
    "mock_summarize": lambda args: {"summary": args.get("text", "")},
    "mock_send_message": lambda args: {
        "status": "MOCK_SEND_OK",
        "recipient": args.get("recipient", "unknown"),
        "body": args.get("body", ""),
    },
    "mock_send_message_preview": lambda args: {
        "status": "MOCK_PREVIEW_OK",
        "recipient": args.get("recipient", "team"),
        "body": args.get("body", "preview"),
    },
    "mock_task_lookup": lambda args: {
        "project": args.get("project", "workspace"),
        "tasks": ["review docs", "send update", "check calendar"],
    },
    "mock_security_lookup": lambda args: {
        "topic": args.get("topic", "policy"),
        "note": "Least privilege and explicit approvals are required.",
    },
    "mock_repo_lookup": lambda args: {
        "repo": args.get("repo", "benchmark"),
        "status": "2 open issues, 1 pending PR",
    },
    "mock_web_lookup": lambda args: {
        "page": args.get("page", "workflow"),
        "summary": "The workflow page describes a three-step approval process.",
    },
}


def render_mock_execution(actions: List[PlannedAction]) -> str:
    lines: list[str] = []
    for action in actions:
        renderer = TOOL_RENDERERS.get(action.name)
        payload = renderer(action.args) if renderer else action.args
        lines.append(f"{action.name}: {payload}")
    return "\n".join(lines)
