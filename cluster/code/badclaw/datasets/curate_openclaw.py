from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from badclaw.datasets.openclaw_catalog import SkillRecord, build_openclaw_catalog, load_openclaw_catalog


@dataclass
class CuratedSkillRecord:
    category_file: str
    category_name: str
    skill_name: str
    skill_slug: str
    source_url: str
    description: str
    attack_family: str
    expected_tool: str
    expected_route: str
    score: int
    matched_keywords: List[str]


FAMILY_KEYWORDS: Dict[str, Sequence[str]] = {
    "agent_memory": (
        "memory", "remember", "notes", "note", "notebook", "knowledge", "knowledge base",
        "obsidian", "pkm", "archive", "journal", "history", "context", "store", "retrieve",
        "workspace", "crm", "contacts", "vault", "document", "docs"
    ),
    "tool_context": (
        "slack", "gmail", "email", "calendar", "browser", "automation", "github", "git",
        "workflow", "api", "cli", "deploy", "cloud", "scrape", "search", "web", "pdf",
        "spreadsheet", "database", "ticket", "message", "send", "tool", "integration"
    ),
    "multi_turn_interaction": (
        "assistant", "planner", "project", "meeting", "trip", "travel", "research", "support",
        "coach", "tutor", "manager", "agent", "orchestr", "collabor", "review", "draft",
        "customer", "conversation", "follow-up", "follow up", "workflow"
    ),
}

ROUTE_TOOL_HINTS: List[tuple[Sequence[str], str, str]] = [
    (("calendar", "schedule", "meeting"), "scheduler", "mock_calendar_lookup"),
    (("gmail", "email", "slack", "message", "chat"), "communicator", "mock_send_message_preview"),
    (("github", "git", "repo", "pull request"), "coder", "mock_repo_lookup"),
    (("browser", "web", "scrape", "search", "research"), "research", "mock_web_lookup"),
    (("pdf", "docs", "document", "knowledge", "obsidian", "notes", "note"), "research", "mock_doc_search"),
    (("task", "todo", "planner", "project"), "planner", "mock_task_lookup"),
    (("security", "password", "auth"), "security", "mock_security_lookup"),
    (("cloud", "deploy", "devops"), "infra", "mock_doc_search"),
]
DEFAULT_ROUTE = "research"
DEFAULT_TOOL = "mock_doc_search"


_DEFENSE_BLOCKLIST = {
    "crypto", "blockchain", "trade", "trading", "stock", "forex", "betting", "casino", "wallet"
}


def _text(rec: SkillRecord) -> str:
    return f"{rec.skill_name} {rec.description} {rec.category_name}".lower()


def _score_family(rec: SkillRecord, family: str) -> tuple[int, List[str]]:
    text = _text(rec)
    hits: List[str] = []
    for kw in FAMILY_KEYWORDS[family]:
        if kw in text:
            hits.append(kw)
    score = len(hits)

    # category priors
    cat = rec.category_name.lower()
    if family == "agent_memory" and any(x in cat for x in ["notes", "documents", "research", "productivity"]):
        score += 2
    if family == "tool_context" and any(x in cat for x in ["communication", "automation", "devops", "calendar", "git", "browser"]):
        score += 2
    if family == "multi_turn_interaction" and any(x in cat for x in ["productivity", "communication", "research", "calendar"]):
        score += 2
    return score, sorted(set(hits))


def _guess_route_and_tool(rec: SkillRecord) -> tuple[str, str]:
    text = _text(rec)
    for kws, route, tool in ROUTE_TOOL_HINTS:
        if any(kw in text for kw in kws):
            return route, tool
    return DEFAULT_ROUTE, DEFAULT_TOOL


def _blocked(rec: SkillRecord) -> bool:
    text = _text(rec)
    return any(kw in text for kw in _DEFENSE_BLOCKLIST)


def _dedupe_curated(records: Iterable[CuratedSkillRecord]) -> List[CuratedSkillRecord]:
    seen: Set[str] = set()
    out: List[CuratedSkillRecord] = []
    for rec in records:
        key = rec.source_url.strip().lower() or rec.skill_slug.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def curate_catalog(
    catalog_path: str,
    out_json: str = "curated_openclaw_skills.json",
    target_per_family: int = 80,
    seed: int = 13,
) -> List[CuratedSkillRecord]:
    rng = random.Random(seed)
    records = load_openclaw_catalog(catalog_path)

    pools: Dict[str, List[CuratedSkillRecord]] = defaultdict(list)
    for rec in records:
        if _blocked(rec):
            continue
        best_family = None
        best_score = -1
        best_hits: List[str] = []
        for family in FAMILY_KEYWORDS:
            score, hits = _score_family(rec, family)
            if score > best_score:
                best_family = family
                best_score = score
                best_hits = hits
        if not best_family or best_score <= 0:
            continue
        route, tool = _guess_route_and_tool(rec)
        pools[best_family].append(
            CuratedSkillRecord(
                category_file=rec.category_file,
                category_name=rec.category_name,
                skill_name=rec.skill_name,
                skill_slug=rec.skill_slug,
                source_url=rec.source_url,
                description=rec.description,
                attack_family=best_family,
                expected_tool=tool,
                expected_route=route,
                score=best_score,
                matched_keywords=best_hits,
            )
        )

    selected: List[CuratedSkillRecord] = []
    for family, family_pool in pools.items():
        by_cat: Dict[str, List[CuratedSkillRecord]] = defaultdict(list)
        for rec in family_pool:
            by_cat[rec.category_name].append(rec)
        for cat in by_cat:
            by_cat[cat].sort(key=lambda r: (-r.score, r.skill_name.lower()))
        cats = list(by_cat.keys())
        rng.shuffle(cats)

        family_selected: List[CuratedSkillRecord] = []
        per_cat_cap = max(3, target_per_family // max(1, min(12, len(cats))))
        while len(family_selected) < target_per_family and cats:
            progressed = False
            for cat in list(cats):
                if len(family_selected) >= target_per_family:
                    break
                cat_taken = sum(1 for r in family_selected if r.category_name == cat)
                if by_cat[cat] and cat_taken < per_cat_cap:
                    family_selected.append(by_cat[cat].pop(0))
                    progressed = True
                if not by_cat[cat]:
                    cats.remove(cat)
            if not progressed:
                break

        if len(family_selected) < target_per_family:
            leftovers = []
            for recs in by_cat.values():
                leftovers.extend(recs)
            leftovers.sort(key=lambda r: (-r.score, r.skill_name.lower()))
            need = target_per_family - len(family_selected)
            family_selected.extend(leftovers[:need])

        selected.extend(family_selected[:target_per_family])

    selected = _dedupe_curated(selected)
    selected.sort(key=lambda r: (r.attack_family, r.category_name, -r.score, r.skill_name.lower()))
    Path(out_json).write_text(json.dumps([asdict(r) for r in selected], indent=2, ensure_ascii=False), encoding="utf-8")
    return selected


def ensure_curated_catalog(
    curated_path: str = "curated_openclaw_skills.json",
    catalog_path: str = "openclaw_skill_catalog.json",
    target_per_family: int = 80,
    seed: int = 13,
    refresh_catalog: bool = False,
    force_recurate: bool = False,
) -> List[CuratedSkillRecord]:
    catalog_p = Path(catalog_path)
    curated_p = Path(curated_path)
    if refresh_catalog or not catalog_p.exists():
        build_openclaw_catalog(out_json=str(catalog_p))
    if force_recurate or not curated_p.exists():
        return curate_catalog(
            catalog_path=str(catalog_p),
            out_json=str(curated_p),
            target_per_family=target_per_family,
            seed=seed,
        )
    data = json.loads(curated_p.read_text(encoding="utf-8"))
    rows = [CuratedSkillRecord(**row) for row in data]
    rows = _dedupe_curated(rows)
    if len(rows) != len(data):
        curated_p.write_text(json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


if __name__ == "__main__":
    ensure_curated_catalog()
