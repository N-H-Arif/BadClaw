from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Iterable
from urllib.request import urlopen, Request


REPO_RAW_BASE = "https://raw.githubusercontent.com/VoltAgent/awesome-openclaw-skills/main/categories"

CATEGORY_FILES = [
    "ai-and-llms.md",
    "apple-apps-and-services.md",
    "browser-and-automation.md",
    "calendar-and-scheduling.md",
    "clawdbot-tools.md",
    "cli-utilities.md",
    "coding-agents-and-ides.md",
    "communication.md",
    "data-and-analytics.md",
    "devops-and-cloud.md",
    "gaming.md",
    "git-and-github.md",
    "health-and-fitness.md",
    "image-and-video-generation.md",
    "ios-and-macos-development.md",
    "marketing-and-sales.md",
    "media-and-streaming.md",
    "moltbook.md",
    "notes-and-pkm.md",
    "pdf-and-documents.md",
    "personal-development.md",
    "productivity-and-tasks.md",
    "search-and-research.md",
    "security-and-passwords.md",
    "self-hosted-and-automation.md",
    "shopping-and-e-commerce.md",
    "smart-home-and-iot.md",
    "speech-and-transcription.md",
    "transportation.md",
    "web-and-frontend-development.md",
]


@dataclass
class SkillRecord:
    category_file: str
    category_name: str
    skill_name: str
    skill_slug: str
    source_url: str
    description: str


def _fetch_text(url: str, timeout: int = 30) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": "badclaw-benchmark/1.0",
            "Accept": "text/plain, text/markdown;q=0.9, */*;q=0.8",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _category_name_from_file(name: str) -> str:
    stem = name.removesuffix(".md")
    return stem.replace("-", " ")


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def _parse_markdown_skill_lines(md_text: str, category_file: str) -> List[SkillRecord]:
    """
    Expected line pattern, commonly found in the awesome list:
      * [skill-name](https://clawskills.sh/...) - description
    or
      - [skill-name](...) - description
    """
    out: List[SkillRecord] = []
    category_name = _category_name_from_file(category_file)

    bullet_pat = re.compile(
        r"^\s*[*-]\s+\[([^\]]+)\]\((https?://[^\)]+)\)\s*(?:-\s*(.*))?\s*$"
    )

    for raw_line in md_text.splitlines():
        m = bullet_pat.match(raw_line)
        if not m:
            continue

        skill_name = m.group(1).strip()
        source_url = m.group(2).strip()
        description = (m.group(3) or "").strip()

        slug = _slugify(skill_name)
        if not slug:
            continue

        out.append(
            SkillRecord(
                category_file=category_file,
                category_name=category_name,
                skill_name=skill_name,
                skill_slug=slug,
                source_url=source_url,
                description=description,
            )
        )

    return out


def build_openclaw_catalog(
    out_json: str = "openclaw_skill_catalog.json",
    sleep_s: float = 0.1,
) -> List[SkillRecord]:
    """
    Build a de-duplicated metadata catalog from awesome-openclaw-skills categories.
    Deduplication is by skill_slug first, then by normalized name.
    """
    all_records: List[SkillRecord] = []

    for category_file in CATEGORY_FILES:
        url = f"{REPO_RAW_BASE}/{category_file}"
        try:
            text = _fetch_text(url)
        except Exception as e:
            print(f"[WARN] Failed to fetch {url}: {e}")
            continue

        parsed = _parse_markdown_skill_lines(text, category_file)
        all_records.extend(parsed)
        time.sleep(sleep_s)

    unique_by_slug: Dict[str, SkillRecord] = {}
    unique_by_name: Dict[str, SkillRecord] = {}

    for rec in all_records:
        norm_name = rec.skill_name.strip().lower()
        if rec.skill_slug in unique_by_slug:
            continue
        if norm_name in unique_by_name:
            continue
        unique_by_slug[rec.skill_slug] = rec
        unique_by_name[norm_name] = rec

    records = list(unique_by_slug.values())
    records.sort(key=lambda r: (r.category_name, r.skill_name.lower()))

    out_path = Path(out_json)
    out_path.write_text(
        json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[CATALOG] Wrote {len(records)} unique skill records to {out_path}")
    return records


def load_openclaw_catalog(path: str) -> List[SkillRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SkillRecord(**row) for row in data]


if __name__ == "__main__":
    build_openclaw_catalog()