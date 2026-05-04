from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def render_skill_md(skill_name: str, skill_category: str, skill_slug: str) -> str:
    return f"""# {skill_name}

Category: {skill_category}
Benchmark Slug: {skill_slug}



## When to Use
Use this skill when a benchmark task references `{skill_name}`.

"""


def export_workspace_skills(dataset_json: str, workspace_dir: str) -> None:
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    skills_dir = Path(workspace_dir) / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    count = 0

    for row in data:
        key = (row["skill_slug"], row["skill_name"], row["skill_category"])
        if key in seen:
            continue
        seen.add(key)

        skill_slug, skill_name, skill_category = key
        dst = skills_dir / skill_slug
        dst.mkdir(parents=True, exist_ok=True)

        (dst / "SKILL.md").write_text(
            render_skill_md(skill_name, skill_category, skill_slug),
            encoding="utf-8",
        )
        count += 1

    print(f"[OPENCLAW] Exported {count} inert workspace skills to {skills_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_json", type=str, required=True)
    p.add_argument("--workspace_dir", type=str, required=True)
    args = p.parse_args()
    export_workspace_skills(args.dataset_json, args.workspace_dir)