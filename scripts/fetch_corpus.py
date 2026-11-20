"""Fetch a slice of the public GitLab handbook as the "real documents" half of
the corpus.

Why this source: it is a real, messy, version-controlled HR knowledge base with
mixed heading depth, tables, and cross-links - the things that actually break
chunking. The synthetic Majo documents cover the Indonesia-specific policies
that GitLab (a US-remote company) does not.

Usage:
    python -m scripts.fetch_corpus --limit 70
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://gitlab.com/api/v4/projects/gitlab-com%2Fcontent-sites%2Fhandbook/repository"
RAW = "https://gitlab.com/gitlab-com/content-sites/handbook/-/raw/main"
ROOT_PATHS = [
    "content/handbook/people-group",
    "content/handbook/total-rewards",
]
OUT = Path(__file__).resolve().parent.parent / "data" / "docs" / "gitlab-handbook"
SKIP = re.compile(
    r"(index|_index|README|\.gitkeep)$", re.IGNORECASE
)


def _get_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "majo-takehome/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_markdown(limit: int) -> list[str]:
    found: list[str] = []
    for root in ROOT_PATHS:
        page = 1
        while len(found) < limit and page <= 10:
            query = urllib.parse.urlencode(
                {"path": root, "recursive": "true", "per_page": 100, "page": page}
            )
            try:
                tree = _get_json(f"{API}/tree?{query}")
            except Exception as exc:  # noqa: BLE001 - network best effort
                print(f"  tree listing failed for {root}: {exc}", file=sys.stderr)
                break
            if not tree:
                break
            for entry in tree:
                if entry.get("type") != "blob":
                    continue
                name = entry["name"]
                if not name.endswith(".md") or SKIP.search(name.rsplit(".", 1)[0]):
                    continue
                if entry["path"] not in found:
                    found.append(entry["path"])
                if len(found) >= limit:
                    break
            page += 1
    return found[:limit]


def fetch(path: str) -> str | None:
    url = f"{RAW}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "majo-takehome/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        print(f"  skip {path}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=70)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    paths = list_markdown(args.limit)
    print(f"discovered {len(paths)} markdown files")

    saved = 0
    for path in paths:
        text = fetch(path)
        if not text or len(text.strip()) < 400:
            continue
        rel = path.split("content/handbook/", 1)[-1]
        target = OUT / (rel if rel.endswith(".md") else f"{rel}.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        saved += 1
        time.sleep(0.05)

    print(f"saved {saved} documents to {OUT}")


if __name__ == "__main__":
    main()
