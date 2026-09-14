#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/fim-current-evidence.md"
OUTPUT = ROOT / "output/pdf/fim-current-evidence.pdf"


def validated_source(path: Path = SOURCE) -> tuple[str, str]:
    source = path.read_text(encoding="utf-8")
    forbidden = ("todo", "tbd", "placeholder", "results pending")
    if any(marker in source.lower() for marker in forbidden):
        raise ValueError("paper contains a forbidden incomplete marker")
    lines = source.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("paper must begin with a level-one title")
    title = lines[0].removeprefix("# ").strip()
    body = "\n".join(lines[1:]).lstrip()
    if len(body.split()) < 2_000:
        raise ValueError("paper is too short for the evidence-preprint gate")
    return title, body


def compile_pdf(source: Path = SOURCE, output: Path = OUTPUT) -> dict[str, object]:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("pandoc is required to compile the paper")
    try:
        from weasyprint import CSS, HTML
    except ImportError as exc:
        raise RuntimeError("weasyprint is required to compile the paper") from exc
    title, body = validated_source(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fim-paper-") as directory:
        staged = Path(directory) / "paper.md"
        html = Path(directory) / "paper.html"
        staged.write_text(
            "---\n"
            f"title: {json.dumps(title)}\n"
            "author: Anonymous research artifact\n"
            "date: ''\n"
            "---\n\n" + body + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [pandoc, str(staged), "--from=gfm", "--standalone", "--output", str(html)],
            cwd=ROOT,
            check=True,
        )
        HTML(filename=str(html), base_url=str(ROOT / "paper")).write_pdf(
            str(output), stylesheets=[CSS(filename=str(ROOT / "paper/preprint.css"))]
        )
    payload = output.read_bytes()
    if not payload.startswith(b"%PDF-") or len(payload) < 20_000:
        raise RuntimeError("compiled output is not a nontrivial PDF")
    return {
        "pdf": str(output),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main() -> None:
    print(json.dumps(compile_pdf(), indent=2))


if __name__ == "__main__":
    main()
