#!/usr/bin/env python3
"""Write the files from a guide chapter into your repo.

Every file in the guide is introduced by a caption line of the form

    File: `path/relative/to/repo`

directly followed by a fenced code block. This script finds those pairs and writes them.

Usage (run from the repo root):
    python3 ../research-agent-guide/extract_code.py ../research-agent-guide/03-*.md            # dry run
    python3 ../research-agent-guide/extract_code.py ../research-agent-guide/03-*.md --write    # create new files
    python3 ../research-agent-guide/extract_code.py ../research-agent-guide/04-*.md --write --force  # also replace existing

Nothing is written without --write. Existing files are only replaced with --force, and the dry run
tells you which files a chapter changes, so you can see what --force would overwrite.
"""
import argparse
import re
import sys
from pathlib import Path

CAPTION = re.compile(r"^File: `([^`]+)`\s*$")
FENCE = re.compile(r"^(```+)")


def extract(markdown: str):
    lines = markdown.splitlines()
    i, files = 0, []
    while i < len(lines):
        m = CAPTION.match(lines[i])
        if not m:
            i += 1
            continue
        path = m.group(1)
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        fence = FENCE.match(lines[j]) if j < len(lines) else None
        if not fence:
            sys.exit(f"caption for {path} is not followed by a code block (line {i + 1})")
        ticks = fence.group(1)
        body, k = [], j + 1
        while k < len(lines) and lines[k].rstrip() != ticks:
            body.append(lines[k])
            k += 1
        if k == len(lines):
            sys.exit(f"unterminated code block for {path}")
        files.append((path, "\n".join(body) + "\n"))
        i = k + 1
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chapters", nargs="+", type=Path)
    ap.add_argument("--write", action="store_true", help="write files (default: dry run)")
    ap.add_argument("--force", action="store_true", help="replace files that already exist with different content")
    args = ap.parse_args()

    root = Path.cwd()
    blocked = False
    for chapter in args.chapters:
        print(f"== {chapter.name}")
        for rel, content in extract(chapter.read_text()):
            if rel.startswith("/") or ".." in Path(rel).parts:
                sys.exit(f"refusing unsafe path {rel}")
            target = root / rel
            if not target.exists():
                status = "NEW"
            elif target.read_text() == content:
                status = "SAME"
            else:
                status = "CHANGED"
            action = ""
            if args.write and (status == "NEW" or (status == "CHANGED" and args.force)):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                if target.suffix in (".sh", ".py") and (rel.startswith("scripts/") or content.startswith("#!")):
                    target.chmod(0o755)
                action = "written"
            elif args.write and status == "CHANGED":
                action = "skipped (use --force to replace)"
                blocked = True
            print(f"  {status:8} {rel}  {action}")
    if not args.write:
        print("\nDry run only. Add --write to create files (and --force to replace CHANGED ones).")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
