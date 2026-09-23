#!/usr/bin/env python
"""Upload a built Space folder to Hugging Face.

Two bundles exist: demo/_static (the static, browser-only demo, which free
accounts can host) and demo/_build (the Gradio app, which needs PRO).

Requires a write token: `hf auth login`, or HF_TOKEN in the environment.

Usage:
  python scripts/build_space.py --check
  python scripts/publish_space.py --repo-id <user>/microclip --dry-run
  python scripts/publish_space.py --repo-id <user>/microclip
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[1]
BUNDLES = {"static": REPO / "demo" / "_static", "gradio": REPO / "demo" / "_build"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. umutonuryasar/microclip")
    p.add_argument("--sdk", choices=sorted(BUNDLES), default="static",
                   help="which bundle to upload (default: static)")
    p.add_argument("--private", action="store_true",
                   help="create the Space private (can be flipped later in settings)")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be uploaded, upload nothing")
    args = p.parse_args()

    build = BUNDLES[args.sdk]
    if not build.exists():
        script = "build_static.py" if args.sdk == "static" else "build_space.py"
        raise SystemExit(f"{build} missing — run scripts/{script} first")

    ignore = ("__pycache__", ".ipynb_checkpoints")
    files = sorted(p for p in build.rglob("*")
                   if p.is_file() and not any(part in ignore for part in p.parts)
                   and p.suffix != ".pyc")
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} files, {total / 1e6:.0f} MB -> spaces/{args.repo_id}")
    for f in files:
        if f.suffix != ".jpg":  # thumbnails are thousands of files; summarize instead
            print(f"  {f.relative_to(build)}  {f.stat().st_size / 1e6:.1f} MB")
    print(f"  assets/thumbs/*.jpg  ({sum(1 for f in files if f.suffix == '.jpg')} files)")

    if args.dry_run:
        print("dry run — nothing uploaded")
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk=args.sdk,
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(build), repo_id=args.repo_id,
                      repo_type="space", commit_message=f"MicroCLIP {args.sdk} demo",
                      ignore_patterns=["**/__pycache__/**", "*.pyc"])
    print(f"https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
