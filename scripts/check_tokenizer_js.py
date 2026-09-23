#!/usr/bin/env python
"""Write tokenizer fixtures, then run the JS parity check against them."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from microclip.config import load_config  # noqa: E402
from microclip.data.tokenizer import CaptionTokenizer  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = load_config(REPO / "configs" / "base.yml")
    spec_path = REPO / cfg["tokenizer"]["path"]
    max_len = cfg["data"]["max_text_len"]
    tokenizer = CaptionTokenizer(str(spec_path))

    with open(REPO / cfg["data"]["root"] / cfg["data"]["val_ann"]) as f:
        captions = [a["caption"] for a in json.load(f)["annotations"]]

    # Real captions, plus edge cases the demo will actually see from users.
    texts = captions[:3000] + [
        "", " ", "a", "A MAN.", "  leading spaces", "trailing  ",
        "émoji café naïve", "日本語のテキスト", "🐶 dog emoji 🏄",
        "hyphen-ated, punctuation!? 42 numbers 3.14",
        "a " * 80,  # longer than max_len: must truncate identically
    ]
    cases = []
    for text in texts:
        ids, mask = tokenizer.encode(text, max_len)
        cases.append({"text": text, "ids": ids, "mask": mask})

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"spec_path": str(spec_path), "max_len": max_len, "cases": cases}, f)
        fixtures = f.name

    result = subprocess.run(["node", str(REPO / "scripts" / "check_tokenizer_js.mjs"), fixtures])
    Path(fixtures).unlink()
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
