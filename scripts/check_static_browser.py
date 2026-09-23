#!/usr/bin/env python
"""Load the static demo in real headless browsers and run a search.

The Node check covers the search logic; this covers what only a browser does:
module loading, the WebAssembly runtime, rendering. It is the check that would
have caught every bug found after the first upload.

Usage:
  python scripts/check_static_browser.py [URL]   # default: local server on 8011
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011/"
SHOTS = Path(__file__).resolve().parents[1] / "demo" / "_screenshots"
QUERY = "a red double decker bus on a city street"


def run(browser_type) -> bool:
    name = browser_type.name
    browser = browser_type.launch()
    page = browser.new_page(viewport={"width": 1200, "height": 1000})
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("response", lambda r: errors.append(f"HTTP {r.status} {r.url}")
            if r.status >= 400 else None)

    page.goto(URL)
    try:
        page.wait_for_function(
            "() => /Ready|Error|Could not/.test(document.getElementById('status').textContent)",
            timeout=120_000)
    except Exception:
        pass
    status = page.inner_text("#status")
    ok = status.startswith("Ready")

    captions = {}
    if ok:
        page.fill("#query", QUERY)
        page.click("#go")
        page.wait_for_function(
            "() => document.querySelectorAll('#grid-sigmoid figure').length === 8",
            timeout=60_000)
        page.wait_for_timeout(1500)  # let lazy thumbnails paint for the screenshot
        for loss in ("softmax", "sigmoid"):
            captions[loss] = page.eval_on_selector_all(
                f"#grid-{loss} figcaption", "els => els.slice(0, 2).map(e => e.textContent)")
        broken = page.eval_on_selector_all(
            "figure img", "els => els.filter(i => i.complete && i.naturalWidth === 0).length")
        ok = broken == 0 and not errors
        status = page.inner_text("#status") + (f" | {broken} broken images" if broken else "")

    SHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=False)
    browser.close()

    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {status}")
    for loss, caps in captions.items():
        print(f"    {loss}: {caps}")
    for e in errors[:6]:
        print(f"    error: {e}")
    return ok


def main() -> None:
    with sync_playwright() as p:
        results = [run(p.firefox), run(p.chromium)] if "--all" in sys.argv else [run(p.firefox)]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
