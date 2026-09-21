"""
Capture real screenshots of every dashboard page with Playwright.

    python presentation/capture_dashboard.py [port]

Assumes a Streamlit server is already running on the given port (default 8567).
Saves PNGs into presentation/dashboard_shots/.
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "8567"
ONLY = set(sys.argv[2:])   # optional: capture only these slugs
URL = f"http://127.0.0.1:{PORT}"
OUT = Path(__file__).resolve().parent / "dashboard_shots"
OUT.mkdir(parents=True, exist_ok=True)

# slug, sidebar label, a phrase that only exists on that page's body
PAGES = [
    ("overview", "Overview", "Hourly rows"),
    ("predictor", "Energy Predictor", "Predicted energy"),
    ("forecast", "Forecast", "Forecast total"),
    ("model_comparison", "Model Comparison", "chronologically held-out"),
    ("trends", "Trend & Seasonality", "Efficiency by regime"),
    ("sensitivity", "Sensitivity Analysis", "Speed sensitivity"),
    ("whatif", "What-If Scenarios", "What-If Scenario Analysis"),
]


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000},
                                device_scale_factor=2)
        page.goto(URL, wait_until="networkidle", timeout=120000)
        try:
            page.wait_for_selector("text=Energy Consumption Forecast",
                                   timeout=180_000)
        except Exception:
            pass
        time.sleep(6)

        for slug, label, marker in PAGES:
            if ONLY and slug not in ONLY:
                continue
            for attempt in range(4):
                # Streamlit wraps radio labels in an overlay that can swallow a
                # synthetic click, so dispatch the click on the input itself.
                clicked = page.evaluate(
                    """(text) => {
                        const labels = [...document.querySelectorAll('label')];
                        const el = labels.find(l => l.innerText.trim() === text);
                        if (!el) return 'not-found';
                        (el.querySelector('input') || el).click();
                        return 'clicked';
                    }""", label)
                if clicked != "clicked":
                    print(f"    {slug}: sidebar label {label!r} not found")
                # Wait for a phrase that only exists on the requested page, so a
                # slow render can never be mistaken for the page we asked for.
                try:
                    page.wait_for_selector(f'text={marker}', timeout=60_000)
                    break
                except Exception:
                    print(f"    retry {slug}: {marker!r} not on screen yet")
            page.wait_for_timeout(7000)
            target = OUT / f"{slug}.png"
            page.screenshot(path=str(target), full_page=False)
            flag = "OK " if marker in page.content() else "CHECK"
            print(f"[{flag}] {slug} -> {target.name}  (looking for {marker!r})")

        browser.close()


if __name__ == "__main__":
    main()
