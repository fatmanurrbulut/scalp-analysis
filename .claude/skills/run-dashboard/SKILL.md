---
name: run-dashboard
description: Launch and drive the Streamlit dashboard (app.py) in this repo via Docker + Playwright, upload a CSV, select a patient, and screenshot the result.
---

# Running the Scalp Analysis Streamlit dashboard

This is a Streamlit app (`app.py`) served via `docker-compose.yml`'s
`dashboard` service. `chromium-cli` is not available in this environment —
drive it with Python Playwright instead.

## Launch

```bash
cd /Users/fatmanurbulut/Desktop/scalp-analysis
docker compose up -d --build dashboard
curl -sf http://localhost:8501/_stcore/health && echo healthy
```

No fixed sleep needed — poll `/_stcore/health` until it returns `ok`.

## One-time setup (if not already installed)

```bash
pip install playwright
python -m playwright install chromium --with-deps
```

## Drive it

```python
from playwright.sync_api import sync_playwright

CSV_PATH = "/Users/fatmanurbulut/Desktop/scalp-analysis/data/mock_patient_session_analysis_biological.csv"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1400})
    page.on("pageerror", lambda exc: print("PAGE ERROR:", exc))

    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    page.wait_for_selector("text=Scalp Analysis", timeout=20000)

    # Upload CSV — plain file input, set_input_files works directly
    page.locator('input[type="file"]').set_input_files(CSV_PATH)
    page.wait_for_timeout(3000)  # let Streamlit rerun before touching the selectbox

    # Patient selectbox is a BaseWeb Select — do NOT click the placeholder text or a
    # generic div[data-baseweb="select"] selector, it does not reliably open/select.
    # Use role-based locators instead:
    page.get_by_role("combobox").first.click()
    page.wait_for_timeout(1000)
    page.get_by_role("option").first.click()
    page.wait_for_timeout(4000)  # full page (charts, tables) needs time to render

    page.screenshot(path="/tmp/dashboard.png", full_page=True)
    browser.close()
```

## Gotchas

- **Patient selectbox**: see above — `get_by_role("combobox")` / `get_by_role("option")`, not CSS attribute selectors.
- **Anomaly detail table is wide** (`st.dataframe`, glide-grid) and needs horizontal
  scroll to see all columns. Locate it with
  `page.locator('[data-testid="stDataFrame"]').last`, move the mouse over its
  bounding box, then `page.mouse.wheel(800, 0)` a few times (page-level scroll
  or `scrollLeft` on the container does not work — glide-grid renders to canvas).
- **No fixed `sleep`** for the initial page load — poll
  `http://localhost:8501/_stcore/health`.
- Check `page.on("pageerror", ...)` — a page can render its shell while a
  downstream `st.dataframe`/`st.plotly_chart` call silently fails.

## Teardown

```bash
docker compose down
```
