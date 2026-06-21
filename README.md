# TRACE Backend

This repository contains the backend API for the TRACE analyst console — and it
owns both the data and the AI. It scores conversations with a semantic model,
serves the held-out test cases to the frontend, tracks analyst decisions in
memory, and exposes paginated REST endpoints.

## How scoring works

Each conversation is embedded with a sentence-transformer
(`all-MiniLM-L6-v2`). Rather than matching keywords, the model compares the
*meaning* of each message against reference examples of five grooming signals —
off-platform migration, secrecy and isolation, age-probing, emotional
dependency, and gift or money offers — producing a similarity score per signal.
A logistic-regression model (its learned coefficients in `weights.json`)
combines those five similarities into a risk probability and band. Because it
matches meaning, it handles slang, misspellings, and novel phrasing a keyword
approach would miss. Every score returns the matched signal, its similarity, and
the contributing line, so predictions stay explainable.

## Included files

- `app.py` — Flask API server; serves the held-out test set to the console
- `scoring.py` — semantic scoring using learned weights and guardrail rules
- `semantic_detectors.py` — sentence-embedding signal detection (the scoring engine)
- `detectors.py` — regex helpers for the minor-age guardrail and the evaluation baseline
- `train.py` — trains the model on semantic features; writes `weights.json` and `test_set.json`
- `evaluate.py` — model evaluation (held-out metrics, semantic-vs-keyword comparison, cross-validation); writes images to `eval_outputs/`
- `build_dataset.py` — the hand-authored dataset builder; writes `dataset.json`
- `dataset.json` — curated synthetic conversation dataset
- `weights.json` — learned model coefficients (over semantic similarity features)
- `test_set.json` — held-out cases the console serves (produced by `train.py`)
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores virtualenv, caches, and local files

## Setup

> **Requires 64-bit Python** (3.10–3.12 recommended). The semantic model pulls in
> PyTorch, which has no 32-bit Windows build. The first run downloads the model
> (~80 MB) automatically; CPU is fine.

### Windows / PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Build, train, evaluate, run

Run these in order. `train.py` must run before `app.py`, because it produces the
`weights.json` and `test_set.json` the server depends on.

```bash
python build_dataset.py   # (optional) rebuild dataset.json from the hand-authored source
python train.py           # train on the data; write weights.json + test_set.json
python evaluate.py        # produce evaluation metrics and images in eval_outputs/
python app.py             # start the API, serving the held-out test cases
```

The API listens on `http://127.0.0.1:8000` by default.

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/cases?tab=queue&page=1&page_size=12&band=all` | Returns paginated cases for a tab (`queue`, `escalated`, `safe`, `uncertain`). Queue results are sorted by risk and can be filtered by band (`high`, `review`, `low`). |
| GET | `/api/stats` | Returns counts for each tab and queue band totals. |
| POST | `/api/cases/<id>/decision` | Body `{"status": "escalated" \| "safe" \| "uncertain"}`. Updates a case decision. |
| POST | `/api/cases/<id>/discard` | Returns the case to the open queue. |

## Notes

- The server serves `test_set.json` — the held-out cases the model never trained
  on — so the console reflects performance on unseen data. If `test_set.json` is
  missing, it falls back to the full `dataset.json` (run `train.py` to create it).
- Case scoring is computed once at startup. Because each conversation is embedded,
  startup takes a few seconds while the model loads and scores the set.
- Analyst decisions are stored in memory and reset when the server restarts.
- A guardrail is enforced server-side: uncertain cases, or cases indicating a
  minor, cannot be cleared straight from the queue — they must be escalated or
  routed for review first.
- All data is synthetic and depicts manipulation *tactics* only, never explicit
  content. Metrics are on synthetic data: they demonstrate the pipeline, not
  real-world performance.
- To change the dataset, edit and run `build_dataset.py`, then re-run `train.py`
  (to refresh the weights and test set) and restart the server.
- A production deployment should persist data in a database instead of memory.