# TRACE Backend

The API and AI layer for the TRACE analyst console. It owns the data and the
scoring: it loads `dataset.json`, scores every conversation server-side using
the learned model weights, holds each case's analyst decision in memory, and
serves paginated, already-scored cases to the React frontend.

## What's here

- `app.py` — Flask API (endpoints below)
- `scoring.py` — applies the learned weights via a sigmoid, builds evidence + guardrail flags
- `detectors.py` — the signal detectors (same patterns as the console)
- `weights.json` — learned model coefficients (from the trace-model project)
- `dataset.json` — synthetic conversations, each with a structured `source` field
- `generate.py` — regenerates `dataset.json` (with sources)
- `requirements.txt`

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(macOS / Linux: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`)

## Run

```powershell
python app.py
```

The API serves on **http://localhost:8000**. Leave it running, then start the
frontend (`npm run dev`) in the other project. The frontend expects the API at
that address.

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/cases?tab=queue&page=1&page_size=12` | Paginated, scored cases for a tab (`queue`, `escalated`, `safe`). Queue is ranked by risk. |
| GET | `/api/stats` | Counts per tab. |
| POST | `/api/cases/<id>/decision` | Body `{"status":"escalated"\|"safe"}`. Moves the case to that tab. Marking a non-clearable case safe is rejected (guardrail). |
| POST | `/api/cases/<id>/discard` | Returns the case to the queue (undo). |

## Notes

- Scoring is computed once at startup; decisions are held in memory and reset
  when the server restarts. A production version would use a database.
- The guardrail is enforced server-side: uncertain or minor-indicated cases
  cannot be marked safe, only escalated or left for review.
- To change the data, edit/run `generate.py`, then restart the server.
