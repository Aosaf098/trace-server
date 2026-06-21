# TRACE Backend

This repository contains the backend API for the TRACE analyst console.
It loads `dataset.json`, computes case scoring using learned model weights,
tracks analyst decisions in memory, and exposes paginated REST endpoints for
a frontend application.

## Included files

- `app.py` — Flask API server
- `scoring.py` — scoring logic using learned weights and guardrail rules
- `detectors.py` — signal detection helpers used in scoring
- `evaluate.py` — model evaluation
- `weights.json` — model coefficients used for scoring
- `dataset.json` — synthetic conversation dataset
- `generate.py` — regenerate the synthetic dataset
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores virtualenv, caches, and local files

## Setup

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

## Run the server

```bash
python app.py
```

The API listens on `http://127.0.0.1:8000` by default.

## API reference

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/cases?tab=queue&page=1&page_size=12` | Returns paginated cases for a tab (`queue`, `escalated`, `safe`, `uncertain`). Queue results are sorted by risk. |
| GET | `/api/stats` | Returns counts for each tab and queue band totals. |
| POST | `/api/cases/<id>/decision` | Body `{"status": "escalated" | "safe" | "uncertain"}`. Updates a case decision. |
| POST | `/api/cases/<id>/discard` | Returns the case to the open queue. |

## Notes

- Case scoring is computed once at startup and stored in memory.
- Analyst decisions are also stored in memory and reset when the server restarts.
- A production deployment should persist data in a database instead of memory.
- To update the dataset, modify and run `generate.py`, then restart the server.
