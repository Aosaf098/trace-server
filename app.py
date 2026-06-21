"""
TRACE backend API.

Owns the data and the AI. Loads dataset.json, scores every conversation at
startup, holds each case's status in memory, and serves paginated, already-
scored cases to the console.

Endpoints:
  GET  /api/cases?tab=queue|escalated|safe&page=1&page_size=12
  GET  /api/stats
  POST /api/cases/<id>/decision   body: {"status": "escalated"|"safe"}
  POST /api/cases/<id>/discard    -> returns case to the queue (status=open)

Run:  python app.py   (serves on http://localhost:8000)
"""
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

from scoring import score_case

app = Flask(__name__)
CORS(app)  # allow the Vite dev server (localhost:5173) to call this API

with open("dataset.json") as f:
    RAW = json.load(f)

# Pre-score every case once at startup (inference happens here, server-side).
SCORED = {c["id"]: score_case(c) for c in RAW}

# In-memory analyst decisions: id -> "open" | "escalated" | "safe".
# (In production this would be a database; in-memory keeps the demo simple.)
STATUS = {}


def view(c):
    """Assemble the full case object the frontend renders."""
    s = SCORED[c["id"]]
    return {
        "id": c["id"],
        "source": c.get("source", {"label": "Unknown", "platform": "Unknown"}),
        "context": c.get("context", ""),
        "msgs": c["msgs"],
        "pct": s["pct"],
        "p": s["p"],
        "band": s["band"],
        "evidence": s["evidence"],
        "minor_present": s["minor_present"],
        "dismiss_allowed": s["dismiss_allowed"],
        "status": STATUS.get(c["id"], "open"),
    }


def cases_for_tab(tab, band="all"):
    views = [view(c) for c in RAW]
    if tab == "escalated":
        return [v for v in views if v["status"] == "escalated"]
    if tab == "safe":
        return [v for v in views if v["status"] == "safe"]
    if tab == "uncertain":
        return [v for v in views if v["status"] == "uncertain"]
    # queue = anything not yet bucketed, optionally filtered by band, ranked by risk
    items = [v for v in views if v["status"] == "open"]
    if band in ("high", "review", "low"):
        items = [v for v in items if v["band"] == band]
    items.sort(key=lambda v: -v["p"])
    return items


@app.get("/api/cases")
def list_cases():
    tab = request.args.get("tab", "queue")
    band = request.args.get("band", "all")
    page = max(1, int(request.args.get("page", 1)))
    page_size = max(1, int(request.args.get("page_size", 12)))

    items = cases_for_tab(tab, band)
    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    paged = items[start:start + page_size]

    return jsonify({
        "items": paged, "tab": tab, "page": page, "page_size": page_size,
        "total": total, "total_pages": total_pages,
    })


@app.get("/api/stats")
def stats():
    views = [view(c) for c in RAW]
    open_views = [v for v in views if v["status"] == "open"]
    return jsonify({
        "queue": len(open_views),
        "uncertain": sum(1 for v in views if v["status"] == "uncertain"),
        "escalated": sum(1 for v in views if v["status"] == "escalated"),
        "safe": sum(1 for v in views if v["status"] == "safe"),
        "queue_bands": {
            "all": len(open_views),
            "high": sum(1 for v in open_views if v["band"] == "high"),
            "review": sum(1 for v in open_views if v["band"] == "review"),
            "low": sum(1 for v in open_views if v["band"] == "low"),
        },
    })


@app.post("/api/cases/<cid>/decision")
def decide(cid):
    if cid not in SCORED:
        return jsonify({"error": "case not found"}), 404
    body = request.get_json(force=True, silent=True) or {}
    status = body.get("status")
    if status not in ("escalated", "safe", "uncertain"):
        return jsonify({"error": "status must be 'escalated', 'uncertain', or 'safe'"}), 400
    current = STATUS.get(cid, "open")
    # Guardrail: an uncertain/minor case can't be cleared straight from the queue
    # (it must be escalated or routed to Uncertain first). But once it sits in
    # Uncertain, a human has reviewed it, so marking it safe is an allowed override.
    if status == "safe" and not SCORED[cid]["dismiss_allowed"] and current != "uncertain":
        return jsonify({"error": "guardrail: clear this from the Uncertain queue after review, not directly"}), 409
    STATUS[cid] = status
    return jsonify({"id": cid, "status": status})


@app.post("/api/cases/<cid>/discard")
def discard(cid):
    """Return a case to the queue — undo an escalation or a safe-mark."""
    if cid not in SCORED:
        return jsonify({"error": "case not found"}), 404
    STATUS[cid] = "open"
    return jsonify({"id": cid, "status": "open"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
