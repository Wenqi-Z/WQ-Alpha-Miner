"""FastAPI API and production host for the WQ Alpha Mining React app.

Development (run from the repository root):
    uv run uvicorn wq_alpha_miner.api.server:app --reload --port 8000

Production-style local run:
    cd frontend && npm ci && npm run build && cd ..
    uv run uvicorn wq_alpha_miner.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wq_alpha_miner.clients.cached import CHECK_COLS, IS_CORE_CHECKS, CachedWQClient
from wq_alpha_miner.session.jobs import (
    CANDIDATE_PARENT_STATES,
    MINING_KINDS,
    fmt_duration,
    get_auto_restart,
    live_duration,
    maybe_auto_restart_mining,
    mining_engine,
    reconcile_dead_workers,
    session_is_running,
    set_auto_restart,
    spawn_improve_worker,
    start_mining_if_idle,
    start_mining_supervisor,
    stop_mining_supervisor,
)
from wq_alpha_miner.session.store import (
    get_active_session,
    get_alpha_by_id,
    get_ready_to_submit,
    get_session,
    get_session_alphas,
    init_db,
    list_sessions,
    session_note,
    set_alpha_archived,
    set_stop_requested,
)
from wq_alpha_miner.session.utils import database_path, load_config

REPO = Path(__file__).resolve().parent.parent.parent
FRONTEND = REPO / "frontend"
DIST = FRONTEND / "dist"
CONFIG_PATH = REPO / "config.yaml"
FRONTEND_BUILD_COMMAND = "cd frontend && npm ci && npm run build"

app = FastAPI(title="WQ Alpha Mining API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartMiningBody(BaseModel):
    auto_restart: bool = True


def _db() -> Path:
    return database_path(load_config(CONFIG_PATH))


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _alpha_checks(alpha: dict) -> list[dict]:
    """Per-check results for hover tooltips (name + PASS/FAIL/…)."""
    return [
        {"name": name, "result": alpha.get(col) or "—"}
        for name, col in zip(IS_CORE_CHECKS, CHECK_COLS, strict=True)
    ]


def _ui_st(state: str) -> str:
    st = state.lower()
    if st in ("sampling", "gp_running", "refining", "stopping"):
        return "running" if st != "stopping" else "queued"
    return st


def _session_summary_row(s: dict, db_path: Path) -> dict:
    sid = s["id"]
    kind = s.get("kind", "gp")
    alphas = get_session_alphas(db_path, sid, process=kind)
    best_sharpe = max((abs(a.get("sharpe") or 0) for a in alphas), default=0.0)
    best_fit = max((abs(a.get("fitness") or 0) for a in alphas), default=0.0)
    config = json.loads(s.get("config_json") or "{}")
    eligible = _eligible_alphas(db_path, sid, kind=kind)
    simulation = config.get("simulation", {})
    region = simulation.get("region", "")
    universe = simulation.get("universe", "")
    reg = f"{region} · {universe}".strip(" ·")
    created = s.get("created_at") or 0
    age_sec = time.time() - created if created else 0
    if age_sec < 3600:
        when = f"{int(age_sec / 60)}m ago"
    elif age_sec < 86400:
        when = f"{int(age_sec / 3600)}h ago"
    else:
        when = f"{int(age_sec / 86400)}d ago"
    return {
        "id": sid,
        "kind": kind,
        "st": _ui_st(s["state"]),
        "reg": reg,
        "n": len(alphas),
        "sharpe": round(best_sharpe, 2),
        "fit": round(best_fit, 2),
        "cand": len(eligible),
        "when": when,
        "state": s["state"],
    }


def _eligible_alphas(
    db_path: Path, session_id: str, *, kind: str = "gp"
) -> list[dict]:
    # Live config.yaml — triage thresholds are ops settings, not frozen per session.
    filt = load_config(CONFIG_PATH).get("candidate_filter", {})
    alphas = get_session_alphas(
        db_path,
        session_id,
        process=kind,
        min_sharpe=float(filt.get("min_sharpe", 1.0)),
        min_fitness=float(filt.get("min_fitness", 0.8)),
        order_by_score=True,
        submitted=False,
        archived=False,
    )
    # Six passing IS tests means the alpha is already at the
    # self-correlation/submission stage; it does not need LLM improvement.
    return [alpha for alpha in alphas if sum(alpha.get(col) == "PASS" for col in CHECK_COLS) < 6]


def _improvement_states(db_path: Path) -> dict[str, str]:
    states: dict[str, str] = {}
    for session in list_sessions(db_path, limit=1000):
        alpha_id = session_note(session).get("seed_alpha_id")
        if session.get("kind") != "improve" or not alpha_id:
            continue
        if alpha_id in states:
            continue
        state = session.get("state")
        if state == "COMPLETED":
            states[alpha_id] = "improved"
        elif state in ("PENDING", "REFINING", "STOPPING"):
            states[alpha_id] = "improving"
        else:
            states[alpha_id] = "open"
    return states


@app.on_event("startup")
def startup() -> None:
    CachedWQClient(config_path=CONFIG_PATH)
    init_db(_db())
    # auto_restart is persisted on disk (db/ui_state.json) so it survives across
    # a single server session's polling, but it must not survive a backend
    # restart — otherwise mining would auto-spawn on boot with no button click.
    set_auto_restart(False)
    # Background loop chains sessions while auto_restart is on, even with no UI.
    start_mining_supervisor()


@app.on_event("shutdown")
def shutdown() -> None:
    stop_mining_supervisor()


@app.get("/api/status")
def api_status() -> dict:
    db_path = _db()
    reconcile_dead_workers(db_path)
    maybe_auto_restart_mining(db_path)
    active_mining = get_active_session(db_path, kind=mining_engine())
    active_improve = get_active_session(db_path, kind="improve")
    mining_sessions = [
        s
        for s in list_sessions(db_path, limit=1000)
        if s.get("kind", "gp") in MINING_KINDS and s.get("state") in CANDIDATE_PARENT_STATES
    ]
    open_cand = sum(
        len(
            _eligible_alphas(
                db_path,
                s["id"],
                kind=s.get("kind", "gp"),
            )
        )
        for s in mining_sessions
    )
    with _connect(db_path) as conn:
        n_alphas = conn.execute("SELECT COUNT(*) FROM alpha_results").fetchone()[0]
    try:
        cache = CachedWQClient(config_path=CONFIG_PATH).cache_size()
        n_cached = cache.get("alpha_results", n_alphas)
    except Exception:
        n_cached = n_alphas
    return {
        "mining": {
            "engine": mining_engine(),
            "active": active_mining,
            "running": session_is_running(active_mining),
            "auto_restart": get_auto_restart(),
        },
        "improve": {
            "active": active_improve,
            "running": session_is_running(active_improve),
        },
        "open_candidates": open_cand,
        "submit_ready": len(get_ready_to_submit(db_path)),
        "alphas_cached": n_cached,
        "worker_label": (
            "running"
            if session_is_running(active_mining) or session_is_running(active_improve)
            else "idle"
        ),
    }


@app.get("/api/overview")
def api_overview() -> dict:
    db_path = _db()
    reconcile_dead_workers(db_path)
    sessions = [s for s in list_sessions(db_path, limit=50) if s.get("kind", "gp") in MINING_KINDS]
    rows = [_session_summary_row(s, db_path) for s in sessions]
    running = sum(1 for r in rows if r["st"] == "running")
    ready = get_ready_to_submit(db_path)
    improve_sessions = [s for s in list_sessions(db_path, limit=50) if s.get("kind") == "improve"]
    with _connect(db_path) as conn:
        total_alphas = conn.execute("SELECT COUNT(*) FROM alpha_results").fetchone()[0]
    sharpe_bins = [0, 0, 0, 0, 0, 0]
    for s in sessions:
        for a in get_session_alphas(db_path, s["id"], process=s.get("kind", "gp")):
            sh = abs(a.get("sharpe") or 0)
            if sh < 0.5:
                sharpe_bins[0] += 1
            elif sh < 1.0:
                sharpe_bins[1] += 1
            elif sh < 1.5:
                sharpe_bins[2] += 1
            elif sh < 2.0:
                sharpe_bins[3] += 1
            elif sh < 2.5:
                sharpe_bins[4] += 1
            else:
                sharpe_bins[5] += 1
    chart_sessions = rows[:5]
    return {
        "kpis": {
            "sessions": len(sessions),
            "running": running,
            "total_alphas": total_alphas,
            "submit_ready": len(ready),
            "improve_sessions": len(improve_sessions),
        },
        "leaderboard": rows,
        "chart_sessions": {
            "labels": [r["id"][:8] for r in chart_sessions],
            "values": [r["sharpe"] for r in chart_sessions],
        },
        "sharpe_hist": {
            "labels": ["<0.5", "0.5", "1.0", "1.5", "2.0", "2.5+"],
            "values": sharpe_bins,
        },
    }


@app.get("/api/sessions/{session_id}")
def api_session_detail(session_id: str) -> dict:
    db_path = _db()
    s = get_session(db_path, session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    kind = s.get("kind", "gp")
    alphas = get_session_alphas(db_path, session_id, process=kind)
    best_sharpe = max((abs(a.get("sharpe") or 0) for a in alphas), default=0.0)
    best_fit = max((abs(a.get("fitness") or 0) for a in alphas), default=0.0)
    cfg = json.loads(s.get("config_json") or "{}")
    gp_cfg = cfg.get("gp", {})
    sim_cfg = cfg.get("simulation", {})
    generations = int(gp_cfg.get("generations", 5))
    eligible_ids = {r["alpha_id"] for r in _eligible_alphas(db_path, session_id, kind=kind)}
    improvement_states = _improvement_states(db_path)
    alpha_rows = []
    for a in sorted(
        alphas,
        key=lambda x: abs(x.get("sharpe") or 0),
        reverse=True,
    )[:50]:
        aid = a.get("alpha_id", "")
        improve_state = improvement_states.get(aid, "open")
        alpha_rows.append(
            {
                "expr": a.get("expression", ""),
                "sharpe": a.get("sharpe"),
                "fit": a.get("fitness"),
                "turnover": a.get("turnover"),
                "returns": a.get("returns"),
                "drawdown": a.get("drawdown"),
                "tests_passed": sum(a.get(col) == "PASS" for col in CHECK_COLS),
                "tests_total": len(CHECK_COLS),
                "checks": _alpha_checks(a),
                "eligible": aid in eligible_ids,
                "improved": improve_state in ("improved", "improving"),
                "submit_ready": bool(a.get("submittable")),
            }
        )
    progress = sorted([abs(a.get("sharpe") or 0) for a in alphas if a.get("sharpe") is not None])
    cumulative_best = []
    best = 0.0
    for v in progress:
        best = max(best, v)
        cumulative_best.append(round(best, 3))
    return {
        "session": s,
        "duration": fmt_duration(live_duration(s)),
        "generations": generations,
        "n_alphas": len(alphas),
        "best_sharpe": round(best_sharpe, 2),
        "best_fit": round(best_fit, 2),
        "eligible_count": len(eligible_ids),
        "config": sim_cfg,
        "gp_config": {
            "population_size": gp_cfg.get("population_size"),
            "generations": generations,
        },
        "alphas": alpha_rows,
        "fitness_progress": cumulative_best[-20:] if cumulative_best else [],
    }


@app.get("/api/candidates")
def api_candidates() -> dict:
    db_path = _db()
    improve_running = session_is_running(get_active_session(db_path, kind="improve"))
    improvement_states = _improvement_states(db_path)
    groups: list[dict] = []
    for session in list_sessions(db_path, limit=1000):
        kind = session.get("kind", "gp")
        if kind not in MINING_KINDS or session.get("state") not in CANDIDATE_PARENT_STATES:
            continue
        sid = session["id"]
        config = json.loads(session.get("config_json") or "{}")
        simulation = config.get("simulation", {})
        items = []
        for alpha in _eligible_alphas(db_path, sid, kind=kind):
            status = improvement_states.get(alpha["alpha_id"], "open")
            items.append(
                {
                    "id": alpha["alpha_id"],
                    "expr": alpha.get("expression", ""),
                    "sharpe": float(alpha.get("sharpe") or 0),
                    "fit": float(alpha.get("fitness") or 0),
                    "tests_passed": sum(alpha.get(col) == "PASS" for col in CHECK_COLS),
                    "tests_total": len(CHECK_COLS),
                    "checks": _alpha_checks(alpha),
                    "status": status,
                    "can_improve": status == "open" and not improve_running,
                }
            )
        if items:
            groups.append(
                {
                    "session": sid,
                    "reg": (
                        f"{simulation.get('region', '')} · {simulation.get('universe', '')}"
                    ).strip(" ·"),
                    "st": _ui_st(session["state"]),
                    "items": items,
                }
            )
    ready = get_ready_to_submit(db_path)
    submit_ready = [
        {
            "expr": r.get("expression", ""),
            "src": (r.get("session_id") or "")[:14],
            "sharpe": float(r.get("sharpe") or 0),
            "fit": float(r.get("fitness") or 0),
            "alpha_id": r.get("alpha_id"),
            "self_correlation": r.get("self_correlation"),
        }
        for r in ready
    ]
    return {
        "groups": groups,
        "submit_ready": submit_ready,
        "improve_running": improve_running,
    }


@app.post("/api/candidates/{session_id}/{alpha_id}/improve")
def api_improve_candidate(session_id: str, alpha_id: str) -> dict:
    db_path = _db()
    parent = get_session(db_path, session_id)
    if not parent or parent.get("kind", "gp") not in MINING_KINDS:
        raise HTTPException(400, "Parent mining session not found")
    if parent.get("state") not in CANDIDATE_PARENT_STATES:
        raise HTTPException(
            400,
            f"Parent mining session state {parent.get('state')!r} cannot be improved",
        )
    if session_is_running(get_active_session(db_path, kind="improve")):
        raise HTTPException(409, "Improvement job already running")
    eligible_ids = {
        alpha["alpha_id"]
        for alpha in _eligible_alphas(db_path, session_id, kind=parent.get("kind", "gp"))
    }
    if alpha_id not in eligible_ids:
        raise HTTPException(400, "Alpha does not meet candidate thresholds")
    pid = spawn_improve_worker(alpha_id)
    return {"pid": pid, "session_id": session_id, "alpha_id": alpha_id}


@app.post("/api/candidates/{session_id}/{alpha_id}/archive")
def api_archive_candidate(session_id: str, alpha_id: str) -> dict:
    db_path = _db()
    row = get_alpha_by_id(db_path, alpha_id)
    if not row or row.get("session_id") != session_id:
        raise HTTPException(404, "Alpha not found in session")
    set_alpha_archived(db_path, alpha_id, True)
    return {"ok": True, "alpha_id": alpha_id, "archived": True}


@app.post("/api/jobs/mining/start")
def api_start_mining(body: StartMiningBody) -> dict:
    set_auto_restart(body.auto_restart)
    engine = mining_engine()
    pid = start_mining_if_idle()
    if pid is None:
        raise HTTPException(409, "Mining session already running")
    return {"pid": pid, "auto_restart": body.auto_restart, "engine": engine}


@app.post("/api/jobs/mining/stop")
def api_stop_mining() -> dict:
    db_path = _db()
    set_auto_restart(False)
    active = get_active_session(db_path, kind=mining_engine())
    if not active:
        return {"ok": True, "message": "No active mining session"}
    set_stop_requested(db_path, active["id"])
    return {"ok": True, "session_id": active["id"]}


@app.post("/api/jobs/improve/stop")
def api_stop_improve() -> dict:
    db_path = _db()
    active = get_active_session(db_path, kind="improve")
    if not active:
        return {"ok": True, "message": "No active improvement session"}
    set_stop_requested(db_path, active["id"])
    return {"ok": True, "session_id": active["id"]}


def _improve_summary(db_path: Path, s: dict) -> dict:
    """Shared seed/best/lift stats for improve list + detail."""
    sid = s["id"]
    refinements = get_session_alphas(db_path, sid, process="refinement")
    seed_id = session_note(s).get("seed_alpha_id")
    seed = get_alpha_by_id(db_path, seed_id) if seed_id else None
    seed_sharpe = float(seed.get("sharpe") or 0) if seed else 0.0
    seed_expr = (seed.get("expression") or "") if seed else ""
    best_sharpe = seed_sharpe
    for r in refinements:
        best_sharpe = max(best_sharpe, float(r.get("sharpe") or 0))
    created = s.get("created_at") or 0
    age_sec = time.time() - created if created else 0
    if age_sec < 3600:
        when = f"{int(age_sec / 60)}m ago"
    elif age_sec < 86400:
        when = f"{int(age_sec / 3600)}h ago"
    else:
        when = f"{int(age_sec / 86400)}d ago"
    st = s["state"].lower()
    if st in ("refining", "pending", "stopping"):
        st = "running" if st != "stopping" else "queued"
    return {
        "id": sid,
        "st": st,
        "seed_expr": seed_expr[:80],
        "seed_sharpe": round(seed_sharpe, 2),
        "best_sharpe": round(best_sharpe, 2),
        "lift": round(best_sharpe - seed_sharpe, 2),
        "variants": len(refinements),
        "when": when,
        "seed_alpha_id": seed_id,
        "state": s["state"],
        "seed": seed,
        "refinements": refinements,
        "seed_sharpe_raw": seed_sharpe,
        "best_sharpe_raw": best_sharpe,
    }


@app.get("/api/improve")
def api_improve_list() -> dict:
    db_path = _db()
    items = []
    for s in list_sessions(db_path, limit=100):
        if s.get("kind") != "improve":
            continue
        summary = _improve_summary(db_path, s)
        items.append(
            {
                "id": summary["id"],
                "st": summary["st"],
                "seed_expr": summary["seed_expr"],
                "seed_sharpe": summary["seed_sharpe"],
                "best_sharpe": summary["best_sharpe"],
                "lift": summary["lift"],
                "variants": summary["variants"],
                "when": summary["when"],
                "seed_alpha_id": summary["seed_alpha_id"],
            }
        )
    return {"items": items}


@app.get("/api/improve/{session_id}")
def api_improve_detail(session_id: str) -> dict:
    db_path = _db()
    s = get_session(db_path, session_id)
    if not s or s.get("kind") != "improve":
        raise HTTPException(404, "Improvement session not found")
    summary = _improve_summary(db_path, s)
    seed_sharpe = summary["seed_sharpe_raw"]
    sorted_ref = sorted(summary["refinements"], key=lambda r: r.get("cached_at") or 0)
    seed = summary.get("seed") or {}
    seed_expr = (seed.get("expression") or "") if seed else ""
    best_sharpe = seed_sharpe
    round_best = [seed_sharpe]
    round_log = [
        {
            "round": 0,
            "variant": seed_expr,
            "rationale": "",
            "sharpe": seed_sharpe,
            "delta": "+0.00",
            "status": "seed",
        }
    ]
    for i, r in enumerate(sorted_ref, start=1):
        sh = float(r.get("sharpe") or 0)
        delta = sh - (round_best[-1] if round_best else seed_sharpe)
        if sh > best_sharpe:
            best_sharpe = sh
            status = "best"
        else:
            status = "kept"
        round_best.append(max(best_sharpe, sh))
        round_log.append(
            {
                "round": i,
                "variant": r.get("expression") or "",
                "rationale": r.get("rationale") or "",
                "sharpe": sh,
                "delta": f"{delta:+.2f}",
                "status": status,
            }
        )
    cfg = load_config(CONFIG_PATH)
    max_rounds = int(cfg.get("agent", {}).get("max_iterations", 10))
    rounds_done = len(sorted_ref)
    return {
        "session": s,
        "seed_sharpe": round(seed_sharpe, 2),
        "best_sharpe": round(best_sharpe, 2),
        "lift": round(best_sharpe - seed_sharpe, 2),
        "variants": len(summary["refinements"]),
        "rounds_done": rounds_done,
        "max_rounds": max_rounds,
        "round_best": round_best[:20],
        "round_log": round_log,
        "refinements": sorted_ref,
    }


@app.post("/api/check-self-correlation/{alpha_id}")
def api_check_self_correlation(alpha_id: str) -> dict:
    db_path = _db()
    row = get_alpha_by_id(db_path, alpha_id)
    if not row or not row.get("submittable"):
        raise HTTPException(400, "Alpha not submittable")
    client = CachedWQClient(config_path=CONFIG_PATH)
    result = client.check_self_correlation(alpha_id)
    return {"result": result, "alpha_id": alpha_id}


@app.post("/api/submit/{alpha_id}")
def api_submit(alpha_id: str) -> dict:
    db_path = _db()
    row = get_alpha_by_id(db_path, alpha_id)
    if not row or not row.get("submittable"):
        raise HTTPException(400, "Alpha not submittable")
    if row.get("self_correlation") != "PASS":
        raise HTTPException(400, "Check self-correlation before submitting")
    client = CachedWQClient(config_path=CONFIG_PATH)
    try:
        status = client.submit(alpha_id)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": status, "alpha_id": alpha_id}


@app.get("/")
def index() -> FileResponse:
    index_path = DIST / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            503,
            f"Frontend not built. Run: {FRONTEND_BUILD_COMMAND}",
        )
    return FileResponse(index_path)


# check_dir=False lets the API start before the frontend has been built. This is
# useful in development, and also allows a build created after startup to be
# served without changing route registration.
app.mount(
    "/assets",
    StaticFiles(directory=str(DIST / "assets"), check_dir=False),
    name="assets",
)


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> FileResponse:
    """Serve SPA index for client-side routes (not /api)."""
    if full_path.startswith("api/") or full_path == "api":
        raise HTTPException(404, "Not found")
    # Prefer real files in dist (favicon, etc.)
    candidate = DIST / full_path
    if candidate.is_file() and DIST in candidate.resolve().parents:
        return FileResponse(candidate)
    index_path = DIST / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            503,
            f"Frontend not built. Run: {FRONTEND_BUILD_COMMAND}",
        )
    return FileResponse(index_path)
