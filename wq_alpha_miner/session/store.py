"""
session/store.py
Single DAL for db/cache.db: alpha_results, sessions.

Public API
----------
    init_db(db_path)                -> None   (idempotent; call at process start)

    Session CRUD
    ------------
    create_session(db_path, *, config_json, kind, note) -> session_id
    update_session(db_path, session_id, **fields)
    set_stop_requested(db_path, session_id)
    get_session(db_path, session_id)     -> dict | None
    get_active_session(db_path)          -> dict | None
    list_sessions(db_path, limit)        -> list[dict]
    session_note(session)                -> dict  (parsed note JSON)

    Alpha results (session-tagged cache rows)
    -----------------------------------------
    get_session_alphas(db_path, session_id, *, process, min_sharpe, min_fitness,
                       order_by_score, submitted, archived) -> list[dict]
    get_ready_to_submit(db_path, session_id=None)  -> list[dict]
    get_alpha_by_id(db_path, alpha_id)   -> dict | None
    set_alpha_archived(db_path, alpha_id, archived=True) -> None

    Discovery caches (parquet)
    --------------------------
    load_operators(config)              -> list[dict]
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from wq_alpha_miner.session.utils import connect_db

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"COMPLETED", "STOPPED", "FAILED"}


# ── connection ───────────────────────────────────────────────────────────────


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return dict(row)


# ── init ─────────────────────────────────────────────────────────────────────


def init_db(db_path: Path) -> None:
    """Idempotent. Creates the sessions table."""
    db_path = Path(db_path)
    with connect_db(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              TEXT PRIMARY KEY,
                state           TEXT NOT NULL,
                kind            TEXT NOT NULL DEFAULT 'gp',
                note            TEXT,
                config_json     TEXT,
                pid             INTEGER,
                stop_requested  INTEGER NOT NULL DEFAULT 0,
                error           TEXT,
                created_at      REAL NOT NULL,
                started_at      REAL,
                ended_at        REAL,
                duration_sec    REAL
            );

            DROP TABLE IF EXISTS candidates;
        """)


# ── sessions ─────────────────────────────────────────────────────────────────


def _parse_note(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def session_note(session: dict | None) -> dict[str, Any]:
    """Parse sessions.note JSON (GP sampling metadata / improve seed_alpha_id)."""
    if not session:
        return {}
    return _parse_note(session.get("note"))


def create_session(
    db_path: Path,
    *,
    config_json: str,
    kind: str = "gp",
    note: str | dict | None = None,
) -> str:
    """Insert a new session in PENDING state. Returns the new session id."""
    session_id = str(uuid.uuid4())
    now = time.time()
    if isinstance(note, dict):
        note = json.dumps(note) if note else None
    with connect_db(db_path) as conn:
        conn.execute(
            """INSERT INTO sessions
               (id, state, kind, note, config_json, created_at)
               VALUES (?, 'PENDING', ?, ?, ?, ?)""",
            (
                session_id,
                kind,
                note,
                config_json,
                now,
            ),
        )
    return session_id


def update_session(
    db_path: Path,
    session_id: str,
    **fields: Any,
) -> None:
    """Generic updater for sessions table columns."""
    if not fields:
        return

    if "note" in fields and isinstance(fields["note"], dict):
        fields["note"] = json.dumps(fields["note"]) if fields["note"] else None

    if "state" in fields and fields["state"] in _TERMINAL_STATES:
        if "ended_at" not in fields:
            fields["ended_at"] = time.time()
        if "duration_sec" not in fields:
            with connect_db(db_path) as conn:
                row = conn.execute(
                    "SELECT started_at FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
            if row and row["started_at"]:
                fields["duration_sec"] = fields["ended_at"] - row["started_at"]

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [session_id]
    with connect_db(db_path) as conn:
        conn.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", values)


def set_stop_requested(db_path: Path, session_id: str) -> None:
    """Set stop_requested=1 and transition to STOPPING state."""
    with connect_db(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET stop_requested = 1, state = 'STOPPING' WHERE id = ?",
            (session_id,),
        )


def get_session(db_path: Path, session_id: str) -> dict | None:
    with connect_db(db_path) as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _row_to_dict(row)


def get_active_session(
    db_path: Path,
    *,
    kind: str | None = None,
) -> dict | None:
    """Return the most recent non-terminal session, optionally filtered by kind."""
    clauses = ["state NOT IN ('COMPLETED', 'STOPPED', 'FAILED')"]
    params: list[Any] = []
    if kind is not None:
        clauses.append("kind = ?")
        params.append(kind)
    where = " AND ".join(clauses)
    with connect_db(db_path) as conn:
        row = conn.execute(
            f"""SELECT * FROM sessions
                WHERE {where}
                ORDER BY created_at DESC LIMIT 1""",
            params,
        ).fetchone()
    return _row_to_dict(row)


def list_sessions(db_path: Path, limit: int = 50) -> list[dict]:
    with connect_db(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── alpha_results ────────────────────────────────────────────────────────────


def get_session_alphas(
    db_path: Path,
    session_id: str,
    *,
    process: str | None = None,
    min_sharpe: float | None = None,
    min_fitness: float | None = None,
    order_by_score: bool = False,
    submitted: bool | None = None,
    archived: bool | None = None,
) -> list[dict]:
    """Return alpha_results rows for a session with optional filters."""
    clauses = ["session_id = ?"]
    params: list[Any] = [session_id]

    if process is not None:
        clauses.append("process = ?")
        params.append(process)
    if min_sharpe is not None:
        clauses.append("ABS(sharpe) > ?")
        params.append(min_sharpe)
    if min_fitness is not None:
        clauses.append("ABS(fitness) > ?")
        params.append(min_fitness)
    if submitted is True:
        clauses.append("submitted = 1")
    elif submitted is False:
        clauses.append("submitted = 0")
    if archived is True:
        clauses.append("COALESCE(archived, 0) = 1")
    elif archived is False:
        clauses.append("COALESCE(archived, 0) = 0")

    order = (
        "ORDER BY MAX(COALESCE(ABS(sharpe), 0), 0) * COALESCE(ABS(fitness), 0) DESC"
        if order_by_score
        else "ORDER BY cached_at"
    )
    where = " AND ".join(clauses)

    with connect_db(db_path) as conn:
        rows = conn.execute(
            f"""SELECT * FROM alpha_results
                WHERE {where}
                {order}""",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_ready_to_submit(
    db_path: Path,
    session_id: str | None = None,
) -> list[dict]:
    """Return submittable alphas not yet submitted."""
    clauses = ["submittable = 1", "submitted = 0"]
    params: list[Any] = []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)

    where = " AND ".join(clauses)
    with connect_db(db_path) as conn:
        rows = conn.execute(
            f"""SELECT * FROM alpha_results
                WHERE {where}
                ORDER BY cached_at""",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_alpha_by_id(
    db_path: Path,
    alpha_id: str,
) -> dict | None:
    """Return full alpha_results row for an alpha_id, or None."""
    with connect_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM alpha_results WHERE alpha_id = ?",
            (alpha_id,),
        ).fetchone()
    return _row_to_dict(row)


def set_alpha_archived(
    db_path: Path,
    alpha_id: str,
    archived: bool = True,
) -> None:
    """Hide/show an alpha from the candidates list."""
    with connect_db(db_path) as conn:
        conn.execute(
            "UPDATE alpha_results SET archived = ? WHERE alpha_id = ?",
            (int(archived), alpha_id),
        )


# ── discovery caches ─────────────────────────────────────────────────────────


def load_operators(config: dict) -> list[dict]:
    """Load the operator catalogue configured in data_files.operators."""
    path = Path(config["data_files"]["operators"])
    if not path.exists():
        raise FileNotFoundError(f"Operators cache not found: {path} (run init_wiki.py first)")
    df = pd.read_parquet(path)
    logger.info("Loaded operators parquet: %d rows", len(df))
    return df.to_dict(orient="records")
