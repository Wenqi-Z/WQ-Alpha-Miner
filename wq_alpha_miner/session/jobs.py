"""
session/jobs.py
Subprocess job orchestration shared by the API server.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from wq_alpha_miner.session.store import (
    create_session,
    get_active_session,
    init_db,
    list_sessions,
    update_session,
)
from wq_alpha_miner.session.utils import database_path, load_config

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"
UI_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "db" / "ui_state.json"

RUNNING_STATES = frozenset({"SAMPLING", "GP_RUNNING", "RL_RUNNING", "REFINING"})
STOPPING_STATES = frozenset({"STOPPING"})
ACTIVE_STATES = RUNNING_STATES | STOPPING_STATES
TERMINAL_STATES = frozenset({"COMPLETED", "STOPPED", "FAILED", "PENDING"})
# Mining sessions whose alphas may appear on Candidates / be improved.
CANDIDATE_PARENT_STATES = frozenset(
    {"COMPLETED", "STOPPED", "STOPPING", "SAMPLING", "GP_RUNNING", "RL_RUNNING"}
)

# Mining session kinds (as opposed to kind="improve"). A session's kind is
# fixed at creation time to whichever engine produced it, so historical
# sessions remain visible as "mining" sessions even if mining.engine is
# later switched in config.yaml.
MINING_KINDS = frozenset({"gp", "rl"})
DEFAULT_MINING_ENGINE = "gp"


def is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def session_is_running(session: dict | None) -> bool:
    if not session:
        return False
    state = session["state"]
    if state in RUNNING_STATES | STOPPING_STATES:
        return True
    if state == "PENDING":
        pid = session.get("pid")
        return bool(pid and is_alive(pid))
    return False


def _read_ui_state() -> dict:
    if not UI_STATE_PATH.exists():
        return {}
    try:
        return json.loads(UI_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_ui_state(data: dict) -> None:
    UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_STATE_PATH.write_text(json.dumps(data))


def get_auto_restart() -> bool:
    return bool(_read_ui_state().get("auto_restart", False))


def set_auto_restart(enabled: bool) -> None:
    state = _read_ui_state()
    state["auto_restart"] = enabled
    _write_ui_state(state)


def mining_engine() -> str:
    """Which worker spawn_mining_worker() runs (config.yaml mining.engine)."""
    config = load_config(CONFIG_PATH)
    return (config.get("mining") or {}).get("engine", DEFAULT_MINING_ENGINE)


def _db_path() -> Path:
    return database_path(load_config(CONFIG_PATH))


def _spawn_worker(module: str, kind: str, db_path: Path) -> int:
    """
    Create the session row (with a live pid) before the subprocess even starts.

    The worker subprocess imports heavy deps (pandas/sklearn/...) before it would
    otherwise create its own session row, leaving a window where the process is
    running but no DB row reflects it — a concurrent status poll would see "no
    active session" and spawn a duplicate. Pre-creating the row here, with the
    pid we already know from Popen, closes that window.
    """
    session_id = create_session(
        db_path, config_json=json.dumps(load_config(CONFIG_PATH)), kind=kind
    )
    cmd = [
        sys.executable,
        "-m",
        module,
        "--config",
        str(CONFIG_PATH),
        "--session-id",
        session_id,
    ]
    proc = subprocess.Popen(cmd, start_new_session=True)
    update_session(db_path, session_id, pid=proc.pid)
    return proc.pid


def spawn_gp_worker() -> int:
    return _spawn_worker("wq_alpha_miner.session.gp_worker", "gp", _db_path())


def spawn_rl_worker() -> int:
    return _spawn_worker("wq_alpha_miner.session.rl_worker", "rl", _db_path())


_MINING_SPAWNERS = {"gp": spawn_gp_worker, "rl": spawn_rl_worker}


def spawn_mining_worker() -> int:
    """Spawn the worker configured in config.yaml's mining.engine."""
    engine = mining_engine()
    try:
        spawn = _MINING_SPAWNERS[engine]
    except KeyError:
        raise ValueError(f"Unknown mining.engine {engine!r} in config.yaml") from None
    return spawn()


_mining_spawn_lock = threading.Lock()


def start_mining_if_idle() -> int | None:
    """
    Spawn a mining worker unless one is already active. Returns its pid, or
    None if a session was already running.

    The check-then-spawn is guarded by a lock so that two requests racing on
    the same server process (e.g. a status poll firing right as the UI's
    "Start" click is invalidating queries) can't both see "idle" and both spawn.
    """
    with _mining_spawn_lock:
        db_path = _db_path()
        active = get_active_session(db_path, kind=mining_engine())
        if session_is_running(active):
            return None
        return spawn_mining_worker()


def spawn_improve_worker(alpha_id: str) -> int:
    cmd = [
        sys.executable,
        "-m",
        "wq_alpha_miner.session.improve_worker",
        "--alpha-id",
        alpha_id,
        "--config",
        str(CONFIG_PATH),
    ]
    proc = subprocess.Popen(cmd, start_new_session=True)
    return proc.pid


def reconcile_dead_workers(db_path: Path) -> None:
    """Mark sessions FAILED when pid is dead but state still looks active."""
    for session in list_sessions(db_path, limit=200):
        state = session["state"]
        pid = session.get("pid")
        if not pid:
            continue
        if state not in ACTIVE_STATES and state != "PENDING":
            continue
        if is_alive(pid):
            continue
        update_session(
            db_path,
            session["id"],
            state="FAILED",
            error="PID dead (detected by server)",
        )


def maybe_auto_restart_mining(db_path: Path) -> int | None:
    """Spawn next mining worker if auto-restart enabled and none running."""
    if not get_auto_restart():
        return None
    init_db(db_path)
    reconcile_dead_workers(db_path)
    return start_mining_if_idle()


_supervisor_stop = threading.Event()
_supervisor_thread: threading.Thread | None = None
_SUPERVISOR_INTERVAL_SEC = 5.0


def _mining_supervisor_loop(interval: float) -> None:
    """Keep chaining mining sessions while auto_restart is on — UI not required."""
    while not _supervisor_stop.wait(interval):
        try:
            if not get_auto_restart():
                continue
            maybe_auto_restart_mining(_db_path())
        except Exception:
            logger.exception("mining supervisor tick failed")


def start_mining_supervisor(interval: float = _SUPERVISOR_INTERVAL_SEC) -> None:
    """Start the background auto-restart loop (idempotent)."""
    global _supervisor_thread
    if _supervisor_thread is not None and _supervisor_thread.is_alive():
        return
    _supervisor_stop.clear()
    _supervisor_thread = threading.Thread(
        target=_mining_supervisor_loop,
        args=(interval,),
        name="mining-supervisor",
        daemon=True,
    )
    _supervisor_thread.start()


def stop_mining_supervisor() -> None:
    """Signal the supervisor thread to exit (best-effort)."""
    _supervisor_stop.set()


def live_duration(session: dict) -> float | None:
    if session.get("duration_sec") is not None:
        return float(session["duration_sec"])
    if session.get("started_at") and session.get("state") not in TERMINAL_STATES:
        return time.time() - float(session["started_at"])
    return None


def fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
