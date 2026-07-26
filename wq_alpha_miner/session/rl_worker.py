"""
session/rl_worker.py
Placeholder for reinforcement-learning alpha mining.

Not implemented yet. Selected via config.yaml's mining.engine: rl and
dispatched by session/jobs.py alongside gp_worker, so the session lifecycle
and UI wiring already work end-to-end — the session is created, then
immediately marked FAILED with a clear "not implemented" error.

Usage
-----
    python -m wq_alpha_miner.session.rl_worker [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from wq_alpha_miner.session.store import create_session, init_db, update_session
from wq_alpha_miner.session.utils import database_path, load_config, setup_session_logger

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WQ Alpha-Mining RL worker (placeholder, not implemented)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Adopt an existing session row instead of creating a new one",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    config = load_config(args.config)
    db_path = database_path(config)
    init_db(db_path)

    # session_id is normally pre-created by session/jobs.py (spawn_rl_worker)
    # before this process even starts; falls back to self-registering for
    # CLI/headless use.
    session_id = args.session_id or create_session(
        db_path, config_json=json.dumps(config), kind="rl"
    )
    update_session(db_path, session_id, pid=os.getpid())
    log = setup_session_logger(session_id, "rl")
    log.info("Worker started  session=%s  pid=%d", session_id, os.getpid())

    msg = "RL mining engine is not implemented yet"
    log.error(msg)
    update_session(db_path, session_id, state="FAILED", error=msg)


if __name__ == "__main__":
    main()
