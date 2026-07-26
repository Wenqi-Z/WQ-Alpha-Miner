"""Shared helpers: config loading, SQLite connections, session loggers."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def database_path(config: dict) -> Path:
    path = (config.get("database") or {}).get("path")
    if not path:
        raise ValueError("No database.path in config")
    return Path(path)


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def setup_session_logger(session_id: str, worker_name: str) -> logging.Logger:
    """Return a named worker logger (propagates to root / stdout)."""
    session_logger = logging.getLogger(f"{worker_name}.{session_id}")
    session_logger.setLevel(logging.INFO)
    return session_logger
