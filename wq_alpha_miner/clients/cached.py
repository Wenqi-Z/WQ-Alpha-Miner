"""
CachedWQClient — WQClient wrapper that persists simulations to alpha_results.

Cached resources
----------------
  simulations (keyed by expression + sim settings): simulate → alpha_id
  IS checks (one column per core check in alpha_results).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from wq_alpha_miner.clients.wordquant import WQClient
from wq_alpha_miner.session.utils import connect_db, database_path, load_config

logger = logging.getLogger(__name__)

SIM_KEY_COLS = (
    "expression",
    "instrument_type",
    "region",
    "universe",
    "neutralization",
    "decay",
    "truncation",
    "delay",
    "pasteurization",
    "nan_handling",
)

IS_CORE_CHECKS = (
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "SELF_CORRELATION",
)

# Checks that gate "ready to submit" / UI score. SELF_CORRELATION is live-only at submit.
PASS_CHECKS = tuple(n for n in IS_CORE_CHECKS if n != "SELF_CORRELATION")
CHECK_COLS = tuple(name.lower() for name in IS_CORE_CHECKS)
PASS_COLS = tuple(name.lower() for name in PASS_CHECKS)

__all__ = [
    "CachedWQClient",
    "SIM_KEY_COLS",
    "IS_CORE_CHECKS",
    "PASS_CHECKS",
    "CHECK_COLS",
    "PASS_COLS",
]

_SIM_META_KEYS = frozenset({"session_id", "process", "base_alpha_id", "submittable"})


class CachedWQClient:
    """Wraps WQClient; persists simulate() results to alpha_results."""

    def __init__(
        self,
        config_path: Path | str,
        env_file: str = ".env",
    ):
        config_path = Path(config_path)
        config = load_config(config_path)
        sim_settings = config.get("simulation")
        if not sim_settings:
            raise ValueError(f"No simulation section in {config_path}")
        self._client = WQClient(env_file)
        self._db_path = database_path(config)
        self._db_path.parent.mkdir(exist_ok=True)
        self.sim_settings = dict(sim_settings)
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _init_db(self):
        sim_key_ddl = ", ".join(f"{c} TEXT NOT NULL" for c in SIM_KEY_COLS)
        check_ddl = ", ".join(f"{c} TEXT" for c in CHECK_COLS)
        with connect_db(self._db_path) as conn:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS alpha_results (
                    {sim_key_ddl},
                    alpha_id       TEXT NOT NULL,
                    sharpe         REAL,
                    fitness        REAL,
                    turnover       REAL,
                    returns        REAL,
                    drawdown       REAL,
                    session_id     TEXT,
                    process        TEXT,
                    base_alpha_id  TEXT,
                    submittable    INTEGER NOT NULL DEFAULT 0,
                    submitted      INTEGER NOT NULL DEFAULT 0,
                    submit_status  TEXT,
                    archived       INTEGER NOT NULL DEFAULT 0,
                    {check_ddl},
                    cached_at      REAL NOT NULL,
                    PRIMARY KEY ({", ".join(SIM_KEY_COLS)})
                );
            """
            )
            cols = {r[1] for r in conn.execute("PRAGMA table_info(alpha_results)")}
            if "archived" not in cols:
                conn.execute(
                    "ALTER TABLE alpha_results ADD COLUMN archived INTEGER NOT NULL DEFAULT 0"
                )

    def __getattr__(self, name):
        return getattr(self._client, name)

    def get_alpha_checks(self, alpha_id: str) -> dict[str, str | None]:
        """Return {check_col: result} from alpha_results (flat)."""
        cols_sql = ", ".join(CHECK_COLS)
        with connect_db(self._db_path) as conn:
            row = conn.execute(
                f"SELECT {cols_sql} FROM alpha_results WHERE alpha_id = ?",
                (alpha_id,),
            ).fetchone()
        if not row:
            return {}
        return dict(zip(CHECK_COLS, row, strict=True))

    def check_self_correlation(self, alpha_id: str) -> str:
        """Wait for live SELF_CORRELATION and persist its final result."""
        while True:
            check_data = self._client.get_alpha_check(alpha_id) or {}
            checks = check_data.get("checks") or (check_data.get("is") or {}).get("checks") or []
            result = next(
                (c.get("result") for c in checks if c.get("name") == "SELF_CORRELATION"),
                None,
            )
            if result not in (None, "PENDING"):
                break
            logger.info("Waiting for SELF_CORRELATION  alpha_id=%s", alpha_id)
            time.sleep(30)

        with connect_db(self._db_path) as conn:
            conn.execute(
                """UPDATE alpha_results
                   SET self_correlation = ?,
                       submittable = CASE WHEN ? = 'PASS' THEN submittable ELSE 0 END
                   WHERE alpha_id = ?""",
                (result, result, alpha_id),
            )
        return result

    def submit(self, alpha_id: str) -> str:
        """Submit one fully checked alpha and update its cached state."""
        by_col = self.get_alpha_checks(alpha_id)
        if by_col.get("self_correlation") != "PASS":
            with connect_db(self._db_path) as conn:
                conn.execute(
                    "UPDATE alpha_results SET submittable = 0 WHERE alpha_id = ?",
                    (alpha_id,),
                )
            raise RuntimeError(
                f"SELF_CORRELATION={by_col.get('self_correlation')!r}  alpha_id={alpha_id}"
            )

        reject: requests.HTTPError | None = None
        status = ""

        try:
            result = self._client.submit_and_poll(alpha_id)
            status = result.get("status", str(result)) if isinstance(result, dict) else str(result)
            with connect_db(self._db_path) as conn:
                conn.execute(
                    """UPDATE alpha_results
                       SET submitted = 1, submit_status = ?
                       WHERE alpha_id = ?""",
                    (status, alpha_id),
                )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                with connect_db(self._db_path) as conn:
                    conn.execute(
                        "UPDATE alpha_results SET submittable = 0 WHERE alpha_id = ?",
                        (alpha_id,),
                    )
                reject = exc
            else:
                raise

        if reject:
            raise RuntimeError(str(reject)) from reject
        return status

    @staticmethod
    def _sim_key(code: str, settings: dict) -> dict:
        return {
            "expression": code,
            "instrument_type": str(settings.get("instrument_type", "")),
            "region": str(settings.get("region", "")),
            "universe": str(settings.get("universe", "")),
            "neutralization": str(settings.get("neutralization", "")),
            "decay": str(settings.get("decay", "")),
            "truncation": str(settings.get("truncation", "")),
            "delay": str(settings.get("delay", "")),
            "pasteurization": str(settings.get("pasteurization", "")),
            "nan_handling": str(settings.get("nan_handling", "")),
        }

    def simulate(
        self,
        code: str,
        session_id: str = None,
        process: str = None,
        base_alpha_id: str = None,
        submittable: bool = False,
        **kwargs,
    ) -> dict:
        """Run simulation always waits and persist results to alpha_results.

        Returns dict with alpha_id, sharpe, fitness, turnover, returns, drawdown,
        flat check columns, submittable.
        """
        resolved = {**self.sim_settings, **kwargs}
        key = self._sim_key(code, resolved)
        where = " AND ".join(f"{c} = ?" for c in SIM_KEY_COLS)
        values = tuple(key[c] for c in SIM_KEY_COLS)
        result_cols = (
            "alpha_id",
            "sharpe",
            "fitness",
            "turnover",
            "returns",
            "drawdown",
            "submittable",
        ) + CHECK_COLS

        with connect_db(self._db_path) as conn:
            row = conn.execute(
                f"SELECT {', '.join(result_cols)} FROM alpha_results WHERE {where}",
                values,
            ).fetchone()
        if row:
            logger.info("sim cache hit  | %s", code)
            data = dict(zip(result_cols, row, strict=True))
            return {
                "alpha_id": data["alpha_id"],
                "sharpe": data["sharpe"],
                "fitness": data["fitness"],
                "turnover": data["turnover"],
                "returns": data["returns"],
                "drawdown": data["drawdown"],
                "submittable": bool(data["submittable"]),
                **{col: data[col] for col in CHECK_COLS},
            }

        # If not in cache, simulate and persist results to alpha_results.
        api_kwargs = {
            k: v for k, v in resolved.items() if k not in ("instrument_type", *_SIM_META_KEYS)
        }

        alpha_id = self._client.simulate(code=code, wait=True, **api_kwargs)
        if not alpha_id:
            raise RuntimeError(f"Simulation returned no alpha_id for: {code}")

        alpha_data = self._client.get_alpha(alpha_id)
        checks = list((alpha_data.get("is") or {}).get("checks") or [])
        by_name = {c.get("name"): c.get("result") for c in checks}
        by_col = {
            col: by_name.get(name) for name, col in zip(IS_CORE_CHECKS, CHECK_COLS, strict=True)
        }
        by_col["self_correlation"] = "PENDING"
        stats = alpha_data.get("is", {})
        sharpe = float(stats.get("sharpe", 0.0) or 0.0)
        fitness = float(stats.get("fitness", 0.0) or 0.0)
        turnover = float(stats["turnover"]) if stats.get("turnover") not in (None, "") else None
        returns = float(stats["returns"]) if stats.get("returns") not in (None, "") else None
        drawdown = float(stats["drawdown"]) if stats.get("drawdown") not in (None, "") else None
        is_submittable = bool(submittable or all(by_col.get(col) == "PASS" for col in PASS_COLS))
        cols = (
            SIM_KEY_COLS
            + (
                "alpha_id",
                "sharpe",
                "fitness",
                "turnover",
                "returns",
                "drawdown",
                "session_id",
                "process",
                "base_alpha_id",
                "submittable",
                "submitted",
                "submit_status",
            )
            + CHECK_COLS
            + ("cached_at",)
        )
        row_vals = (
            values
            + (
                alpha_id,
                sharpe,
                fitness,
                turnover,
                returns,
                drawdown,
                session_id,
                process,
                base_alpha_id,
                int(is_submittable),
                0,
                None,
            )
            + tuple(by_col.get(col) for col in CHECK_COLS)
            + (time.time(),)
        )
        with connect_db(self._db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO alpha_results ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' * len(cols))})",
                row_vals,
            )
        logger.info(
            "sim done  sharpe=%.3f fitness=%.3f  | %s",
            sharpe,
            fitness,
            code,
        )
        return {
            "alpha_id": alpha_id,
            "sharpe": sharpe,
            "fitness": fitness,
            "turnover": turnover,
            "returns": returns,
            "drawdown": drawdown,
            "submittable": is_submittable,
            **by_col,
        }

    def cache_size(self) -> dict:
        with connect_db(self._db_path) as conn:
            results = conn.execute("SELECT COUNT(*) FROM alpha_results").fetchone()[0]
        return {"alpha_results": results}
