"""
session/gp_worker.py
Session state machine.  Runnable as a subprocess (spawned by UI) or directly
via CLI for headless testing.

Lifecycle (GP-only)
-------------------
    PENDING → SAMPLING → GP_RUNNING → COMPLETED
                                   ↘ STOPPING → STOPPED
    (any stage) → FAILED on unrecoverable error

    LLM improvement runs as a separate job (wq_alpha_miner.session.improve_worker).

Stopping
--------
    The UI/CLI writes stop_requested=1 to the sessions row, which also sets
    state=STOPPING.  After every generation (or before starting the next
    pipeline stage) the worker checks stop_requested.  On detection it finishes
    the in-flight WQ simulation batch (= current GP generation), persists, and
    transitions to STOPPED.

    GP stopping granularity is one generation, because GP runs simulations in
    parallel threads via joblib and cannot be interrupted mid-generation without
    leaving the population in an inconsistent state.

Per-generation streaming
------------------------
    We run `SymbolicTransformer` one generation at a time using warm_start=True.
    After each generation completes we write a gp_generations row to DB so the
    UI can render live charts.

Usage
-----
    python -m wq_alpha_miner.session.gp_worker [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from wq_alpha_miner.clients.cached import CachedWQClient
from wq_alpha_miner.gp_miner import (
    WQ_FUNCTION_SET,
    SymbolicTransformer,
    wq_fitness,
)
from wq_alpha_miner.session.sampling import sample_session
from wq_alpha_miner.session.store import (
    create_session,
    get_session,
    init_db,
    update_session,
)
from wq_alpha_miner.session.utils import load_config, setup_session_logger

logger = logging.getLogger(__name__)

# ── state machine ─────────────────────────────────────────────────────────────


class GPSessionWorker:
    """
    Runs one full alpha-mining session end-to-end.
    All state is persisted in db/cache.db; the worker is restartable (but
    not resumable mid-generation — restarts re-run from the last completed stage).
    """

    def __init__(
        self,
        client: CachedWQClient,
        config_path: str | Path = "config.yaml",
        session_id: str | None = None,
    ):
        self.client = client
        self.db_path = client.db_path
        self.config = load_config(Path(config_path))

        init_db(self.db_path)

        # session_id is normally pre-created by session/jobs.py (spawn_gp_worker)
        # before this process even starts, to avoid a window where the worker is
        # running but no DB row reflects it yet. Falls back to self-registering
        # for CLI/headless use.
        self.session_id = session_id or create_session(
            self.db_path,
            config_json=json.dumps(self.config),
        )

        update_session(self.db_path, self.session_id, pid=os.getpid())
        self.log = setup_session_logger(self.session_id, "gp")
        self.log.info(
            "Worker started  session=%s  pid=%d",
            self.session_id,
            os.getpid(),
        )

    # ── control ──────────────────────────────────────────────────────────────

    def _should_stop(self) -> bool:
        s = get_session(self.db_path, self.session_id)
        return bool(s and s.get("stop_requested"))

    def _enter_stopped(self):
        self.log.info("Graceful stop complete — session STOPPED")
        update_session(self.db_path, self.session_id, state="STOPPED")

    def _fail(self, exc: Exception):
        msg = f"{type(exc).__name__}: {exc}"
        self.log.error("Session FAILED: %s", msg)
        update_session(self.db_path, self.session_id, state="FAILED", error=msg)

    # ── entry point ──────────────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self._fail(exc)
            raise

    def _run(self) -> None:
        # ── Stage 1: SAMPLING ────────────────────────────────────────────────
        update_session(
            self.db_path,
            self.session_id,
            state="SAMPLING",
            started_at=time.time(),
        )
        self.log.info("Stage 1: sampling categories")

        try:
            sampling = sample_session(self.config)
        except RuntimeError as exc:
            self._fail(exc)
            return

        update_session(
            self.db_path,
            self.session_id,
            note={
                "n_categories": sampling.n_categories,
                "data_sets": sampling.data_sets,
                "data_fields": sampling.data_fields,
            },
        )
        self.log.info(
            "Sampled %d categories → %d fields",
            sampling.n_categories,
            len(sampling.data_fields),
        )

        if self._should_stop():
            self._enter_stopped()
            return

        # ── Stage 2: GP_RUNNING ──────────────────────────────────────────────
        self.log.info("Stage 2: GP mining")
        stopped = self._run_gp(self.client, sampling)
        if stopped:
            self._enter_stopped()
            return

        # ── Complete ─────────────────────────────────────────────────────────
        update_session(self.db_path, self.session_id, state="COMPLETED")
        self.log.info("Session COMPLETED")

    # ── GP stage ─────────────────────────────────────────────────────────────

    def _run_gp(self, client: CachedWQClient, sampling) -> bool:
        """
        Run SymbolicTransformer one generation at a time.
        Returns True if a graceful stop was requested (caller should enter STOPPED).
        """
        update_session(self.db_path, self.session_id, state="GP_RUNNING")

        # Deep-copy the GP config so pops don't mutate self.config
        cfg_gp = dict(self.config["gp"])
        sim_cfg = self.config.get("simulation", {})

        init_depth = tuple(cfg_gp.pop("init_depth"))
        const_range_raw = cfg_gp.pop("const_range", None)
        const_range = tuple(const_range_raw) if const_range_raw else None
        total_generations = int(cfg_gp.pop("generations"))

        # wq_fitness passes session_id through to CachedWQClient.simulate so
        # every alpha_results row gets tagged with this session.
        sim_kwargs = {
            "client": client,
            "session_id": self.session_id,
            "process": "gp",
            **sim_cfg,
        }

        gp = SymbolicTransformer(
            **cfg_gp,
            init_depth=init_depth,
            const_range=const_range,
            function_set=WQ_FUNCTION_SET,
            metric=wq_fitness,
            feature_names=sampling.data_fields,
            warm_start=True,
            generations=1,
        )

        for gen in range(total_generations):
            gp.generations = gen + 1
            self.log.info("GP generation %d / %d", gen + 1, total_generations)

            gp.fit(data=None, kwargs=sim_kwargs, log_dir=None)

            self.log.info("GP generation %d complete", gen + 1)

            # Check stop AFTER generation fully completes (all in-flight sims done).
            if self._should_stop():
                self.log.info("Stop requested — finishing after generation %d", gen + 1)
                return True  # caller will enter STOPPED

        self.log.info("GP complete: %d generations", total_generations)
        return False


# ── module entry point ────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WQ Alpha-Mining session worker",
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

    client = CachedWQClient(config_path=args.config)
    worker = GPSessionWorker(client=client, config_path=args.config, session_id=args.session_id)
    worker.run()


if __name__ == "__main__":
    main()
