"""
session/improve_worker.py
Standalone LLM improvement job for one mining candidate.

May run while the parent mining session is still active, stopping, stopped, or
completed (one improve job at a time).

Usage
-----
    python -m wq_alpha_miner.session.improve_worker \\
        --alpha-id <alpha_id> \\
        [--config config.yaml]
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
from wq_alpha_miner.session.agent import run_candidate
from wq_alpha_miner.session.store import (
    create_session,
    get_alpha_by_id,
    get_session,
    init_db,
    update_session,
)
from wq_alpha_miner.session.utils import load_config, setup_session_logger

logger = logging.getLogger(__name__)


class ImproveWorker:
    """Run LLM refinement on a single GP candidate alpha."""

    def __init__(
        self,
        alpha_id: str,
        client: CachedWQClient,
        config_path: str | Path = "config.yaml",
    ):
        self.alpha_id = alpha_id
        self.client = client
        self.db_path = client.db_path
        self.config = load_config(Path(config_path))

        init_db(self.db_path)

        alpha = get_alpha_by_id(self.db_path, alpha_id)
        if alpha is None:
            raise ValueError(f"Alpha not found: {alpha_id}")
        self.session_id = create_session(
            self.db_path,
            config_json=json.dumps(self.config),
            kind="improve",
            note={"seed_alpha_id": alpha_id},
        )
        self.candidate = alpha

        update_session(self.db_path, self.session_id, pid=os.getpid())
        self.log = setup_session_logger(self.session_id, "improve")
        self.log.info(
            "Improve worker started  session=%s  alpha=%s  pid=%d",
            self.session_id,
            alpha_id,
            os.getpid(),
        )

    def _should_stop(self) -> bool:
        s = get_session(self.db_path, self.session_id)
        return bool(s and s.get("stop_requested"))

    def _enter_stopped(self) -> None:
        self.log.info("Graceful stop complete — improve session STOPPED")
        update_session(self.db_path, self.session_id, state="STOPPED")

    def _fail(self, exc: Exception) -> None:
        msg = f"{type(exc).__name__}: {exc}"
        self.log.error("Improve session FAILED: %s", msg)
        update_session(self.db_path, self.session_id, state="FAILED", error=msg)

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            self._fail(exc)
            raise

    def _run(self) -> None:
        update_session(
            self.db_path,
            self.session_id,
            state="REFINING",
            started_at=time.time(),
        )

        agent_config = self.config.get("agent", {})
        max_iterations = int(agent_config.get("max_iterations", 10))
        max_sim_errors = int(agent_config.get("max_sim_errors", 20))
        state = None
        ready = False

        for iteration in range(1, max_iterations + 1):
            if self._should_stop():
                self._enter_stopped()
                return

            self.log.info("Improve iteration %d / %d", iteration, max_iterations)
            ready, state = run_candidate(
                session_id=self.session_id,
                candidate=self.candidate,
                client=self.client,
                config=self.config,
                should_stop=self._should_stop,
                state=state,
                iteration_limit=iteration,
            )

            if self._should_stop():
                self._enter_stopped()
                return

            self.log.info("Improve iteration %d complete", iteration)
            if ready or state.get("sim_errors", 0) >= max_sim_errors:
                break

        update_session(self.db_path, self.session_id, state="COMPLETED")
        self.log.info(
            "Improve session COMPLETED  ready_to_submit=%s",
            ready,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WQ Alpha-Mining improvement worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alpha-id",
        required=True,
        help="Seed alpha id",
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    client = CachedWQClient(config_path=args.config)
    worker = ImproveWorker(
        client=client,
        alpha_id=args.alpha_id,
        config_path=args.config,
    )
    worker.run()


if __name__ == "__main__":
    main()
