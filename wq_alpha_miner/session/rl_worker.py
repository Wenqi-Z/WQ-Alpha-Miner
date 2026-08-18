"""
session/rl_worker.py
RL alpha-mining session worker. Runnable as a subprocess (spawned by UI) or
directly via CLI for headless testing.

Lifecycle
---------
    PENDING → SAMPLING → RL_RUNNING → COMPLETED
                                   ↘ STOPPING → STOPPED
    (any stage) → FAILED on unrecoverable error

Stopping
--------
    The UI/CLI writes stop_requested=1 to the sessions row.  BudgetCallback
    checks after every env step and returns False (SB3 stop) when either the
    simulation budget is exhausted or stop_requested is set — episode-granular.

Usage
-----
    python -m wq_alpha_miner.session.rl_worker [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from wq_alpha_miner.clients.cached import CachedWQClient
from wq_alpha_miner.rl_miner import (
    DELTA_TIMES,
    AlphaCfg,
    AlphaEnv,
    LSTMSharedNet,
    build_operators,
)
from wq_alpha_miner.session.sampling import sample_session
from wq_alpha_miner.session.store import (
    create_session,
    get_session,
    get_session_alphas,
    init_db,
    update_session,
)
from wq_alpha_miner.session.utils import load_config, setup_session_logger

logger = logging.getLogger(__name__)


class BudgetCallback(BaseCallback):
    """Stop MaskablePPO.learn when sim budget is spent or stop_requested is set."""

    def __init__(
        self,
        session_id: str,
        db_path: Path,
        max_simulations: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.session_id = session_id
        self.db_path = db_path
        self.max_simulations = max_simulations
        self.stopped_by_user = False
        self.stopped_by_budget = False

    def _new_sims(self) -> int:
        # Cache hits short-circuit without writing a session-tagged row, so only
        # genuine new sims for this RL session count against the budget.
        return len(get_session_alphas(self.db_path, self.session_id, process="rl"))

    def _on_step(self) -> bool:
        s = get_session(self.db_path, self.session_id)
        if s and s.get("stop_requested"):
            self.stopped_by_user = True
            return False
        if self._new_sims() >= self.max_simulations:
            self.stopped_by_budget = True
            return False
        return True


class RLSessionWorker:
    """
    Runs one RL alpha-mining session end-to-end.
    All state is persisted in db/cache.db.
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

        self.session_id = session_id or create_session(
            self.db_path,
            config_json=json.dumps(self.config),
            kind="rl",
        )

        update_session(self.db_path, self.session_id, pid=os.getpid())
        self.log = setup_session_logger(self.session_id, "rl")
        self.log.info(
            "Worker started  session=%s  pid=%d",
            self.session_id,
            os.getpid(),
        )

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
            state="SAMPLING",
            started_at=time.time(),
        )
        self.log.info("Stage 1: sampling categories")

        try:
            sampling = sample_session(self.config)
        except RuntimeError as exc:
            self._fail(exc)
            return

        cfg_rl = self.config.get("rl", {})
        max_features = int(cfg_rl.get("max_features", 30))
        fields = list(sampling.data_fields)
        if len(fields) > max_features:
            rng = random.Random(cfg_rl.get("random_state", 42))
            fields = rng.sample(fields, max_features)

        update_session(
            self.db_path,
            self.session_id,
            note={
                "n_categories": sampling.n_categories,
                "data_sets": sampling.data_sets,
                "data_fields": fields,
            },
        )
        self.log.info(
            "Sampled %d categories → %d fields (capped from %d)",
            sampling.n_categories,
            len(fields),
            len(sampling.data_fields),
        )

        if self._should_stop():
            self._enter_stopped()
            return

        self.log.info("Stage 2: RL mining")
        stopped = self._run_rl(fields)
        if stopped:
            self._enter_stopped()
            return

        update_session(self.db_path, self.session_id, state="COMPLETED")
        self.log.info("Session COMPLETED")

    def _run_rl(self, features: list[str]) -> bool:
        """
        Train MaskablePPO until sim budget or stop_requested.
        Returns True if a graceful stop was requested.
        """
        update_session(self.db_path, self.session_id, state="RL_RUNNING")

        cfg_rl = dict(self.config.get("rl", {}))
        sim_cfg = self.config.get("simulation", {})

        max_simulations = int(cfg_rl.get("max_simulations", 250))
        max_expr_length = int(cfg_rl.get("max_expr_length", 20))
        reward_per_step = float(cfg_rl.get("reward_per_step", 0.0))
        constants = [float(c) for c in cfg_rl.get("constants", [1, 2, 5, 10])]
        device_str = str(cfg_rl.get("device", "cpu"))
        device = torch.device(device_str)
        seed = int(cfg_rl.get("random_state", 42))
        verbose = int(cfg_rl.get("verbose", 0))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        operators = build_operators()
        alpha_cfg = AlphaCfg(
            OPERATORS=operators,
            FEATURES=features,
            DELTA_TIMES=list(DELTA_TIMES),
            CONSTANTS=constants,
            MAX_EXPR_LENGTH=max_expr_length,
            MAX_EPISODE_LENGTH=max_expr_length,
            REWARD_PER_STEP=reward_per_step,
        )

        client = self.client
        session_id = self.session_id
        log = self.log

        def evaluate_fn(expression: str) -> float:
            score = 0.0
            try:
                result = client.simulate(
                    code=expression,
                    session_id=session_id,
                    process="rl",
                    **sim_cfg,
                )
                sharpe = float(result.get("sharpe") or 0.0)
                fitness = float(result.get("fitness") or 0.0)
                score = sharpe * fitness
                log.info(
                    "score=%.4f sharpe=%.3f fitness=%.3f  |  %s",
                    score,
                    sharpe,
                    fitness,
                    expression,
                )
            except Exception as exc:
                log.warning("sim FAIL (%s)  |  %s", exc, expression)
            return score

        env = AlphaEnv(alpha_cfg, evaluate_fn=evaluate_fn)
        env = ActionMasker(env, lambda e: e.action_masks())

        policy_kwargs = {
            "features_extractor_class": LSTMSharedNet,
            "features_extractor_kwargs": {
                "n_layers": int(cfg_rl.get("n_layers", 2)),
                "d_model": int(cfg_rl.get("d_model", 64)),
                "dropout": float(cfg_rl.get("dropout", 0.1)),
                "device": device,
            },
        }

        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=int(cfg_rl.get("n_steps", 64)),
            batch_size=int(cfg_rl.get("batch_size", 64)),
            gamma=float(cfg_rl.get("gamma", 1.0)),
            ent_coef=float(cfg_rl.get("ent_coef", 0.01)),
            learning_rate=float(cfg_rl.get("learning_rate", 3e-4)),
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device_str,
        )

        callback = BudgetCallback(
            session_id=session_id,
            db_path=self.db_path,
            max_simulations=max_simulations,
        )

        # Large ceiling — BudgetCallback is the real stop condition.
        total_timesteps = max(max_simulations * max_expr_length * 4, 10_000)
        self.log.info(
            "MaskablePPO learn  max_sims=%d  features=%d  ops=%d  timesteps_cap=%d",
            max_simulations,
            len(features),
            len(operators),
            total_timesteps,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)

        new_sims = callback._new_sims()
        self.log.info(
            "RL complete: new_sims=%d/%d  eval_cnt=%d  stopped_by_user=%s",
            new_sims,
            max_simulations,
            getattr(env.unwrapped, "eval_cnt", -1),
            callback.stopped_by_user,
        )
        return callback.stopped_by_user


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WQ Alpha-Mining RL session worker",
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
    worker = RLSessionWorker(client=client, config_path=args.config, session_id=args.session_id)
    worker.run()


if __name__ == "__main__":
    main()
