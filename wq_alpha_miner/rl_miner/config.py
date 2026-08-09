from __future__ import annotations

from typing import List

from wq_alpha_miner.rl_miner.data.expression import Operator


class AlphaCfg:
    def __init__(
        self,
        OPERATORS: List[Operator],
        FEATURES: List[str],
        DELTA_TIMES: List[int],
        CONSTANTS: List[float],
        MAX_EXPR_LENGTH: int,
        MAX_EPISODE_LENGTH: int,
        REWARD_PER_STEP: float,
    ) -> None:
        self.OPERATORS = OPERATORS
        self.FEATURES = FEATURES
        self.DELTA_TIMES = DELTA_TIMES
        self.CONSTANTS = CONSTANTS
        self.MAX_EXPR_LENGTH = MAX_EXPR_LENGTH
        self.MAX_EPISODE_LENGTH = MAX_EPISODE_LENGTH
        self.REWARD_PER_STEP = REWARD_PER_STEP
