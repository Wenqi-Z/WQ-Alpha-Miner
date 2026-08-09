from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np

from wq_alpha_miner.rl_miner.config import AlphaCfg
from wq_alpha_miner.rl_miner.core import AlphaEnvCore
from wq_alpha_miner.rl_miner.data.tokens import *


class AlphaEnvWrapper(gym.Wrapper):
    state: np.ndarray
    env: AlphaEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(self, env: AlphaEnvCore, cfg: AlphaCfg):
        super().__init__(env)
        self._init_spaces(cfg)
        self.action_space = gym.spaces.Discrete(self.SIZE_ACTION)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.SIZE_ALL - 1, shape=(self.MAX_EXPR_LENGTH,), dtype=np.int32
        )

    def _init_spaces(self, cfg: AlphaCfg):
        self.cfg = cfg
        self.MAX_EXPR_LENGTH = cfg.MAX_EXPR_LENGTH
        self.REWARD_PER_STEP = cfg.REWARD_PER_STEP

        SIZE_NULL = 1
        SIZE_OP = len(self.cfg.OPERATORS)
        SIZE_FEATURE = len(self.cfg.FEATURES)
        SIZE_DELTA_TIME = len(self.cfg.DELTA_TIMES)
        SIZE_CONSTANT = len(self.cfg.CONSTANTS)
        SIZE_SEP = 1

        self.SIZE_OP = SIZE_OP
        self.SIZE_FEATURE = SIZE_FEATURE
        self.SIZE_DELTA_TIME = SIZE_DELTA_TIME
        self.SIZE_CONSTANT = SIZE_CONSTANT

        self.SIZE_ALL = (
            SIZE_NULL + SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP
        )
        self.SIZE_ACTION = self.SIZE_ALL - SIZE_NULL

        self.OFFSET_OP = SIZE_NULL
        self.OFFSET_FEATURE = self.OFFSET_OP + SIZE_OP
        self.OFFSET_DELTA_TIME = self.OFFSET_FEATURE + SIZE_FEATURE
        self.OFFSET_CONSTANT = self.OFFSET_DELTA_TIME + SIZE_DELTA_TIME
        self.OFFSET_SEP = self.OFFSET_CONSTANT + SIZE_CONSTANT

    def _action2token(self, action_raw: int) -> Token:
        action = action_raw + 1
        if action < self.OFFSET_OP:
            raise ValueError
        elif action < self.OFFSET_FEATURE:
            return OperatorToken(self.cfg.OPERATORS[action - self.OFFSET_OP])
        elif action < self.OFFSET_DELTA_TIME:
            return FeatureToken(self.cfg.FEATURES[action - self.OFFSET_FEATURE])
        elif action < self.OFFSET_CONSTANT:
            return DeltaTimeToken(self.cfg.DELTA_TIMES[action - self.OFFSET_DELTA_TIME])
        elif action < self.OFFSET_SEP:
            return ConstantToken(self.cfg.CONSTANTS[action - self.OFFSET_CONSTANT])
        elif action == self.OFFSET_SEP:
            return SequenceIndicatorToken(SequenceIndicatorType.SEP)
        else:
            assert False

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.counter = 0
        self.state = np.zeros(self.MAX_EXPR_LENGTH, dtype=np.int32)
        self.env.reset()
        return self.state, {}

    def step(self, action: int):
        _, reward, done, truncated, info = self.env.step(self.action(action), self.MAX_EXPR_LENGTH)
        if not done:
            # Store action+1 so raw action 0 (first operator) is distinct from padding 0.
            self.state[self.counter] = action + 1
            self.counter += 1
        return self.state, self.reward(reward), done, truncated, info

    def action(self, action: int) -> Token:
        return self._action2token(action)

    def reward(self, reward: float) -> float:
        return reward + self.REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.SIZE_ACTION, dtype=bool)
        valid = self.env.valid_action_types()
        for i in range(self.OFFSET_OP, self.OFFSET_OP + self.SIZE_OP):
            if valid["op"][self.cfg.OPERATORS[i - self.OFFSET_OP].category_type()]:
                res[i - 1] = True
        if valid["select"][1]:  # FEATURE
            for i in range(self.OFFSET_FEATURE, self.OFFSET_FEATURE + self.SIZE_FEATURE):
                res[i - 1] = True
        if valid["select"][2]:  # CONSTANT
            for i in range(self.OFFSET_CONSTANT, self.OFFSET_CONSTANT + self.SIZE_CONSTANT):
                res[i - 1] = True
        if valid["select"][3]:  # DELTA_TIME
            for i in range(self.OFFSET_DELTA_TIME, self.OFFSET_DELTA_TIME + self.SIZE_DELTA_TIME):
                res[i - 1] = True
        if valid["select"][4]:  # SEP
            res[self.OFFSET_SEP - 1] = True
        return res


def AlphaEnv(cfg: AlphaCfg, **kwargs):
    return AlphaEnvWrapper(AlphaEnvCore(**kwargs), cfg)
