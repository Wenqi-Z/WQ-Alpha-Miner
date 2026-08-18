from __future__ import annotations

import math
import random
from typing import Callable, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from wq_alpha_miner.rl_miner.data.expression import *
from wq_alpha_miner.rl_miner.data.tokens import *
from wq_alpha_miner.rl_miner.data.tree import ExpressionBuilder


class AlphaEnvCore(gym.Env):
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool

    def __init__(
        self,
        evaluate_fn: Callable[[str], float],
        print_expr: bool = False,
    ):
        super().__init__()

        self._evaluate_fn = evaluate_fn
        self._print_expr = print_expr

        self.eval_cnt = 0

        self.render_mode = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[List[Token], dict]:
        if seed is not None:
            self.seed(seed)
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()

    def step(
        self, action: Token, MAX_EXPR_LENGTH: int
    ) -> Tuple[List[Token], float, bool, bool, dict]:
        if (
            isinstance(action, SequenceIndicatorToken)
            and action.indicator == SequenceIndicatorType.SEP
        ):
            reward = self._evaluate()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = 0.0
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.0

        if math.isnan(reward):
            reward = 0.0

        truncated = False  # gymnasium
        return self._tokens, reward, done, truncated, self._valid_action_types()

    def _evaluate(self):
        expr: Expression = self._builder.get_tree()
        if self._print_expr:
            print(expr)

        ret = float(self._evaluate_fn(str(expr)))
        self.eval_cnt += 1
        return ret

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_rolling_featured = self._builder.validate_op(RollingOperatorFeatured)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = (
            valid_op_unary
            or valid_op_binary
            or valid_op_rolling
            or valid_op_rolling_featured
            or valid_op_pair_rolling
        )
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_feature()
        valid_stop = self._builder.is_valid()

        ret = {
            "select": [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            "op": {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                RollingOperatorFeatured: valid_op_rolling_featured,
                PairRollingOperator: valid_op_pair_rolling,
            },
        }
        return ret

    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass
