"""
Translate gp_miner WQ_FUNCTION_SET into rl_miner Operator instances.

Operator categories match tree.py / wrapper.py action masks:
  - UnaryOperator / BinaryOperator  (arity 1 / 2, non-ts)
  - RollingOperator                 (ts, arity 1 + window)
  - PairRollingOperator             (ts, arity 2 + window)
  - GroupOperator                   (UnaryOperator subclass with baked-in group)
"""

from __future__ import annotations

from typing import List

from wq_alpha_miner.gp_miner.functions import _Function
from wq_alpha_miner.gp_miner.wq_functions import FULL_WINDOWS, WQ_FUNCTION_SET
from wq_alpha_miner.rl_miner.data.expression import (
    BinaryOperator,
    Operator,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)


class GroupOperator(UnaryOperator):
    """Unary op that renders as ``op(x, group)`` with a fixed group literal.

    Inherits ``category_type() -> UnaryOperator`` so action masks and
    ExpressionBuilder validation need no special cases.
    """

    def __init__(self, op: str, group: str) -> None:
        super().__init__(op)
        self._group = group

    def __str__(self) -> str:
        return f"{self._op}({self._operand}, {self._group})"


def _to_operator(fn: _Function) -> Operator:
    if fn.group_arg:
        return GroupOperator(fn.name, fn.group_arg)
    if fn.is_ts and fn.arity == 1:
        return RollingOperator(fn.name)
    if fn.is_ts and fn.arity == 2:
        return PairRollingOperator(fn.name)
    if fn.arity == 1:
        return UnaryOperator(fn.name)
    if fn.arity == 2:
        return BinaryOperator(fn.name)
    raise ValueError(f"Unsupported WQ function for RL: {fn.name} arity={fn.arity} is_ts={fn.is_ts}")


def build_operators(function_set: List[_Function] | None = None) -> List[Operator]:
    """Build RL operator tokens from a GP function set (default: WQ_FUNCTION_SET)."""
    return [_to_operator(fn) for fn in (function_set or WQ_FUNCTION_SET)]


# Shared with GP so both engines sample the same window pool.
DELTA_TIMES: List[int] = list(FULL_WINDOWS)
