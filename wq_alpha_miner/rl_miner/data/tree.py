from __future__ import annotations

from copy import deepcopy
from typing import List, Type

from wq_alpha_miner.rl_miner.data.expression import *
from wq_alpha_miner.rl_miner.data.tokens import *


class ExpressionBuilder:
    stack: List[Expression]

    def __init__(self):
        self.stack = []

    def get_tree(self) -> Expression:
        if len(self.stack) == 1:
            return self.stack[0]
        else:
            raise InvalidExpressionException(f"Expected only one tree, got {len(self.stack)}")

    def add_token(self, token: Token):
        if not self.validate(token):
            raise InvalidExpressionException(
                f"Token {token} not allowed here, stack: {self.stack}."
            )
        if isinstance(token, OperatorToken):
            n_args: int = token.operator.n_args()
            children = []
            for _ in range(n_args):
                children.append(self.stack.pop())
            self.stack.append(deepcopy(token.operator)(*reversed(children)))
        elif isinstance(token, ConstantToken):
            self.stack.append(Constant(token.constant))
        elif isinstance(token, DeltaTimeToken):
            self.stack.append(DeltaTime(token.delta_time))
        elif isinstance(token, FeatureToken):
            self.stack.append(Feature(token.feature))
        else:
            assert False

    def is_valid(self) -> bool:
        return len(self.stack) == 1 and self.stack[0].is_featured

    def validate(self, token: Token) -> bool:
        if isinstance(token, OperatorToken):
            return self.validate_op(token.operator)
        elif isinstance(token, DeltaTimeToken):
            return self.validate_dt()
        elif isinstance(token, ConstantToken):
            return self.validate_const()
        elif isinstance(token, FeatureToken):
            return self.validate_feature()
        else:
            assert False

    def validate_op(self, op: Type[Operator]) -> bool:
        if len(self.stack) < op.n_args():
            return False
        if op == UnaryOperator or isinstance(op, UnaryOperator):
            if not self.stack[-1].is_featured:
                return False
        elif op == BinaryOperator or isinstance(op, BinaryOperator):
            if not self.stack[-1].is_featured and not self.stack[-2].is_featured:
                return False
            if isinstance(self.stack[-1], DeltaTime) or isinstance(self.stack[-2], DeltaTime):
                return False
        elif op == RollingOperator or isinstance(op, RollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured:
                return False
        elif op == RollingOperatorFeatured or isinstance(op, RollingOperatorFeatured):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
        elif op == PairRollingOperator or isinstance(op, PairRollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured or not self.stack[-3].is_featured:
                return False
        else:
            assert False
        return True

    def validate_dt(self) -> bool:
        return len(self.stack) > 0 and self.stack[-1].is_featured

    def validate_const(self) -> bool:
        return len(self.stack) == 0 or self.stack[-1].is_featured

    def validate_feature(self) -> bool:
        return not (len(self.stack) >= 1 and isinstance(self.stack[-1], DeltaTime))


class InvalidExpressionException(ValueError):
    pass
