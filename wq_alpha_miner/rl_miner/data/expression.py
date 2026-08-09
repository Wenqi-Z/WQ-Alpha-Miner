from abc import ABCMeta, abstractmethod
from typing import List, Type, Union


class Expression(metaclass=ABCMeta):
    def __repr__(self) -> str:
        return str(self)

    @property
    def is_featured(self):
        raise NotImplementedError


class Feature(Expression):
    def __init__(self, feature: str) -> None:
        self._feature = feature

    def __str__(self) -> str:
        return str(self._feature)

    @property
    def is_featured(self):
        return True


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self._value = value

    def __str__(self) -> str:
        return str(self._value)

    @property
    def is_featured(self):
        return False


class DeltaTime(Expression):
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def __str__(self) -> str:
        return str(self._delta_time)

    @property
    def is_featured(self):
        return False


# Operator base classes


class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type["Operator"]: ...


class UnaryOperator(Operator):
    def __init__(self, op: str) -> None:
        self._op = op
        self._is_featured = False

    def __call__(self, operand: Expression) -> None:
        self._operand = operand
        self._is_featured = operand.is_featured
        return self

    @classmethod
    def n_args(cls) -> int:
        return 1

    @classmethod
    def category_type(cls) -> Type["Operator"]:
        return UnaryOperator

    def __str__(self) -> str:
        return f"{self._op}({self._operand})"

    @property
    def is_featured(self):
        return self._is_featured


class BinaryOperator(Operator):
    def __init__(self, op: str) -> None:
        self._op = op
        self._is_featured = False

    def __call__(self, lhs: Expression, rhs: Expression) -> None:
        self._lhs = lhs
        self._rhs = rhs
        self._is_featured = lhs.is_featured or rhs.is_featured
        return self

    @classmethod
    def n_args(cls) -> int:
        return 2

    @classmethod
    def category_type(cls) -> Type["Operator"]:
        return BinaryOperator

    def __str__(self) -> str:
        return f"{self._op}({self._lhs}, {self._rhs})"

    @property
    def is_featured(self):
        return self._is_featured


class RollingOperator(Operator):
    def __init__(self, op: str) -> None:
        self._op = op
        self._is_featured = False

    def __call__(self, operand: Expression, delta_time: DeltaTime) -> None:
        self._operand = operand
        self._delta_time = delta_time
        self._is_featured = self._operand.is_featured
        return self

    @classmethod
    def n_args(cls) -> int:
        return 2

    @classmethod
    def category_type(cls) -> Type["Operator"]:
        return RollingOperator

    def __str__(self) -> str:
        return f"{self._op}({self._operand}, {self._delta_time._delta_time})"

    @property
    def is_featured(self):
        return self._is_featured


class RollingOperatorFeatured(Operator):
    def __init__(self, op: str, features: List[str]) -> None:
        self._op = op
        self._features = features
        self._is_featured = True

    def __call__(self, delta_time: DeltaTime) -> None:
        self._delta_time = delta_time
        return self

    @classmethod
    def n_args(cls) -> int:
        return 1

    @classmethod
    def category_type(cls) -> Type["Operator"]:
        return RollingOperatorFeatured

    def __str__(self) -> str:
        return f"{self._op}({', '.join(self._features)}, {self._delta_time._delta_time})"

    @property
    def is_featured(self):
        return self._is_featured


class PairRollingOperator(Operator):
    def __init__(self, op: str) -> None:
        self._op = op
        self._is_featured = False

    def __call__(self, lhs: Expression, rhs: Expression, delta_time: DeltaTime) -> None:
        self._lhs = lhs
        self._rhs = rhs
        self._delta_time = delta_time
        self._is_featured = lhs.is_featured or rhs.is_featured
        return self

    @classmethod
    def n_args(cls) -> int:
        return 3

    @classmethod
    def category_type(cls) -> Type["Operator"]:
        return PairRollingOperator

    def __str__(self) -> str:
        return f"{self._op}({self._lhs}, {self._rhs}, {self._delta_time._delta_time})"

    @property
    def is_featured(self):
        return self._is_featured
