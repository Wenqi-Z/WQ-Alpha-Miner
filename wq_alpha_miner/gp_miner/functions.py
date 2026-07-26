import random

import numpy as np
import pandas as pd


def register_function(name, function):
    """Register a new function.

    Parameters
    ----------
    name : str
        The name of the function.

    fitness : _Function
        The function to register.

    """
    _function_map[name] = function


class _Function(object):
    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    is_ts : bool
        Whether the function is a timeseries function. If True, the function
        will be called with an additional argument 'n'.

    parmas_need : list
        The list of parameters needed for the constraint of the indicator.
        For example, the MIDPRICE indicator needs 'high' and 'low' prices.
        MIDPRICE(high, low)

    window_pool : list
        The list of window sizes for the timeseries function.

    """

    def __init__(
        self,
        function,
        name,
        arity,
        is_ts=False,
        parmas_need=[],
        window_pool=None,
        group_arg=None,
    ):
        self.function = function
        self.name = name
        self.arity = arity

        self.is_ts = is_ts
        if is_ts:
            self.window_pool = window_pool
            self.n = None
        self.parmas_need = parmas_need
        self.group_arg = group_arg  # e.g. 'sector' → appends ", sector)" on close

    def __call__(self, *args):
        if not self.is_ts:
            return self.function(*args)
        else:
            if not self.n:
                raise ValueError("Timeseries window is not set")
            return self.function(*args, self.n)

    def set_window(self, random_state):
        self.n = random_state.choice(self.window_pool)


_function_map = {}
