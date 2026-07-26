import numpy as np


class _Fitness(object):
    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def nan_filter(y, pct=0.1):
    if len(y.shape) == 1:
        return np.isnan(y).sum() / len(y) > pct
    elif len(y.shape) == 2:
        return np.isnan(y).sum() / (y.shape[0] * y.shape[1]) > pct
    else:
        raise ValueError("y must be 1D or 2D.")


def unique_filter(y, n=10):
    return len(np.unique(y)) < n


def select_non_nan_inf(y, y_pred):
    if len(y.shape) == 1:
        mask = np.isfinite(y_pred)
        return y[mask], y_pred[mask]
    elif len(y.shape) == 2:
        mask = np.all(np.isfinite(y_pred), axis=0)
        return y[:, mask], y_pred[:, mask]
    else:
        raise ValueError("y must be 1D or 2D.")


_fitness_map = {}


def register_fitness(name, fitness):
    """Register a new fitness function.

    Parameters
    ----------
    name : str
        The name of the fitness function.

    fitness : _Fitness
        The fitness function to register.

    """
    _fitness_map[name] = fitness
