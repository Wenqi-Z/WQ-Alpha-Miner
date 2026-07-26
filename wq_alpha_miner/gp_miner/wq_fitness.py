"""
WQ Brain fitness for GP.

Submits the generated alpha expression to WQ Brain, waits for the
simulation to finish, then returns sharpe * fitness as score.

Persistence is handled by CachedWQClient (db/cache.db); simulate()
short-circuits on already-seen expressions without hitting the API.

Return contract required by _Program.raw_fitness / _parallel_evolve:
    [score]   — list so fitness_[0] is the numeric score
"""

import logging

from wq_alpha_miner.clients.cached import CachedWQClient

from .fitness import _Fitness, register_fitness

logger = logging.getLogger(__name__)


def _wq_score(data, expression: str, args: dict):
    """
    Parameters
    ----------
    data        : ignored (no local execution)
    expression  : WQ FASTEXPR alpha string, e.g. "rank(ts_mean(close, 20))"
    args        : dict with keys
                    client      – CachedWQClient instance
                    region, universe, neutralization, decay, truncation,
                    delay, pasteurization, nan_handling

    Returns
    -------
    [score]
        score = sharpe * fitness from WQ IS stats
    """
    client: CachedWQClient = args["client"]
    sim_kwargs = {k: v for k, v in args.items() if k not in ("client", "wait")}

    score = 0.0
    sharpe = 0.0
    try:
        result = client.simulate(code=expression, **sim_kwargs)
        sharpe = float(result.get("sharpe") or 0.0)
        fitness = float(result.get("fitness") or 0.0)
        score = sharpe * fitness
        logger.info(
            "score=%.4f sharpe=%.3f fitness=%.3f  |  %s",
            score,
            sharpe,
            fitness,
            expression,
        )
    except Exception as exc:
        logger.warning("sim FAIL (%s)  |  %s", exc, expression)

    return [score]


wq_fitness = _Fitness(function=_wq_score, greater_is_better=True)
register_fitness("wq", wq_fitness)
