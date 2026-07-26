"""
WQ Brain operator definitions for GP tree evolution.

Each _Function object carries:
  - name   : the exact WQ operator name used in expressions
  - arity  : number of sub-expressions it consumes
  - is_ts  : True for time-series ops that take a trailing window arg d
  - window_pool : candidate d values sampled during tree construction

The function callable is a no-op because expressions are never evaluated
locally — only the string representation is submitted to WQ Brain.
"""

from .functions import _Function, _function_map

_noop = lambda *a: None

# ── window sizes (days) ───────────────────────────────────────────────────────
FULL_WINDOWS = [5, 10, 20, 40, 60, 120]


def _make(name, arity, is_ts=False):
    return _Function(_noop, name, arity=arity, is_ts=is_ts, window_pool=FULL_WINDOWS)


# ── Arithmetic arity-2 ────────────────────────────────────────────────────────
add = _make("add", 2)
subtract = _make("subtract", 2)
multiply = _make("multiply", 2)
divide = _make("divide", 2)
signed_power = _make("signed_power", 2)
wq_min = _make("min", 2)
wq_max = _make("max", 2)

# ── Arithmetic arity-1 ────────────────────────────────────────────────────────
sqrt = _make("sqrt", 1)
log = _make("log", 1)
wq_abs = _make("abs", 1)
sign = _make("sign", 1)
reverse = _make("reverse", 1)
inverse = _make("inverse", 1)

# ── Cross-sectional arity-1 ───────────────────────────────────────────────────
rank = _make("rank", 1)
zscore = _make("zscore", 1)
normalize = _make("normalize", 1)
winsorize = _make("winsorize", 1)
scale = _make("scale", 1)

# ── Time-series arity-1 + window ─────────────────────────────────────────────
ts_mean = _make("ts_mean", 1, is_ts=True)
ts_sum = _make("ts_sum", 1, is_ts=True)
ts_std_dev = _make("ts_std_dev", 1, is_ts=True)
ts_rank = _make("ts_rank", 1, is_ts=True)
ts_delta = _make("ts_delta", 1, is_ts=True)
ts_delay = _make("ts_delay", 1, is_ts=True)
ts_zscore = _make("ts_zscore", 1, is_ts=True)
ts_decay_linear = _make("ts_decay_linear", 1, is_ts=True)
ts_av_diff = _make("ts_av_diff", 1, is_ts=True)

# ── Time-series arity-2 + window ─────────────────────────────────────────────
# generate: ts_corr(x, y, d)
ts_corr = _make("ts_corr", 2, is_ts=True)
ts_covariance = _make("ts_covariance", 2, is_ts=True)


# ── Group operators arity-1 + fixed group arg ────────────────────────────────
# Each generates e.g. group_neutralize(x, sector) — group is not evolved.
def _grp(op, group):
    return _Function(_noop, op, arity=1, group_arg=group)


group_neutralize_sector = _grp("group_neutralize", "sector")
group_neutralize_industry = _grp("group_neutralize", "industry")
group_neutralize_subindustry = _grp("group_neutralize", "subindustry")

group_rank_sector = _grp("group_rank", "sector")
group_rank_industry = _grp("group_rank", "industry")
group_rank_subindustry = _grp("group_rank", "subindustry")

group_zscore_sector = _grp("group_zscore", "sector")
group_zscore_industry = _grp("group_zscore", "industry")
group_zscore_subindustry = _grp("group_zscore", "subindustry")

WQ_FUNCTION_SET = [
    add,
    subtract,
    multiply,
    divide,
    signed_power,
    wq_min,
    wq_max,
    sqrt,
    log,
    wq_abs,
    sign,
    reverse,
    inverse,
    rank,
    zscore,
    normalize,
    winsorize,
    scale,
    ts_mean,
    ts_sum,
    ts_std_dev,
    ts_rank,
    ts_delta,
    ts_delay,
    ts_zscore,
    ts_decay_linear,
    ts_av_diff,
    ts_corr,
    ts_covariance,
    group_neutralize_sector,
    group_neutralize_industry,
    group_neutralize_subindustry,
    group_rank_sector,
    group_rank_industry,
    group_rank_subindustry,
    group_zscore_sector,
    group_zscore_industry,
    group_zscore_subindustry,
]

# Register so genetic.py can look them up by string name.
# Use name+group_arg as key so variants don't overwrite each other.
for _f in WQ_FUNCTION_SET:
    _key = _f.name + (f"_{_f.group_arg}" if _f.group_arg else "")
    _function_map[_key] = _f
