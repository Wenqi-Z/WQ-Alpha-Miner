"""Sample WQ dataset categories and build the GP terminal field set."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_USABLE_TYPES = {"MATRIX"}


@dataclass
class SamplingResult:
    n_categories: int
    data_sets: list[str]  # sampled dataset ids
    data_fields: list[str]  # field ids from sampled datasets


def sample_session(config: dict) -> SamplingResult:
    """
    Full sampling pipeline for one session.

    Steps
    -----
    1. Load the configured data-fields parquet.
    2. Filter fields by region / delay / universe / type=MATRIX / non-event ids.
    3. Sample N subcategories uniformly from [min_categories, max_categories].
    4. Collect field ids from sampled subcategories.
    """

    cfg_gp_session = config.get("gp_session", {})
    cfg_sim = config.get("simulation", {})

    region = cfg_sim.get("region", "USA")
    universe = cfg_sim.get("universe", "TOP3000")
    delay = int(cfg_sim.get("delay", 1))

    parquet_path = Path(config["data_files"]["data_fields"])

    # Step 1 — load parquet
    df_all = pd.read_parquet(parquet_path)
    logger.info(
        "Loaded data_fields parquet: %d rows, %d columns",
        len(df_all),
        len(df_all.columns),
    )

    # Step 2 — filter fields
    mask = (
        (df_all["region"] == region)
        & (df_all["delay"] == delay)
        & (df_all["universe"] == universe)
        & (df_all["type"].isin(_USABLE_TYPES))
        & (~df_all["id"].str.lower().str.contains("event", na=False))
    )
    df_filtered = df_all[mask].copy()
    logger.info(
        "After filtering (region=%s delay=%d universe=%s type=MATRIX): %d fields",
        region,
        delay,
        universe,
        len(df_filtered),
    )
    if df_filtered.empty:
        raise RuntimeError(
            f"No MATRIX fields found for region={region} delay={delay} universe={universe}"
        )

    # Step 3 — sample subcategories

    sub = (
        df_filtered[["subcategory.id"]]
        .drop_duplicates("subcategory.id")
        .sort_values("subcategory.id")
    )
    all_subcategory_ids = sub["subcategory.id"].tolist()
    if not all_subcategory_ids:
        raise RuntimeError("No subcategories found in filtered field set")

    lo = max(1, int(cfg_gp_session.get("min_categories", 1)))
    hi = max(lo, int(cfg_gp_session.get("max_categories", 5)))
    hi = min(hi, len(all_subcategory_ids))
    lo = min(lo, hi)
    n = random.randint(lo, hi)

    chosen_subcategory_ids = random.sample(all_subcategory_ids, n)
    matching_df = df_filtered[df_filtered["subcategory.id"].isin(chosen_subcategory_ids)]
    if matching_df.empty:
        raise RuntimeError("Sampled subcategories yielded no matching fields")

    # Step 4 — collect field ids
    data_fields = matching_df["id"].tolist()

    logger.info(
        "Sampled %d subcategories → %d fields",
        len(chosen_subcategory_ids),
        len(data_fields),
    )

    return SamplingResult(
        n_categories=len(chosen_subcategory_ids),
        data_sets=chosen_subcategory_ids,
        data_fields=data_fields,
    )


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repo = Path(__file__).resolve().parent.parent.parent
    config = yaml.safe_load((repo / "config.yaml").read_text())
    result = sample_session(config)

    print(f"n_categories: {result.n_categories}")
    print(f"data_sets: {result.data_sets}")
    print(f"data_fields: {len(result.data_fields)} total")
    print(f"sample: {result.data_fields[:5]}")
