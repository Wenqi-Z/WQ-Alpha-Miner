"""Fetch WQ discovery data and cache as parquet in db/."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from wq_alpha_miner.clients.cached import CachedWQClient
from wq_alpha_miner.session.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    region = config["simulation"]["region"]
    data_files = config["data_files"]
    operators_path = Path(data_files["operators"])
    fields_path = Path(data_files["data_fields"])
    datasets_path = Path(data_files["data_sets"])
    for path in (operators_path, fields_path, datasets_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    client = CachedWQClient(config_path=args.config)
    operators = client.get_operators()
    universes = client.get_universes(region=region)

    field_frames = []
    dataset_frames = []
    for universe in universes:
        fields = client.get_data_fields(universe=universe, region=region)
        fdf = pd.json_normalize(fields)
        fdf["universe"] = universe
        field_frames.append(fdf)
        logging.info("%s: %d fields", universe, len(fields))

        datasets = client.get_data_sets(universe=universe, region=region)
        ddf = pd.json_normalize(datasets)
        ddf["universe"] = universe
        dataset_frames.append(ddf)
        logging.info("%s: %d datasets", universe, len(datasets))

    fields_df = pd.concat(field_frames, ignore_index=True) if field_frames else pd.DataFrame()
    datasets_df = pd.concat(dataset_frames, ignore_index=True) if dataset_frames else pd.DataFrame()

    pd.json_normalize(operators).to_parquet(operators_path, index=False)
    fields_df.to_parquet(fields_path, index=False)
    datasets_df.to_parquet(datasets_path, index=False)

    logging.info(
        "Wrote %d operators, %d fields, %d datasets (%d universes)",
        len(operators),
        len(fields_df),
        len(datasets_df),
        len(universes),
    )


if __name__ == "__main__":
    main()
