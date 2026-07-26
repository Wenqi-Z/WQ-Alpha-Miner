#!/bin/sh
# Seed WQ discovery parquets on first boot if the mounted db/ is empty.
set -e
cd /app

need_seed=0
for f in db/operators.parquet db/data_fields.parquet db/data_sets.parquet; do
  if [ ! -f "$f" ]; then
    need_seed=1
    break
  fi
done

if [ "$need_seed" -eq 1 ]; then
  echo "db/ discovery caches missing — running scripts/init_wiki.py"
  python scripts/init_wiki.py --config config.yaml
fi

exec "$@"
