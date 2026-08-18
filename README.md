# WQ Alpha-Mining

Automated alpha-discovery pipeline for [WorldQuant Brain](https://platform.worldquantbrain.com/).

A mining engine generates alpha expressions and evaluates them via the WQ simulation API. A LangGraph deep agent then iteratively refines selected candidates into submittable alphas. A FastAPI backend + React/Vite frontend provide control and visibility.

## Features

- **Mining** — samples WQ dataset categories, evolves alpha expressions with either a
  Genetic Programming engine (`SymbolicTransformer`) or an RL engine (`MaskablePPO` +
  FASTEXPR operators), selectable via `mining.engine`, and streams live Sharpe/Fitness
  stats to the UI.
- **Candidates** — alphas above configurable Sharpe/Fitness thresholds, ready for review or
  LLM improvement.
- **Deep-agent refinement** — a LangGraph agent (`propose → evaluate → decide`) rewrites
  expressions and tunes `decay` / `truncation` / `neutralization` for a chosen candidate.
- **Submission** — alphas that pass all WQ in-sample checks (including self-correlation) are
  flagged `submittable` and can be submitted from the UI.
- **Web UI** — start/stop mining, browse sessions, triage candidates, track refinement
  rounds, and submit — all from `frontend/`.

## Architecture

```
config.yaml + .env
      │
      ├─ gp_worker.py / rl_worker.py   (subprocess, kind=gp|rl — selected by mining.engine)
      │     samples categories → mines alphas → alpha_results rows in db/cache.db
      │
      └─ improve_worker.py             (subprocess, kind=improve)
            gated on parent mining session COMPLETED
            LangGraph agent refines one seed alpha → alpha_results (process=refinement)

api/server.py + frontend/ — web UI; one mining job and one improvement job may run at once
```

## Prerequisites

- Python 3.10+ and [uv](https://docs.astral.sh/uv/) — or [Docker](https://docs.docker.com/) for the containerized single-server path
- [Node.js](https://nodejs.org/) 20.19+ (or 22.12+) and npm (local frontend only; not needed for Docker)
- WorldQuant Brain account ([sign up](https://platform.worldquantbrain.com/))
- OpenAI API key (for deep-agent refinement); LangSmith API key optional, for tracing

## Quick start

```bash
git clone <repo-url> && cd WQ-Alpha-Miner
uv sync
cd frontend && npm ci && cd ..

# create .env (see .env.example): WQ_EMAIL, WQ_PASSWORD, OPENAI_API_KEY
# (optional: LANGSMITH_API_KEY, LANGSMITH_PROJECT)

# fetch WQ operators / data fields / datasets → db/*.parquet (required once)
uv run python scripts/init_wiki.py
```

Run backend commands from the repository root.

**Development** — two terminals; Vite proxies `/api` to FastAPI on port 8000:

```bash
uv run uvicorn wq_alpha_miner.api.server:app --reload --port 8000   # terminal 1
cd frontend && npm run dev                                          # terminal 2
```

Open <http://localhost:5173>.

**Single-server** — build once, then FastAPI serves both API and static UI:

```bash
cd frontend && npm ci && npm run build && cd ..
uv run uvicorn wq_alpha_miner.api.server:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>.

**Docker** — multi-stage image builds the frontend and runs the same single-server setup.
On first start, if `db/*.parquet` are missing, the entrypoint runs `scripts/init_wiki.py`
(needs WQ credentials in `.env`):

```bash
# create .env (same as local)

docker build -t wq-alpha-miner .

docker run --rm -p 8000:8000 --env-file .env -v "$PWD/config.yaml:/app/config.yaml:ro" -v "$PWD/db:/app/db" wq-alpha-miner
```

Open <http://localhost:8000>. Mount `db/` so SQLite and parquet caches persist across restarts.

Workers can also be run directly, e.g. for headless testing:

```bash
uv run python -m wq_alpha_miner.session.gp_worker --config config.yaml
```

**Lint** — same as CI (`.github/workflows/ci.yml`). Ruff lives in the `dev` group:

```bash
uv sync --group dev

uv run ruff check .            # lint (isort, unused imports, …)
uv run ruff format --check .   # format check
uv run ruff check --fix .      # auto-fix (e.g. I001 import order)
uv run ruff format .           # apply formatting
```

## Configuration

Everything tunable lives in `config.yaml`. Key sections:

```yaml
mining:
  engine: gp            # gp | rl

simulation:              # fixed for all sessions
  region: USA
  universe: TOP3000

gp:
  population_size: 20
  generations: 5

rl:
  max_simulations: 250
  max_features: 30

agent:
  model: gpt-4o

candidate_filter:
  min_sharpe: 1.0      # strict >
  min_fitness: 0.8     # strict >
```

## Project structure

```
WQ-Alpha-Miner/
├── wq_alpha_miner/          # Python package
│   ├── api/server.py        # FastAPI backend + serves frontend/dist SPA
│   ├── clients/              # WQ Brain REST client + simulation cache
│   ├── gp_miner/              # Genetic Programming engine
│   ├── rl_miner/              # RL expression builder + MaskablePPO policy
│   └── session/
│       ├── store.py          # SQLite data access layer
│       ├── sampling.py       # Category/dataset sampling
│       ├── agent.py          # LangGraph deep-agent
│       ├── gp_worker.py      # GP mining session (subprocess entry point)
│       ├── rl_worker.py      # RL mining session (subprocess entry point)
│       ├── improve_worker.py # LLM improvement job
│       └── jobs.py           # Subprocess spawn/stop + auto-restart
├── frontend/                 # React + Vite + TypeScript UI
├── scripts/init_wiki.py      # Fetch WQ discovery data → db/*.parquet
├── Dockerfile                # Single-server image (API + frontend/dist)
├── config.yaml
└── db/                       # Local runtime data (gitignored)
```

## Database (`db/cache.db`)

| Table           | Purpose                                                                                                                                                                                                                                                                                                         |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `alpha_results` | Every simulated alpha — expression, Sharpe, fitness, `session_id`, `process` (`gp` / `rl` / `refinement`), one column per IS check (`low_sharpe`, `low_fitness`, `low_turnover`, `high_turnover`, `concentrated_weight`, `low_sub_universe_sharpe`, `self_correlation`). Cache key = expression + sim settings. |
| `sessions`      | One row per job — `kind` (`gp` / `rl` / `improve`), state, duration, PID.                                                                                                                                                                                                                                       |

## Session lifecycle

```
PENDING → (SAMPLING → GP_RUNNING | RL_RUNNING | REFINING) → COMPLETED
                                                         ↘ STOPPING → STOPPED
(any stage) → FAILED
```

A graceful stop sets `stop_requested` on the session row; the worker finishes its current
unit of work (one GP generation, one RL episode, or the in-flight refinement) before
transitioning to `STOPPED`. Improvement can only be launched once its parent mining
session is `COMPLETED`.

## Planned features

- [x] **RL mining** — `MaskablePPO` engine selectable via `mining.engine: rl`.
- [ ] **Advanced deep-agent improver** — richer refinement loops (multi-candidate search,
  better operator/field sampling, stronger failure diagnosis) beyond the current
  `propose → evaluate → decide` pass.
- [ ] **More LLM interfaces** — providers beyond OpenAI (Anthropic, local/OpenAI-compatible
  endpoints, etc.) via `agent.provider` / model config.

## License

MIT — see [LICENSE](LICENSE).
