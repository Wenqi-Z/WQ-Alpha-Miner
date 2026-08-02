"""
session/agent.py
LangGraph deep-agent for alpha refinement.

Public API
----------
    run_candidate(session_id, candidate,
                  client, config, should_stop, state, iteration_limit)
        -> (ready, state)

Architecture
------------
    Full looping graph compiled once per candidate:
        START → propose → evaluate → decide → (propose | END)

    propose   : LLM with structured AlphaProposal output (sampled operators in human prompt).

    evaluate  : Simulates via CachedWQClient (persists to alpha_results), uses
                returned stats/checks. Returns updated valid_iter / sim_errors.

    decide    : Router node — reads history[-1].submittable, valid_iter, sim_errors,
                max_iterations, max_sim_errors, should_stop(); routes to
                "propose" or END.

    ImproveWorker advances one valid iteration target per graph invocation and
    passes the returned state into the next invocation.
    LangSmith tracing is controlled via .env (LANGSMITH_*).
"""

from __future__ import annotations

import logging
import operator
import os
import random
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from wq_alpha_miner.clients.cached import CHECK_COLS, PASS_CHECKS, PASS_COLS
from wq_alpha_miner.session.store import load_operators

if TYPE_CHECKING:
    from wq_alpha_miner.clients.cached import CachedWQClient

logger = logging.getLogger(__name__)

# How many FASTEXPR operators to sample into each propose human prompt.
_N_SAMPLED_OPERATORS = 5


# ── structured output schema ──────────────────────────────────────────────────

Neutralization = Literal["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]


class AlphaProposal(BaseModel):
    """A proposed modification to a WorldQuant Brain alpha expression."""

    expression: str = Field(
        description=(
            "Complete FASTEXPR alpha expression. Keep data fields from the base "
            "expression; you should add standard price/volume fields to enhance (open, high, low, "
            "close, volume, vwap, returns). Examples: rank(ts_mean(close, 20)), "
            "group_neutralize(rank(returns), sector)"
        )
    )
    decay: int | None = Field(
        None,
        description="Decay parameter in days (0–512). Leave None to keep current.",
    )
    truncation: float | None = Field(
        None,
        description="Truncation parameter (0.0–0.15). Leave None to keep current.",
    )
    neutralization: Neutralization | None = Field(
        None,
        description="Neutralization level. Leave None to keep current.",
    )
    rationale: str = Field(
        description=(
            "Concise explanation (1–3 sentences) of what was changed and why it "
            "should help the alpha pass the failing checks."
        )
    )


# ── LangGraph state ───────────────────────────────────────────────────────────


class AgentState(TypedDict):
    # Static candidate context (set once, never mutated)
    candidate_id: str
    base_expr: str
    base_alpha_id: str
    base_sharpe: float
    base_fitness: float
    base_checks: dict[str, str | None]
    operators: list[dict]
    sim_settings: dict
    tuning_ranges: dict
    max_iterations: int
    max_sim_errors: int

    # Mutable iteration counters (replaced each step by evaluate)
    valid_iter: int
    sim_errors: int

    # Propose → evaluate handoff (cleared after evaluate)
    pending: dict | None

    # Completed iterations (operator.add reducer)
    history: Annotated[list[dict], operator.add]


# ── helpers ───────────────────────────────────────────────────────────────────


def _last(state: AgentState) -> dict | None:
    history = state.get("history") or []
    return history[-1] if history else None


def _format_operators(ops: list[dict]) -> str:
    def cell(value: object) -> str:
        return str(value or "").replace("|", "\\|").replace("\n", " ").strip()

    lines = [
        "| Name | Definition | Description |",
        "| --- | --- | --- |",
    ]
    for op in sorted(ops, key=lambda r: str(r.get("name") or "")):
        lines.append(
            f"| {cell(op.get('name'))} | {cell(op.get('definition'))} | "
            f"{cell(op.get('description'))} |"
        )
    return "\n".join(lines)


def _fmt_checks(by_col: dict[str, str | None]) -> str:
    lines = []
    for name, col in zip(PASS_CHECKS, PASS_COLS, strict=True):
        result = by_col.get(col)
        if result is None:
            continue
        lines.append(f"  {name}: {result}")
    return "\n".join(lines) if lines else "  (no checks)"


def _build_system_prompt(state: AgentState) -> str:
    tr = state["tuning_ranges"]
    decay_range = tr.get("decay", [0, 512])
    trunc_range = tr.get("truncation", [0.0, 0.15])
    neut_options = tr.get(
        "neutralization", ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"]
    )

    sim = state["sim_settings"]

    return f"""You are an expert WorldQuant Brain FASTEXPR alpha researcher.

Your task: improve an alpha expression so that ALL WorldQuant IS checks pass.
Checks that must pass: LOW_SHARPE (≥1.25), LOW_FITNESS (≥1.0), LOW_TURNOVER,
HIGH_TURNOVER, CONCENTRATED_WEIGHT, LOW_SUB_UNIVERSE_SHARPE, SELF_CORRELATION,
MATCHES_COMPETITION.

FASTEXPR rules:
- Group args (second arg to group_*): sector, industry, subindustry
- Time-series operators take a numeric window: ts_mean(close, 20)
- Cross-sectional: rank(x), zscore(x), group_neutralize(x, sector)
- Arithmetic: add(x, y), multiply(x, y), etc.

Data fields:
- Start from the data fields already present in the base alpha expression.
- You should also add standard price/volume fields when helpful:
  open, high, low, close, volume, vwap, returns, adv20, cap.

Simulation settings (fixed):
  region={sim.get("region", "USA")}, universe={sim.get("universe", "TOP3000")}, delay={sim.get("delay", 1)}

Tunable parameters:
  decay: {decay_range[0]}–{decay_range[1]} days
  truncation: {trunc_range[0]}–{trunc_range[1]}
  neutralization: {", ".join(str(n) for n in neut_options)}

Current defaults: decay={sim.get("decay", 6)}, truncation={sim.get("truncation", 0.08)}, neutralization={sim.get("neutralization", "SUBINDUSTRY")}

Rules:
1. Keep the base alpha's data fields; add price/volume fields only when they help.
2. Focus on fixing the failing checks.
3. Propose a complete, valid expression (not a fragment).
4. Tuning decay/truncation/neutralization significantly affects Sharpe and fitness.
5. SELF_CORRELATION fails when the alpha is too similar to existing submissions."""


def _build_human_prompt(state: AgentState) -> str:
    sim = state["sim_settings"]
    history = state.get("history", [])
    last = _last(state)

    cur_decay = sim.get("decay", 6)
    cur_truncation = sim.get("truncation", 0.08)
    cur_neutralization = sim.get("neutralization", "SUBINDUSTRY")
    if last:
        cur_decay = last.get("decay", cur_decay)
        cur_truncation = last.get("truncation", cur_truncation)
        cur_neutralization = last.get("neutralization", cur_neutralization)

    base_checks = state.get("base_checks") or {}
    base_failing = [
        name
        for name, col in zip(PASS_CHECKS, PASS_COLS, strict=True)
        if base_checks.get(col) != "PASS"
    ]
    lines = [
        f"Base alpha: {state['base_expr']}",
        f"Base stats: sharpe={state['base_sharpe']:.3f}, fitness={state['base_fitness']:.3f}",
        "Base check results:",
        _fmt_checks(base_checks),
    ]
    if base_failing:
        lines.append(f"Base failing checks: {', '.join(base_failing)}")
    lines.extend(
        [
            f"Current settings: decay={cur_decay}, truncation={cur_truncation}, "
            f"neutralization={cur_neutralization}",
            "",
        ]
    )

    successes = [e for e in history if not e.get("sim_error")]
    if successes:
        lines.append(f"Previous iterations ({len(successes)} with results):")
        for entry in successes:
            lines.append(
                f"  Iter {entry.get('iteration', '?')} — "
                f"expr={entry.get('expr', '?')[:60]}  "
                f"sharpe={entry.get('sharpe', 0):.3f}  fitness={entry.get('fitness', 0):.3f}"
            )
            if entry.get("failing_checks"):
                lines.append(f"    Failing: {', '.join(entry['failing_checks'])}")
        lines.append("")

    if last and last.get("sim_error") and last.get("error_msg"):
        lines.append(f"LAST ATTEMPT FAILED with error: {last['error_msg']}")
        lines.append("Propose a different expression that avoids this error.")
    elif not last:
        lines.append(
            "First iteration — analyse the base alpha and propose an improvement."
        )

    ops = state.get("operators") or []
    n = min(_N_SAMPLED_OPERATORS, len(ops))
    sampled = random.sample(ops, n) if n else []
    lines.append("")
    if sampled:
        lines.append(
            f"This round's operators (sample of {n}) — try to incorporate one "
            "of these into your improvement if it helps:"
        )
        lines.append(_format_operators(sampled))
    else:
        lines.append(
            "No operators available — improve using only FASTEXPR rules above."
        )

    lines.append("")
    lines.append(
        "Also incorporate price/volume data to help where possible "
        "(open, high, low, close, volume, vwap, returns, adv20, cap)."
    )
    tip_sharpe = state["base_sharpe"]
    tip_fitness = state["base_fitness"]
    if successes:
        tip_sharpe = float(successes[-1].get("sharpe") or 0.0)
        tip_fitness = float(successes[-1].get("fitness") or 0.0)
    if tip_sharpe < 0 and tip_fitness < 0:
        lines.append(
            "Tip: Sharpe and fitness are both negative — try the negated "
            "alpha (multiply by -1 / wrap in reverse) first before other changes."
        )
    lines.append("\nPropose an improved alpha expression.")
    return "\n".join(lines)


# ── graph node factories ──────────────────────────────────────────────────────


def _make_propose_node(llm: ChatOpenAI) -> Callable:
    """Returns the propose node — structured AlphaProposal from the LLM."""
    structured = llm.with_structured_output(AlphaProposal)

    def propose(state: AgentState) -> dict:
        proposal: AlphaProposal = structured.invoke(
            [
                SystemMessage(content=_build_system_prompt(state)),
                HumanMessage(content=_build_human_prompt(state)),
            ]
        )
        return {
            "pending": {
                "expr": proposal.expression.strip(),
                "decay": proposal.decay,
                "truncation": proposal.truncation,
                "neutralization": (
                    proposal.neutralization.upper() if proposal.neutralization else None
                ),
                "rationale": proposal.rationale,
            }
        }

    return propose


def _make_evaluate_node(
    client: CachedWQClient,
    session_id: str,
    candidate_id: str,
    base_alpha_id: str = "",
) -> Callable:
    """Returns the evaluate node — simulates the proposal and updates counters."""

    def evaluate(state: AgentState) -> dict:
        pending = state.get("pending") or {}
        expr = pending.get("expr") or state["base_expr"]
        proposed_decay = pending.get("decay")
        proposed_truncation = pending.get("truncation")
        proposed_neutralization = pending.get("neutralization")
        rationale = pending.get("rationale", "")
        sim_settings = state["sim_settings"]

        eff_decay = (
            proposed_decay
            if proposed_decay is not None
            else int(sim_settings.get("decay", 6))
        )
        eff_truncation = (
            proposed_truncation
            if proposed_truncation is not None
            else float(sim_settings.get("truncation", 0.08))
        )
        eff_neutralization = proposed_neutralization or sim_settings.get(
            "neutralization", "SUBINDUSTRY"
        )

        valid_iter: int = state.get("valid_iter", 0)
        sim_errors: int = state.get("sim_errors", 0)

        alpha_id: str | None = None
        sharpe = 0.0
        fitness = 0.0
        checks: dict[str, str | None] = {}
        submittable = False
        sim_error = False
        error_msg: str | None = None

        try:
            # Client owns default sim settings; only override tunable params.
            result = client.simulate(
                code=expr,
                session_id=session_id,
                process="refinement",
                base_alpha_id=base_alpha_id or None,
                decay=eff_decay,
                truncation=eff_truncation,
                neutralization=eff_neutralization,
            )
            alpha_id = result.get("alpha_id")
            sharpe = float(result.get("sharpe") or 0.0)
            fitness = float(result.get("fitness") or 0.0)
            checks = {col: result.get(col) for col in CHECK_COLS}
            submittable = bool(result.get("submittable"))
        except Exception as exc:
            sim_error = True
            error_msg = f"Simulate error: {type(exc).__name__}: {exc}"
            logger.warning("simulate failed  expr=%s: %s", expr[:60], exc)

        # ── Update counters ───────────────────────────────────────────────────
        if sim_error:
            sim_errors += 1
        else:
            valid_iter += 1
        iteration_label = valid_iter

        logger.info(
            "Candidate %s  iter=%d  sim_err=%d  sharpe=%.3f  fitness=%.3f  submittable=%s",
            candidate_id,
            valid_iter,
            sim_errors,
            sharpe,
            fitness,
            submittable,
        )

        failing_checks = [
            name
            for name, col in zip(PASS_CHECKS, PASS_COLS, strict=True)
            if checks.get(col) != "PASS"
        ]
        return {
            "valid_iter": valid_iter,
            "sim_errors": sim_errors,
            "pending": None,
            "history": [
                {
                    "iteration": iteration_label,
                    "expr": expr,
                    "decay": eff_decay,
                    "truncation": eff_truncation,
                    "neutralization": eff_neutralization,
                    "rationale": rationale,
                    "alpha_id": alpha_id,
                    "sharpe": sharpe,
                    "fitness": fitness,
                    "checks": checks,
                    "failing_checks": failing_checks,
                    "submittable": submittable,
                    "sim_error": sim_error,
                    "error_msg": error_msg,
                }
            ],
        }

    return evaluate


def _make_decide_node_and_router(
    should_stop: Callable[[], bool],
) -> tuple[Callable, Callable]:
    """
    Returns (decide_node, decide_router).

    decide_node   — no-op state update; purely a routing boundary in the graph.
    decide_router — reads state and returns "propose" or END.
                    Checks: history[-1].submittable, valid_iter >= max_iterations,
                    sim_errors >= max_sim_errors, should_stop().
    """

    def decide_node(state: AgentState) -> dict:
        return {}

    def decide_router(state: AgentState) -> str:
        last = _last(state)
        if last and last.get("submittable"):
            return END
        if state.get("valid_iter", 0) >= state.get("max_iterations", 10):
            return END
        if state.get("sim_errors", 0) >= state.get("max_sim_errors", 20):
            return END
        if should_stop():
            logger.info(
                "Stop requested — halting after candidate %s iteration %d",
                state.get("candidate_id", "?"),
                state.get("valid_iter", 0),
            )
            return END
        return "propose"

    return decide_node, decide_router


# ── graph builder ─────────────────────────────────────────────────────────────


def _build_graph(
    client: CachedWQClient,
    session_id: str,
    candidate_id: str,
    config: dict,
    should_stop: Callable[[], bool],
    base_alpha_id: str = "",
) -> Any:
    """
    Compile the full looping refinement graph:
        START → propose → evaluate → decide → (propose | END)
    """
    cfg_agent = config.get("agent", {})
    model_name = cfg_agent.get("model", "gpt-4o")
    api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.2)

    propose_node = _make_propose_node(llm)
    evaluate_node = _make_evaluate_node(
        client,
        session_id,
        candidate_id,
        base_alpha_id=base_alpha_id,
    )
    decide_node, decide_router = _make_decide_node_and_router(should_stop)

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("propose", propose_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("decide", decide_node)

    builder.set_entry_point("propose")
    builder.add_edge("propose", "evaluate")
    builder.add_edge("evaluate", "decide")
    builder.add_conditional_edges(
        "decide",
        decide_router,
        {"propose": "propose", END: END},
    )

    return builder.compile()


# ── public entry point ────────────────────────────────────────────────────────


def run_candidate(
    session_id: str,
    candidate: dict,
    client: CachedWQClient,
    config: dict,
    should_stop: Callable[[], bool],
    state: AgentState | None = None,
    iteration_limit: int | None = None,
) -> tuple[bool, AgentState]:
    """
    Run refinement until the requested valid-iteration limit.

    Returns (ready_to_submit, updated_state). Passing the returned state into
    the next call preserves proposal history and counters.
    """
    cfg_agent = config.get("agent", {})
    load_dotenv(override=False)

    max_iterations = int(cfg_agent.get("max_iterations", 10))
    max_sim_errors = int(cfg_agent.get("max_sim_errors", 20))
    tuning_ranges = cfg_agent.get("tuning_ranges", {})
    candidate_id = str(candidate.get("alpha_id") or "?")
    base_alpha_id = str(candidate.get("alpha_id") or "")

    if state is None:
        state = {
            "candidate_id": candidate_id,
            "base_expr": candidate.get("expression", ""),
            "base_alpha_id": base_alpha_id,
            "base_sharpe": float(candidate.get("sharpe") or 0.0),
            "base_fitness": float(candidate.get("fitness") or 0.0),
            "base_checks": {col: candidate.get(col) for col in CHECK_COLS},
            "operators": load_operators(config),
            "sim_settings": dict(client.sim_settings),
            "tuning_ranges": tuning_ranges,
            "max_iterations": max_iterations,
            "max_sim_errors": max_sim_errors,
            "valid_iter": 0,
            "sim_errors": 0,
            "pending": None,
            "history": [],
        }
    state["max_iterations"] = iteration_limit or max_iterations

    logger.info(
        "Candidate id=%s  expr=%s  sharpe=%.3f  fitness=%.3f",
        candidate_id,
        candidate.get("expression", "")[:60],
        state["base_sharpe"],
        state["base_fitness"],
    )

    graph = _build_graph(
        client,
        session_id,
        candidate_id,
        config,
        should_stop,
        base_alpha_id=base_alpha_id,
    )

    try:
        final_state = graph.invoke(state)
    except Exception as exc:
        logger.error("Unexpected graph error for candidate %s: %s", candidate_id, exc)
        raise

    last = _last(final_state)
    if last and last.get("submittable"):
        logger.info(
            "Candidate %s READY TO SUBMIT  alpha_id=%s  sharpe=%.3f  fitness=%.3f",
            candidate_id,
            last.get("alpha_id"),
            last.get("sharpe", 0),
            last.get("fitness", 0),
        )
        return True, final_state

    logger.info(
        "Candidate %s exhausted  valid_iter=%d  sim_errors=%d  not submittable",
        candidate_id,
        final_state.get("valid_iter", 0),
        final_state.get("sim_errors", 0),
    )
    return False, final_state


if __name__ == "__main__":
    import yaml

    from wq_alpha_miner.clients.cached import CachedWQClient
    from wq_alpha_miner.session.store import (
        get_active_session,
        get_session_alphas,
        list_sessions,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(override=False)

    repo = Path(__file__).resolve().parent.parent.parent
    config_path = repo / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    client = CachedWQClient(config_path=config_path)
    db_path = client.db_path
    filt = config.get("candidate_filter", {})

    session = get_active_session(db_path) or list_sessions(db_path, limit=1)[0]
    session_id = session["id"]
    candidates = get_session_alphas(
        db_path,
        session_id,
        process="gp",
        min_sharpe=float(filt.get("min_sharpe", 1.0)),
        min_fitness=float(filt.get("min_fitness", 0.8)),
        order_by_score=True,
    )
    if not candidates:
        raise SystemExit(f"No GP candidates in {db_path} for session {session_id}")

    candidate = candidates[0]
    ready, _ = run_candidate(
        session_id=session_id,
        candidate=candidate,
        client=client,
        config=config,
        should_stop=lambda: False,
    )
    print(f"session: {session_id}")
    print(f"candidate: {candidate['expression'][:80]}")
    print(f"ready_to_submit: {ready}")
