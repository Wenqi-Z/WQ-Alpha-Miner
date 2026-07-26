export interface SessionRef {
  id: string
  kind?: string
  state?: string
  [key: string]: unknown
}

export interface StatusResponse {
  mining: {
    engine: string
    active: SessionRef | null
    running: boolean
    auto_restart: boolean
  }
  improve: {
    active: SessionRef | null
    running: boolean
  }
  open_candidates: number
  submit_ready: number
  alphas_cached: number
  worker_label: string
}

export interface LeaderboardRow {
  id: string
  kind: string
  st: string
  reg: string
  n: number
  sharpe: number
  fit: number
  cand: number
  when: string
  state: string
}

export interface OverviewResponse {
  kpis: {
    sessions: number
    running: number
    total_alphas: number
    submit_ready: number
    improve_sessions: number
  }
  leaderboard: LeaderboardRow[]
  chart_sessions: { labels: string[]; values: number[] }
  sharpe_hist: { labels: string[]; values: number[] }
}

export interface CheckResult {
  name: string
  result: string
}

export interface SessionAlpha {
  expr: string
  sharpe: number | null
  fit: number | null
  turnover: number | null
  returns: number | null
  drawdown: number | null
  tests_passed: number
  tests_total: number
  checks?: CheckResult[]
  eligible: boolean
  improved: boolean
  submit_ready: boolean
}

export interface SessionDetailResponse {
  session: SessionRef & {
    region?: string
    universe?: string
    state: string
  }
  duration: string
  generations: number
  n_alphas: number
  best_sharpe: number
  best_fit: number
  eligible_count: number
  config: {
    region?: string
    universe?: string
    delay?: number
    decay?: number
    neutralization?: string
    truncation?: number
  }
  gp_config: {
    population_size?: number
    generations?: number
  }
  alphas: SessionAlpha[]
  fitness_progress: number[]
}

export interface CandidateItem {
  id: string
  expr: string
  sharpe: number
  fit: number
  tests_passed: number
  tests_total: number
  checks: CheckResult[]
  status: string
  can_improve: boolean
}

export interface CandidateGroup {
  session: string
  reg: string
  st: string
  items: CandidateItem[]
}

export interface SubmitReadyItem {
  expr: string
  src: string
  sharpe: number
  fit: number
  alpha_id?: string
  self_correlation?: string
}

export interface CandidatesResponse {
  groups: CandidateGroup[]
  submit_ready: SubmitReadyItem[]
  improve_running: boolean
}

export interface ImproveRoundLog {
  round: number
  variant: string
  rationale: string
  sharpe: number
  delta: string
  status: string
}

export interface ImproveDetailResponse {
  session: SessionRef & { state: string }
  seed_sharpe: number
  best_sharpe: number
  lift: number
  variants: number
  rounds_done: number
  max_rounds: number
  round_best: number[]
  round_log: ImproveRoundLog[]
}

export interface ImproveListItem {
  id: string
  st: string
  seed_expr: string
  seed_sharpe: number
  best_sharpe: number
  lift: number
  variants: number
  when: string
  seed_alpha_id?: string
}

export interface ImproveListResponse {
  items: ImproveListItem[]
}
