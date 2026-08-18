export function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—'
  const n = Number(v)
  if (Number.isNaN(n)) return '—'
  return `${(n * 100).toFixed(2)}%`
}

export function shortId(id: string, n = 14): string {
  if (id.length <= n) return id
  return `${id.slice(0, n)}…`
}

export function uiState(state: string): string {
  const st = state.toLowerCase()
  if (['sampling', 'gp_running', 'rl_running', 'refining'].includes(st)) return 'running'
  if (st === 'stopping') return 'queued'
  return st
}

export function stClass(s: string): string {
  return (
    (
      {
        running: 'running',
        completed: 'completed',
        queued: 'queued',
        failed: 'failed',
        stopping: 'queued',
        stopped: 'stopped',
      } as Record<string, string>
    )[s] || ''
  )
}
