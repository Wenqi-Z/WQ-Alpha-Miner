import { NavLink } from 'react-router-dom'
import { useStatus } from '../api/queries'

export function Sidebar() {
  const { data: st } = useStatus()
  const running = st?.worker_label === 'running'
  const miningRunning = !!st?.mining?.running
  const improveRunning = !!st?.improve?.running
  const openCand = st?.open_candidates ?? 0
  const submitReady = st?.submit_ready ?? 0

  return (
    <aside className="sidebar">
      <div className="brand">
        <div>
          <h1>Alpha Mining</h1>
          <span>WorldQuant BRAIN</span>
        </div>
      </div>
      <nav className="nav">
        <NavLink to="/" end>
          <span className="ico">▧</span> Overview
        </NavLink>
        <NavLink to="/sessions">
          <span className="ico">◫</span> Sessions
          {miningRunning && <span className="run-dot" title="Mining running" />}
        </NavLink>
        <NavLink to="/candidates">
          <span className="ico">✦</span> Candidates
          <span className="chip nav-badge">{openCand}</span>
        </NavLink>
        <NavLink to="/submit">
          <span className="ico">⇪</span> Submit
          <span className="chip nav-badge">{submitReady}</span>
        </NavLink>
        <NavLink to="/improve">
          <span className="ico">⟳</span> Improvements
          {improveRunning && <span className="run-dot" title="Improve running" />}
        </NavLink>
      </nav>
      <div className="foot">
        <div>
          <span className={`status-dot ${running ? 'on' : ''}`} />
          Worker: {st?.worker_label ?? '—'}
        </div>
        <div style={{ marginTop: 6 }}>
          WQ sims cached: {st?.alphas_cached?.toLocaleString() ?? '—'}
        </div>
        <div style={{ marginTop: 6 }}>Open candidates: {openCand}</div>
      </div>
    </aside>
  )
}
