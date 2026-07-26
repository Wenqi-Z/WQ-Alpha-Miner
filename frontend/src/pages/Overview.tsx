import { Link, useNavigate } from 'react-router-dom'
import { useOverview, useStatus } from '../api/queries'
import { Badge } from '../components/Badge'
import { BarChart } from '../components/BarChart'
import { Kpi } from '../components/Kpi'
import { shortId } from '../lib/format'

export function Overview() {
  const { data, isLoading, error } = useOverview()
  const { data: st } = useStatus()
  const navigate = useNavigate()

  if (error) return <div className="empty">{(error as Error).message}</div>
  if (isLoading || !data) return <div className="empty">Loading…</div>

  const k = data.kpis
  const top5 = data.leaderboard.slice(0, 5)
  const miningActive = st?.mining?.active
  const improveActive = st?.improve?.active

  return (
    <div className="page">
      <div className="grid cards" style={{ marginBottom: 16 }}>
        <Kpi label="Mining Sessions" value={k.sessions} sub={`${k.running} running`} />
        <Kpi label="Total Alphas" value={k.total_alphas.toLocaleString()} />
        <Kpi
          label="Submission-Ready"
          value={k.submit_ready}
          valueClass="pos"
        />
        <Kpi label="Improvement Sessions" value={k.improve_sessions} />
      </div>

      {(st?.mining?.running || st?.improve?.running) && (
        <div className="panel" style={{ marginBottom: 16 }}>
          <div className="panel-hd">
            <h3>Active Jobs</h3>
          </div>
          <div className="panel-bd">
            {st?.mining?.running && miningActive && (
              <div className="active-job">
                <Badge status="running" />
                <div className="info">
                  <div className="mono">{shortId(miningActive.id)}</div>
                  <div className="faint" style={{ fontSize: 12, marginTop: 2 }}>
                    Mining session
                  </div>
                </div>
                <Link className="linkish" to={`/sessions/${miningActive.id}`}>
                  Open →
                </Link>
              </div>
            )}
            {st?.improve?.running && improveActive && (
              <div className="active-job">
                <Badge status="running" />
                <div className="info">
                  <div className="mono">{shortId(improveActive.id)}</div>
                  <div className="faint" style={{ fontSize: 12, marginTop: 2 }}>
                    LLM improvement
                  </div>
                </div>
                <Link className="linkish" to={`/improve/${improveActive.id}`}>
                  Open →
                </Link>
              </div>
            )}
          </div>
        </div>
      )}

      <div
        className="grid"
        style={{ gridTemplateColumns: '2fr 1fr', marginBottom: 16 }}
      >
        <div className="panel">
          <div className="panel-hd">
            <h3>Best Sharpe by Session</h3>
            <div className="spacer" />
            <span className="legend">
              <span>
                <i style={{ background: 'var(--accent)' }} />
                best alpha
              </span>
            </span>
          </div>
          <div className="panel-bd" style={{ padding: 16 }}>
            <BarChart
              labels={data.chart_sessions.labels}
              values={data.chart_sessions.values}
            />
          </div>
        </div>
        <div className="panel">
          <div className="panel-hd">
            <h3>Sharpe Distribution</h3>
          </div>
          <div className="panel-bd" style={{ padding: 16 }}>
            <BarChart
              labels={data.sharpe_hist.labels}
              values={data.sharpe_hist.values}
              color="var(--accent-2)"
            />
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-hd">
          <h3>Recent Sessions</h3>
          <div className="spacer" />
          <Link className="linkish" to="/sessions">
            View all →
          </Link>
        </div>
        <div className="panel-bd table-wrap">
          <table>
            <thead>
              <tr>
                <th>Session</th>
                <th>Status</th>
                <th>Region</th>
                <th className="num">Alphas</th>
                <th className="num">Best Sharpe</th>
                <th className="num">Candidates</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {top5.length === 0 ? (
                <tr>
                  <td colSpan={7} className="empty">
                    No sessions yet
                  </td>
                </tr>
              ) : (
                top5.map((s) => (
                  <tr
                    key={s.id}
                    className="clickable"
                    onClick={() => navigate(`/sessions/${s.id}`)}
                  >
                    <td className="mono">{shortId(s.id)}</td>
                    <td>
                      <Badge status={s.st} />
                    </td>
                    <td className="muted">{s.reg}</td>
                    <td className="num">{s.n.toLocaleString()}</td>
                    <td className={`num ${s.sharpe >= 1.5 ? 'pos' : ''}`}>
                      {s.sharpe.toFixed(2)}
                    </td>
                    <td className="num">
                      {s.cand ? (
                        <span className="badge cand">{s.cand}</span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="faint">{s.when}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
