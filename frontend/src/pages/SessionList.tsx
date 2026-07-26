import { useNavigate } from 'react-router-dom'
import { useOverview } from '../api/queries'
import { Badge } from '../components/Badge'
import { shortId } from '../lib/format'

export function SessionList() {
  const { data, isLoading, error } = useOverview()
  const navigate = useNavigate()

  if (error) return <div className="empty">{(error as Error).message}</div>
  if (isLoading || !data) return <div className="empty">Loading…</div>

  return (
    <div className="page">
      <div style={{ marginBottom: 16 }}>
        <div className="overline">Mining</div>
        <div style={{ fontSize: 20, fontWeight: 650, marginTop: 2 }}>
          Sessions
        </div>
      </div>

      <div className="panel">
        <div className="panel-hd">
          <h3>Session Leaderboard</h3>
          <div className="spacer" />
          <span className="faint" style={{ fontSize: 12 }}>
            click a row for detail
          </span>
        </div>
        <div className="panel-bd table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 120 }}>Session</th>
                <th style={{ width: 100 }}>Status</th>
                <th className="num">Alphas</th>
                <th className="num">Best Sharpe</th>
                <th className="num">Best Fitness</th>
                <th className="num">Candidates</th>
                <th style={{ width: 80 }}>Started</th>
              </tr>
            </thead>
            <tbody>
              {data.leaderboard.length === 0 ? (
                <tr>
                  <td colSpan={7} className="empty">
                    No sessions yet — start a mining session from the toolbar.
                  </td>
                </tr>
              ) : (
                data.leaderboard.map((s) => (
                  <tr
                    key={s.id}
                    className="clickable"
                    onClick={() => navigate(`/sessions/${s.id}`)}
                  >
                    <td className="mono">{shortId(s.id)}</td>
                    <td>
                      <Badge status={s.st} />
                    </td>
                    <td className="num">{s.n.toLocaleString()}</td>
                    <td className={`num ${s.sharpe >= 1.5 ? 'pos' : ''}`}>
                      {s.sharpe.toFixed(2)}
                    </td>
                    <td className="num">{s.fit.toFixed(2)}</td>
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
