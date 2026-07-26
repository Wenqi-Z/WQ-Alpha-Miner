import { useNavigate } from 'react-router-dom'
import { useImproveList } from '../api/queries'
import { Badge } from '../components/Badge'
import { ExprCell } from '../components/ExprCell'
import { shortId } from '../lib/format'

export function ImproveList() {
  const { data, isLoading, error } = useImproveList()
  const navigate = useNavigate()

  if (error) return <div className="empty">{(error as Error).message}</div>
  if (isLoading || !data) return <div className="empty">Loading…</div>

  return (
    <div className="page">
      <div style={{ marginBottom: 16 }}>
        <div className="overline">LLM refinement</div>
        <div style={{ fontSize: 20, fontWeight: 650, marginTop: 2 }}>
          Improvements
        </div>
      </div>

      <div className="panel">
        <div className="panel-hd">
          <h3>Improvement Sessions</h3>
          <div className="spacer" />
          <span className="faint" style={{ fontSize: 12 }}>
            launch from Candidates → Improve
          </span>
        </div>
        <div className="panel-bd table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 120 }}>Session</th>
                <th style={{ width: 100 }}>Status</th>
                <th className="col-expr">Seed Expression</th>
                <th className="num">Seed Sharpe</th>
                <th className="num">Best Sharpe</th>
                <th className="num">Lift</th>
                <th className="num">Variants</th>
                <th style={{ width: 80 }}>Started</th>
              </tr>
            </thead>
            <tbody>
              {data.items.length === 0 ? (
                <tr>
                  <td colSpan={8} className="empty">
                    No improvement sessions yet — pick a candidate and click
                    Improve.
                  </td>
                </tr>
              ) : (
                data.items.map((s) => (
                  <tr
                    key={s.id}
                    className="clickable"
                    onClick={() => navigate(`/improve/${s.id}`)}
                  >
                    <td className="mono">{shortId(s.id)}</td>
                    <td>
                      <Badge status={s.st} />
                    </td>
                    <ExprCell expr={s.seed_expr || '—'} />
                    <td className="num">{s.seed_sharpe.toFixed(2)}</td>
                    <td className={`num ${s.best_sharpe >= 1.5 ? 'pos' : ''}`}>
                      {s.best_sharpe.toFixed(2)}
                    </td>
                    <td className={`num ${s.lift >= 0 ? 'pos' : 'neg'}`}>
                      {s.lift >= 0 ? '+' : ''}
                      {s.lift.toFixed(2)}
                    </td>
                    <td className="num">{s.variants}</td>
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
