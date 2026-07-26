import { useParams } from 'react-router-dom'
import { useSessionDetail } from '../api/queries'
import { Badge, TagBadge } from '../components/Badge'
import { ExprCell } from '../components/ExprCell'
import { Kpi } from '../components/Kpi'
import { LineChart } from '../components/LineChart'
import { TestsCell } from '../components/TestsCell'
import { fmtPct, shortId, uiState } from '../lib/format'

export function SessionDetail() {
  const { id } = useParams<{ id: string }>()
  const { data, isLoading, error } = useSessionDetail(id)

  if (!id) return <div className="empty">No session selected</div>
  if (error) return <div className="empty">{(error as Error).message}</div>
  if (isLoading || !data) return <div className="empty">Loading…</div>

  const s = data.session
  const st = uiState(s.state)
  const cfg = data.config || {}

  return (
    <div className="page">
      <div className="row" style={{ alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div className="overline">Mining Session</div>
          <div style={{ fontSize: 20, fontWeight: 650, marginTop: 2 }}>
            {shortId(id)} · {cfg.region || ''} {cfg.universe || ''}
          </div>
        </div>
        <div style={{ flex: 1 }} />
        <Badge status={st} />
      </div>

      <div
        className="grid cards"
        style={{ marginBottom: 16, gridTemplateColumns: 'repeat(3, 1fr)' }}
      >
        <Kpi label="Alphas Evaluated" value={data.n_alphas} />
        <Kpi
          label="Best Sharpe"
          value={data.best_sharpe.toFixed(2)}
          valueClass="pos"
          sub={`fitness ${data.best_fit.toFixed(2)}`}
        />
        <Kpi label="Elapsed" value={data.duration} />
      </div>

      <div
        className="grid"
        style={{ gridTemplateColumns: '1fr 1fr', marginBottom: 16 }}
      >
        <div className="panel">
          <div className="panel-hd">
            <h3>Best Fitness per Generation</h3>
          </div>
          <div className="panel-bd" style={{ padding: 16 }}>
            <LineChart values={data.fitness_progress} />
          </div>
        </div>
        <div className="panel">
          <div className="panel-hd">
            <h3>Config</h3>
          </div>
          <div
            className="panel-bd"
            style={{
              padding: 16,
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 12,
              fontSize: 13,
            }}
          >
            <div>
              <div className="faint overline">Region</div>
              {cfg.region || '—'}
            </div>
            <div>
              <div className="faint overline">Universe</div>
              {cfg.universe || '—'}
            </div>
            <div>
              <div className="faint overline">Delay</div>
              {cfg.delay ?? '—'}
            </div>
            <div>
              <div className="faint overline">Decay</div>
              {cfg.decay ?? '—'}
            </div>
            <div>
              <div className="faint overline">Neutralization</div>
              {cfg.neutralization || '—'}
            </div>
            <div>
              <div className="faint overline">Truncation</div>
              {cfg.truncation ?? '—'}
            </div>
            <div>
              <div className="faint overline">Pop / Gen</div>
              {data.gp_config?.population_size ?? '—'} / {data.generations}
            </div>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-hd">
          <h3>Alphas</h3>
          <div className="spacer" />
          <span className="chip">{data.n_alphas} rows</span>
          <span className="chip">{data.eligible_count} eligible</span>
        </div>
        <div className="panel-bd table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th className="col-expr" style={{ width: '20%', maxWidth: '20%' }}>
                  Expression
                </th>
                <th className="num">Sharpe</th>
                <th className="num">Fitness</th>
                <th className="num">Turnover</th>
                <th className="num">Returns</th>
                <th className="num" style={{ width: 84 }}>Drawdown</th>
                <th className="num" style={{ width: 64 }}>Tests</th>
                <th className="col-status">Status</th>
              </tr>
            </thead>
            <tbody>
              {data.alphas.map((a, i) => (
                <tr key={i}>
                  <ExprCell expr={a.expr} />
                  <td className={`num ${(a.sharpe || 0) >= 1.5 ? 'pos' : ''}`}>
                    {(a.sharpe || 0).toFixed(2)}
                  </td>
                  <td className="num">{(a.fit || 0).toFixed(2)}</td>
                  <td className="num">{fmtPct(a.turnover)}</td>
                  <td className="num">{fmtPct(a.returns)}</td>
                  <td className="num">{fmtPct(a.drawdown)}</td>
                  <TestsCell
                    passed={a.tests_passed}
                    total={a.tests_total}
                    checks={a.checks}
                  />
                  <td className="col-status">
                    {a.eligible && <TagBadge kind="eligible">eligible</TagBadge>}{' '}
                    {a.improved && <TagBadge kind="improved">improved</TagBadge>}{' '}
                    {a.submit_ready && (
                      <TagBadge kind="submit">submit-ready</TagBadge>
                    )}
                    {!a.eligible && !a.improved && !a.submit_ready && (
                      <span className="faint">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
