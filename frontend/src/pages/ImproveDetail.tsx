import { useParams } from 'react-router-dom'
import { useImproveDetail, useStopImprove } from '../api/queries'
import { Badge, TagBadge } from '../components/Badge'
import { ExprCell } from '../components/ExprCell'
import { Kpi } from '../components/Kpi'
import { LineChart } from '../components/LineChart'
import { shortId, uiState } from '../lib/format'

export function ImproveDetail() {
  const { id } = useParams<{ id: string }>()
  const { data, isLoading, error } = useImproveDetail(id)
  const stopImprove = useStopImprove()

  if (!id) return <div className="empty">No improvement session selected</div>
  if (error) return <div className="empty">{(error as Error).message}</div>
  if (isLoading || !data) return <div className="empty">Loading…</div>

  const st = uiState(data.session.state)
  const progress =
    data.max_rounds > 0 ? (data.rounds_done / data.max_rounds) * 100 : 0

  return (
    <div className="page">
      <div className="row" style={{ alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div className="overline">
            Improvement Session · {shortId(id)}
          </div>
          <div style={{ fontSize: 20, fontWeight: 650, marginTop: 2 }}>
            Iterative LLM Refinement
          </div>
        </div>
        <div style={{ flex: 1 }} />
        <Badge
          status={st}
          label={`${st} · ${data.rounds_done} / ${data.max_rounds}`}
        />
        {st === 'running' && (
          <button
            className="btn sm"
            type="button"
            onClick={() => stopImprove.mutate()}
            disabled={stopImprove.isPending}
          >
            ■ Stop
          </button>
        )}
      </div>

      <div className="grid cards" style={{ marginBottom: 16 }}>
        <Kpi
          label="Seed Sharpe"
          value={data.seed_sharpe.toFixed(2)}
          sub="original candidate"
        />
        <Kpi
          label="Current Best"
          value={data.best_sharpe.toFixed(2)}
          valueClass="pos"
          sub={
            <span className="pos">
              ▲ +{data.lift.toFixed(2)} vs seed
            </span>
          }
        />
        <Kpi label="Variants Tried" value={data.variants} />
        <div className="card kpi">
          <div className="label">Round</div>
          <div className="val">
            {data.rounds_done} / {data.max_rounds}
          </div>
          <div className="bar-track" style={{ marginTop: 10 }}>
            <div className="bar-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 16 }}>
        <div className="panel-hd">
          <h3>Best Sharpe by Round</h3>
        </div>
        <div className="panel-bd" style={{ padding: 16 }}>
          <LineChart values={data.round_best} />
        </div>
      </div>

      <div className="panel">
        <div className="panel-hd">
          <h3>Round Log</h3>
        </div>
        <div className="panel-bd table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: 64 }}>Round</th>
                <th className="col-expr">Variant</th>
                <th className="num">Sharpe</th>
                <th className="num">Δ</th>
                <th style={{ width: 100 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {data.round_log.map((r, i) => (
                <tr key={i}>
                  <td className="mono">R{r.round}</td>
                  <ExprCell expr={r.variant} wrap />
                  <td className={`num ${r.sharpe >= 1.5 ? 'pos' : ''}`}>
                    {r.sharpe.toFixed(2)}
                  </td>
                  <td
                    className={`num ${
                      r.delta.startsWith('+') ? 'pos' : 'neg'
                    }`}
                  >
                    {r.delta}
                  </td>
                  <td>
                    {r.status === 'seed' ? (
                      <TagBadge kind="eligible">seed</TagBadge>
                    ) : r.status === 'best' ? (
                      <TagBadge kind="cand">new best</TagBadge>
                    ) : (
                      <TagBadge kind="queued">kept</TagBadge>
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
