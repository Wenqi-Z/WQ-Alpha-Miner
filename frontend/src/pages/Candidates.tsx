import { useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { useArchiveCandidate, useCandidates, useImproveCandidate } from '../api/queries'
import type { StatusResponse } from '../api/types'
import { Badge, TagBadge } from '../components/Badge'
import { ExprCell } from '../components/ExprCell'
import { TestsCell } from '../components/TestsCell'
import { api } from '../api/client'
import { shortId } from '../lib/format'

function candStatusBadge(status: string) {
  if (status === 'open') return <TagBadge kind="cand">candidate</TagBadge>
  if (status === 'improving') return <TagBadge kind="improved">improving</TagBadge>
  if (status === 'improved') return <TagBadge kind="improved">improved</TagBadge>
  return null
}

export function Candidates() {
  const { data, isLoading, error } = useCandidates()
  const improve = useImproveCandidate()
  const archive = useArchiveCandidate()
  const navigate = useNavigate()
  const qc = useQueryClient()

  const onImprove = async (sessionId: string, alphaId: string) => {
    try {
      await improve.mutateAsync({ sessionId, alphaId })
      const st = await qc.fetchQuery({
        queryKey: ['status'],
        queryFn: () => api<StatusResponse>('/status'),
      })
      if (st.improve?.active?.id) {
        navigate(`/improve/${st.improve.active.id}`)
      } else {
        navigate('/improve')
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : String(err))
    }
  }

  if (error) {
    return <div className="empty">{(error as Error).message}</div>
  }
  if (isLoading || !data) {
    return <div className="empty">Loading…</div>
  }

  return (
    <div className="page">
      <div style={{ marginBottom: 16 }}>
        <div className="overline">Post-mining triage</div>
        <div style={{ fontSize: 20, fontWeight: 650, marginTop: 2 }}>
          Candidates
        </div>
      </div>

      <div className="panel">
        <div className="panel-hd">
          <h3>Candidates by Mining Session</h3>
          <div className="spacer" />
          <span className="faint" style={{ fontSize: 12 }}>
            improve eligible alphas
          </span>
        </div>
        <div className="panel-bd">
          {data.groups.length === 0 ? (
            <div className="empty">
              No candidates yet — eligible alphas appear once a mining session has them.
            </div>
          ) : (
            data.groups.map((g) => (
              <div className="session-group" key={g.session}>
                <div className="session-group-hd">
                  <span className="sid">{shortId(g.session)}</span>
                  <Badge status={g.st} />
                  <span className="meta">{g.reg}</span>
                  <div className="spacer" />
                  <span className="faint" style={{ fontSize: 12 }}>
                    {g.items.filter((i) => i.status === 'open').length} open ·{' '}
                    {g.items.length} total
                  </span>
                </div>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th className="col-expr">Expression</th>
                        <th className="num">Sharpe</th>
                        <th className="num">Fitness</th>
                        <th className="num" style={{ width: 64 }}>
                          Tests
                        </th>
                        <th className="col-status">Status</th>
                        <th className="col-action" style={{ textAlign: 'right' }}>
                          Action
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {g.items.map((c) => (
                        <tr key={c.id}>
                          <ExprCell expr={c.expr} />
                          <td className="num pos">{c.sharpe.toFixed(2)}</td>
                          <td className="num">{c.fit.toFixed(2)}</td>
                          <TestsCell
                            passed={c.tests_passed}
                            total={c.tests_total}
                            checks={c.checks}
                          />
                          <td>{candStatusBadge(c.status)}</td>
                          <td style={{ textAlign: 'right' }}>
                            <div className="actions">
                              {c.status === 'open' ? (
                                <button
                                  className="btn sm primary"
                                  type="button"
                                  disabled={!c.can_improve || improve.isPending}
                                  onClick={() => onImprove(g.session, c.id)}
                                >
                                  Improve
                                </button>
                              ) : (
                                <span className="faint" style={{ fontSize: 12 }}>
                                  {c.status}
                                </span>
                              )}
                              <button
                                className="btn sm"
                                type="button"
                                disabled={archive.isPending}
                                onClick={async () => {
                                  try {
                                    await archive.mutateAsync({
                                      sessionId: g.session,
                                      alphaId: c.id,
                                    })
                                  } catch (err) {
                                    alert(
                                      err instanceof Error
                                        ? err.message
                                        : String(err),
                                    )
                                  }
                                }}
                              >
                                Archive
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
