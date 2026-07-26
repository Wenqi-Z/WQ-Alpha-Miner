import { useCheckSelfCorrelation, useSubmitAlpha } from '../api/queries'
import type { SubmitReadyItem } from '../api/types'

type Notice = { tone: 'success' | 'danger'; message: string }

export function SubmitReadyPanel({
  items,
  onNotice,
}: {
  items: SubmitReadyItem[]
  onNotice: (notice: Notice) => void
}) {
  const submit = useSubmitAlpha()
  const checkSc = useCheckSelfCorrelation()

  return (
    <div className="panel">
      <div className="panel-hd">
        <h3>Submission-Ready</h3>
        <span className="badge submit">{items.length}</span>
        <div className="spacer" />
      </div>
      <div className="panel-bd">
        {items.length === 0 ? (
          <div className="empty">No submission-ready alphas yet.</div>
        ) : (
          items.map((s, i) => (
            <div className="submit-item" key={s.alpha_id ?? i}>
              <div className="expr">
                <div className="mono" style={{ fontSize: 12 }}>
                  {s.expr}
                </div>
                <div className="faint" style={{ fontSize: 11, marginTop: 3 }}>
                  {s.src}
                </div>
              </div>
              <div className="mets">
                <span className="pos">S {s.sharpe.toFixed(2)}</span> · F{' '}
                {s.fit.toFixed(2)}
                {s.alpha_id &&
                  (s.self_correlation === 'PASS' ? (
                    <button
                      className="btn sm"
                      type="button"
                      style={{ marginLeft: 8 }}
                      disabled={submit.isPending}
                      onClick={async () => {
                        try {
                          const r = await submit.mutateAsync(s.alpha_id!)
                          onNotice({
                            tone: 'success',
                            message: `Submitted ${s.alpha_id}: ${r.status}`,
                          })
                        } catch (err) {
                          onNotice({
                            tone: 'danger',
                            message:
                              err instanceof Error ? err.message : String(err),
                          })
                        }
                      }}
                    >
                      Submit
                    </button>
                  ) : (
                    <button
                      className="btn sm"
                      type="button"
                      style={{ marginLeft: 8, minWidth: 148, textAlign: 'center' }}
                      disabled={checkSc.isPending}
                      onClick={async () => {
                        try {
                          const r = await checkSc.mutateAsync(s.alpha_id!)
                          if (r.result === 'PASS') {
                            onNotice({
                              tone: 'success',
                              message: `Self-correlation passed for ${s.alpha_id}. It is ready to submit.`,
                            })
                          } else {
                            onNotice({
                              tone: 'danger',
                              message: `Self-correlation failed for ${s.alpha_id}. It was removed from Submission-Ready.`,
                            })
                          }
                        } catch (err) {
                          onNotice({
                            tone: 'danger',
                            message:
                              err instanceof Error ? err.message : String(err),
                          })
                        }
                      }}
                    >
                      {checkSc.isPending ? 'Checking…' : 'Check self-correlation'}
                    </button>
                  ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
