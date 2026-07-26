import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { useCandidates } from '../api/queries'
import { SubmitReadyPanel } from '../components/SubmitReadyPanel'

export function Submit() {
  const [notice, setNotice] = useState<{
    tone: 'success' | 'danger'
    message: string
  } | null>(null)
  const { data, isLoading, error } = useCandidates()

  useEffect(() => {
    if (!notice) return
    const t = window.setTimeout(() => setNotice(null), 8000)
    return () => window.clearTimeout(t)
  }, [notice])

  const banner =
    notice &&
    createPortal(
      <div className={`notice ${notice.tone}`} role="status">
        <span>{notice.message}</span>
        <button
          type="button"
          aria-label="Dismiss notification"
          onClick={() => setNotice(null)}
        >
          ×
        </button>
      </div>,
      document.body,
    )

  if (error) {
    return (
      <>
        {banner}
        <div className="empty">{(error as Error).message}</div>
      </>
    )
  }
  if (isLoading || !data) {
    return (
      <>
        {banner}
        <div className="empty">Loading…</div>
      </>
    )
  }

  return (
    <div className="page">
      {banner}
      <div style={{ marginBottom: 16 }}>
        <div className="overline">Ready to ship</div>
        <div style={{ fontSize: 20, fontWeight: 650, marginTop: 2 }}>
          Submission-Ready
        </div>
      </div>

      <SubmitReadyPanel items={data.submit_ready} onNotice={setNotice} />
    </div>
  )
}
