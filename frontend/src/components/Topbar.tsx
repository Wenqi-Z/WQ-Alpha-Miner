import { useQueryClient } from '@tanstack/react-query'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import {
  useStartMining,
  useStatus,
  useStopImprove,
  useStopMining,
} from '../api/queries'
import type { StatusResponse } from '../api/types'
import { shortId } from '../lib/format'

function usePageMeta() {
  const loc = useLocation()
  const params = useParams()
  const path = loc.pathname

  if (path === '/') return { title: 'Overview', crumb: '' }
  if (path === '/sessions') return { title: 'Sessions', crumb: '' }
  if (path.startsWith('/sessions/') && params.id) {
    return { title: 'Session Detail', crumb: `Sessions / ${shortId(params.id)}` }
  }
  if (path === '/candidates') return { title: 'Candidates', crumb: '' }
  if (path === '/improve') return { title: 'Improvements', crumb: '' }
  if (path.startsWith('/improve/') && params.id) {
    return {
      title: 'Improvement Detail',
      crumb: `Improvements / ${shortId(params.id)}`,
    }
  }
  return { title: 'Alpha Mining', crumb: '' }
}

export function Topbar() {
  const { title, crumb } = usePageMeta()
  const { data: st } = useStatus()
  const startMining = useStartMining()
  const stopMining = useStopMining()
  const stopImprove = useStopImprove()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const loc = useLocation()

  const onStart = async () => {
    try {
      await startMining.mutateAsync()
      const fresh = await qc.fetchQuery({
        queryKey: ['status'],
        queryFn: () => api<StatusResponse>('/status'),
      })
      if (fresh.mining?.active?.id) navigate(`/sessions/${fresh.mining.active.id}`)
      else navigate('/sessions')
    } catch (err) {
      alert(err instanceof Error ? err.message : String(err))
    }
  }

  // STOPPING is still "running" server-side (busy) but already stop-queued —
  // hide Stop to avoid double-stop; also hide Start until it fully clears.
  const miningStopping = st?.mining?.active?.state === 'STOPPING'
  const improveStopping = st?.improve?.active?.state === 'STOPPING'
  const miningBusy = !!st?.mining?.running
  const showStopMining = miningBusy && !miningStopping
  const showStopImprove =
    !!st?.improve?.running &&
    !improveStopping &&
    loc.pathname.startsWith('/improve')
  const showStartMining = !miningBusy

  return (
    <div className="topbar">
      <h2>{title}</h2>
      {crumb && <span className="crumb">{crumb}</span>}
      <div className="spacer" />
      <button
        className="btn sm ghost"
        type="button"
        onClick={() => qc.invalidateQueries({ predicate: () => true })}
      >
        ⟳ Refresh
      </button>
      {showStopMining && (
        <button
          className="btn sm"
          type="button"
          onClick={() => stopMining.mutate()}
          disabled={stopMining.isPending}
        >
          ■ Stop Mining
        </button>
      )}
      {showStopImprove && (
        <button
          className="btn sm"
          type="button"
          onClick={() => stopImprove.mutate()}
          disabled={stopImprove.isPending}
        >
          ■ Stop Improve
        </button>
      )}
      {showStartMining && (
        <button
          className="btn sm primary"
          type="button"
          onClick={onStart}
          disabled={startMining.isPending}
        >
          + New Mining Session
        </button>
      )}
    </div>
  )
}
