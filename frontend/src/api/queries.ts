import {
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query'
import { api } from './client'
import type {
  CandidatesResponse,
  ImproveDetailResponse,
  ImproveListResponse,
  OverviewResponse,
  SessionDetailResponse,
  StatusResponse,
} from './types'

const POLL = 5000

export const keys = {
  status: ['status'] as const,
  overview: ['overview'] as const,
  session: (id: string) => ['session', id] as const,
  candidates: ['candidates'] as const,
  improveList: ['improveList'] as const,
  improve: (id: string) => ['improve', id] as const,
}

export function useStatus() {
  return useQuery({
    queryKey: keys.status,
    queryFn: () => api<StatusResponse>('/status'),
    refetchInterval: POLL,
  })
}

export function useOverview() {
  return useQuery({
    queryKey: keys.overview,
    queryFn: () => api<OverviewResponse>('/overview'),
    refetchInterval: POLL,
  })
}

export function useSessionDetail(id: string | undefined) {
  return useQuery({
    queryKey: keys.session(id ?? ''),
    queryFn: () => api<SessionDetailResponse>(`/sessions/${id}`),
    enabled: !!id,
    refetchInterval: POLL,
  })
}

export function useCandidates() {
  return useQuery({
    queryKey: keys.candidates,
    queryFn: () => api<CandidatesResponse>('/candidates'),
    refetchInterval: POLL,
  })
}

export function useImproveList() {
  return useQuery({
    queryKey: keys.improveList,
    queryFn: () => api<ImproveListResponse>('/improve'),
    refetchInterval: POLL,
  })
}

export function useImproveDetail(id: string | undefined) {
  return useQuery({
    queryKey: keys.improve(id ?? ''),
    queryFn: () => api<ImproveDetailResponse>(`/improve/${id}`),
    enabled: !!id,
    refetchInterval: POLL,
  })
}

function useInvalidate() {
  const qc = useQueryClient()
  return () =>
    qc.invalidateQueries({
      predicate: () => true,
    })
}

export function useStartMining() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: () =>
      api<{ pid: number }>('/jobs/mining/start', {
        method: 'POST',
        body: JSON.stringify({ auto_restart: true }),
      }),
    onSuccess: invalidate,
  })
}

export function useStopMining() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: () => api('/jobs/mining/stop', { method: 'POST' }),
    onSuccess: invalidate,
  })
}

export function useStopImprove() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: () => api('/jobs/improve/stop', { method: 'POST' }),
    onSuccess: invalidate,
  })
}

export function useImproveCandidate() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ sessionId, alphaId }: { sessionId: string; alphaId: string }) =>
      api<{ pid: number; session_id: string; alpha_id: string }>(
        `/candidates/${sessionId}/${alphaId}/improve`,
        { method: 'POST' },
      ),
    onSuccess: async () => {
      await qc.invalidateQueries({ predicate: () => true })
    },
  })
}

export function useArchiveCandidate() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: ({ sessionId, alphaId }: { sessionId: string; alphaId: string }) =>
      api(`/candidates/${sessionId}/${alphaId}/archive`, { method: 'POST' }),
    onSuccess: invalidate,
  })
}

export function useSubmitAlpha() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (alphaId: string) =>
      api<{ status: string }>(`/submit/${alphaId}`, { method: 'POST' }),
    onSuccess: invalidate,
  })
}

export function useCheckSelfCorrelation() {
  const invalidate = useInvalidate()
  return useMutation({
    mutationFn: (alphaId: string) =>
      api<{ result: string }>(`/check-self-correlation/${alphaId}`, {
        method: 'POST',
      }),
    onSuccess: invalidate,
  })
}
