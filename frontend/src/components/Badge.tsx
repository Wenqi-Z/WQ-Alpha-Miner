import type { ReactNode } from 'react'
import { stClass, uiState } from '../lib/format'

interface BadgeProps {
  status: string
  label?: string
  pulse?: boolean
  className?: string
}

export function Badge({ status, label, pulse, className = '' }: BadgeProps) {
  const ui = uiState(status)
  const cls = stClass(ui)
  return (
    <span
      className={`badge ${cls} dot ${pulse || ui === 'running' ? 'pulse' : ''} ${className}`.trim()}
    >
      {label ?? ui}
    </span>
  )
}

export function TagBadge({
  kind,
  children,
}: {
  kind: 'cand' | 'eligible' | 'submit' | 'improved' | 'queued'
  children: ReactNode
}) {
  return <span className={`badge ${kind}`}>{children}</span>
}
