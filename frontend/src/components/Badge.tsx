import type { ReactNode } from 'react'
import { stClass } from '../lib/format'

interface BadgeProps {
  status: string
  label?: string
  pulse?: boolean
  className?: string
}

export function Badge({ status, label, pulse, className = '' }: BadgeProps) {
  const ui = status.toLowerCase()
  const cls = stClass(ui)
  return (
    <span
      className={`badge ${cls} dot ${pulse || ui === 'running' ? 'pulse' : ''} ${className}`.trim()}
    >
      {label ?? status}
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
