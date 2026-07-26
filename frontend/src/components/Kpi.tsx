import type { ReactNode } from 'react'

interface KpiProps {
  label: string
  value: ReactNode
  sub?: ReactNode
  valueClass?: string
}

export function Kpi({ label, value, sub, valueClass = '' }: KpiProps) {
  return (
    <div className="card kpi">
      <div className="label">{label}</div>
      <div className={`val ${valueClass}`.trim()}>{value}</div>
      {sub != null && <div className="sub">{sub}</div>}
    </div>
  )
}
