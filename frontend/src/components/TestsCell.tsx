import { useContext } from 'react'
import type { CheckResult } from '../api/types'
import { ExprTooltipCtx } from './ExprCell'

function formatChecksTitle(checks: CheckResult[]): string {
  return checks
    .map((c) => {
      const mark = c.result === 'PASS' ? '✓' : c.result === 'FAIL' ? '✗' : '·'
      return `${mark} ${c.name}: ${c.result}`
    })
    .join('\n')
}

/** "4 / 7" with hover listing each IS check result. */
export function TestsCell({
  passed,
  total,
  checks,
}: {
  passed: number
  total: number
  checks?: CheckResult[]
}) {
  const api = useContext(ExprTooltipCtx)
  const tip = checks?.length ? formatChecksTitle(checks) : ''

  return (
    <td
      className={`num tests-cell ${passed === total ? 'pos' : ''}`}
      style={tip ? { cursor: 'help' } : undefined}
      onMouseEnter={(e) => tip && api?.show(tip, e.clientX, e.clientY)}
      onMouseMove={(e) => tip && api?.move(e.clientX, e.clientY)}
      onMouseLeave={() => api?.hide()}
    >
      {passed} / {total}
    </td>
  )
}
