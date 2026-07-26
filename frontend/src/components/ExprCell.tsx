import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'

interface TooltipState {
  text: string
  x: number
  y: number
  visible: boolean
}

interface ExprTooltipApi {
  show: (text: string, x: number, y: number) => void
  move: (x: number, y: number) => void
  hide: () => void
}

const ExprTooltipCtx = createContext<ExprTooltipApi | null>(null)

export { ExprTooltipCtx }

export function ExprTooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TooltipState>({
    text: '',
    x: 0,
    y: 0,
    visible: false,
  })

  const show = useCallback((text: string, x: number, y: number) => {
    setTip({ text, x, y, visible: true })
  }, [])
  const move = useCallback((x: number, y: number) => {
    setTip((t) => (t.visible ? { ...t, x, y } : t))
  }, [])
  const hide = useCallback(() => {
    setTip((t) => ({ ...t, visible: false }))
  }, [])

  useEffect(() => {
    const onScroll = () => hide()
    document.addEventListener('scroll', onScroll, true)
    return () => document.removeEventListener('scroll', onScroll, true)
  }, [hide])

  const pad = 12
  let left = tip.x + pad
  let top = tip.y + pad
  // rough clamp — tooltip measures after paint; good enough
  if (typeof window !== 'undefined') {
    if (left > window.innerWidth - 280) left = tip.x - 280 - pad
    if (top > window.innerHeight - 160) top = tip.y - 160 - pad
    left = Math.max(8, left)
    top = Math.max(8, top)
  }

  return (
    <ExprTooltipCtx.Provider value={{ show, move, hide }}>
      {children}
      {tip.visible && (
        <div
          className="expr-tooltip"
          style={{ left, top }}
          hidden={!tip.visible}
        >
          {tip.text}
        </div>
      )}
    </ExprTooltipCtx.Provider>
  )
}

export function ExprCell({
  expr,
  className = '',
  wrap = false,
}: {
  expr: string
  className?: string
  wrap?: boolean
}) {
  const api = useContext(ExprTooltipCtx)
  return (
    <td
      className={`mono col-expr expr-cell ${wrap ? 'col-expr-wrap' : ''} ${className}`.trim()}
      onMouseEnter={(e) => api?.show(expr, e.clientX, e.clientY)}
      onMouseMove={(e) => api?.move(e.clientX, e.clientY)}
      onMouseLeave={() => api?.hide()}
    >
      {expr}
    </td>
  )
}

export function ExprSpan({ expr }: { expr: string }) {
  const api = useContext(ExprTooltipCtx)
  return (
    <span
      className="mono expr-cell"
      onMouseEnter={(e) => api?.show(expr, e.clientX, e.clientY)}
      onMouseMove={(e) => api?.move(e.clientX, e.clientY)}
      onMouseLeave={() => api?.hide()}
    >
      {expr}
    </span>
  )
}
