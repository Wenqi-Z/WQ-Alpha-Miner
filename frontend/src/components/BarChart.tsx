import {
  Bar,
  BarChart as RBarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

interface BarChartProps {
  labels: string[]
  values: number[]
  color?: string
}

export function BarChart({
  labels,
  values,
  color = 'var(--accent)',
}: BarChartProps) {
  if (!values.length) {
    return <div className="empty">No data</div>
  }
  const data = labels.map((label, i) => ({
    label,
    value: values[i] ?? 0,
  }))
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <RBarChart data={data} margin={{ top: 16, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="var(--border)" strokeOpacity={0.35} vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: 'var(--text-faint)', fontSize: 10 }}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: 'var(--text-faint)', fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            width={36}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--panel-2)',
              border: '1px solid var(--border-2)',
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: 'var(--text-dim)' }}
            itemStyle={{ color: 'var(--text)' }}
          />
          <Bar dataKey="value" fill={color} radius={[3, 3, 0, 0]} />
        </RBarChart>
      </ResponsiveContainer>
    </div>
  )
}
