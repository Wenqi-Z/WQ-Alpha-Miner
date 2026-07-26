import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

interface LineChartProps {
  values: number[]
  color?: string
}

export function LineChart({
  values,
  color = 'var(--accent-2)',
}: LineChartProps) {
  if (values.length < 2) {
    return <div className="empty">Not enough data</div>
  }
  const data = values.map((value, i) => ({ i: i + 1, value }))
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="var(--border)" strokeOpacity={0.35} vertical={false} />
          <XAxis
            dataKey="i"
            tick={{ fill: 'var(--text-faint)', fontSize: 10 }}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: 'var(--text-faint)', fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            width={40}
            domain={['auto', 'auto']}
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
          <Area
            type="monotone"
            dataKey="value"
            fill={color}
            fillOpacity={0.1}
            stroke="none"
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={{ r: 3, fill: color }}
            activeDot={{ r: 4 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
