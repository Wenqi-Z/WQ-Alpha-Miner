import { Navigate, Route, Routes } from 'react-router-dom'
import { Sidebar } from './components/Sidebar'
import { Topbar } from './components/Topbar'
import { ExprTooltipProvider } from './components/ExprCell'
import { Overview } from './pages/Overview'
import { SessionList } from './pages/SessionList'
import { SessionDetail } from './pages/SessionDetail'
import { Candidates } from './pages/Candidates'
import { Submit } from './pages/Submit'
import { ImproveList } from './pages/ImproveList'
import { ImproveDetail } from './pages/ImproveDetail'

export default function App() {
  return (
    <ExprTooltipProvider>
      <Sidebar />
      <div className="main">
        <Topbar />
        <div className="content">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/sessions" element={<SessionList />} />
            <Route path="/sessions/:id" element={<SessionDetail />} />
            <Route path="/candidates" element={<Candidates />} />
            <Route path="/submit" element={<Submit />} />
            <Route path="/improve" element={<ImproveList />} />
            <Route path="/improve/:id" element={<ImproveDetail />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </div>
    </ExprTooltipProvider>
  )
}
