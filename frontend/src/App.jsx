import { Routes, Route, Navigate, Outlet } from 'react-router-dom';
import { useInitCheck } from './hooks/useInitCheck';
import LoadingSpinner from './components/shared/LoadingSpinner';
import InitPage from './pages/InitPage';
import DashboardPage from './pages/DashboardPage';
import RecommendationsPage from './pages/RecommendationsPage';
import RunsPage from './pages/RunsPage';
import RunDetailPage from './pages/RunDetailPage';
import PnlPage from './pages/PnlPage';
import StandingEventsPage from './pages/StandingEventsPage';
import ConstraintsPage from './pages/ConstraintsPage';
import AppShell from './components/layout/AppShell';

function InitGuard() {
  const { initialized, loading } = useInitCheck();

  if (loading) return <LoadingSpinner size="lg" />;
  if (!initialized) return <Navigate to="/init" replace />;
  return <Outlet />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/init" element={<InitPage />} />
      <Route element={<InitGuard />}>
        <Route element={<AppShell />}>
          <Route index element={<DashboardPage />} />
          <Route path="/recommendations" element={<RecommendationsPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
          <Route path="/pnl" element={<PnlPage />} />
          <Route path="/standing-events" element={<StandingEventsPage />} />
          <Route path="/constraints" element={<ConstraintsPage />} />
        </Route>
      </Route>
    </Routes>
  );
}
