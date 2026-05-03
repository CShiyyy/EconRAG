import { useState, useEffect, useCallback } from 'react';
import Card from '../components/shared/Card';
import Badge from '../components/shared/Badge';
import EmptyState from '../components/shared/EmptyState';
import AllocationChart from '../components/charts/AllocationChart';
import LoadingSpinner from '../components/shared/LoadingSpinner';
import TradesSection from '../components/trades/TradesSection';
import { getPortfolio } from '../api/portfolio';
import { getConstraints } from '../api/constraints';
import { triggerRun, getTriggerStatus, getSlotStatus } from '../api/runs';
import { usePolling } from '../hooks/usePolling';

function fmt(n) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n);
}

function pct(n) {
  return `${(n * 100).toFixed(1)}%`;
}

export default function DashboardPage() {
  const [portfolio, setPortfolio] = useState(null);
  const [constraints, setConstraints] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [slotStatus, setSlotStatus] = useState(null);
  const [triggerId, setTriggerId] = useState(null);
  const [triggerBusy, setTriggerBusy] = useState(false);

  const fetchData = useCallback(({ signal } = {}) => {
    setError(null);
    return Promise.allSettled([
      getPortfolio({ signal }),
      getConstraints({ signal }),
    ]).then(([pRes, cRes]) => {
      if (pRes.status === 'fulfilled') {
        setPortfolio(pRes.value);
      } else if (pRes.reason?.name !== 'CanceledError' && pRes.reason?.name !== 'AbortError') {
        setError('Failed to load portfolio data');
      }
      if (cRes.status === 'fulfilled') {
        setConstraints(cRes.value.constraints);
      }
      setLoading(false);
    });
  }, []);

  const fetchSlotStatus = useCallback(() => {
    getSlotStatus().then(setSlotStatus).catch(() => {});
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetchData({ signal: controller.signal });
    fetchSlotStatus();
    return () => controller.abort();
  }, [fetchData, fetchSlotStatus]);

  // Poll slot-status every 60s so the banner stays current across ET time boundaries
  useEffect(() => {
    const id = setInterval(fetchSlotStatus, 60_000);
    return () => clearInterval(id);
  }, [fetchSlotStatus]);

  const pollFn = useCallback(
    () => (triggerId != null ? getTriggerStatus(triggerId) : Promise.resolve(null)),
    [triggerId],
  );
  const { data: triggerData } = usePolling(pollFn, 2000, triggerId != null);

  useEffect(() => {
    if (triggerData?.status === 'completed' || triggerData?.status === 'failed') {
      setTriggerBusy(false);
      if (triggerData.status === 'completed') {
        fetchData();
        fetchSlotStatus();
      }
    }
  }, [triggerData, fetchData, fetchSlotStatus]);

  const handleTrigger = async () => {
    setTriggerBusy(true);
    try {
      const res = await triggerRun({});
      setTriggerId(res.trigger_id);
    } catch {
      setTriggerBusy(false);
    }
  };

  if (loading) return <LoadingSpinner />;

  const isClosedDay = slotStatus?.market_phase === 'closed_day';
  const isMarketOpen = slotStatus?.market_phase === 'open';
  const buttonLabel = triggerBusy ? 'Running...' : 'Trigger Run';

  const holdingsCount = portfolio?.holdings?.length || 0;
  const cashPct = portfolio?.total_value > 0 ? portfolio.cash / portfolio.total_value : 1;

  return (
    <div className="p-6 space-y-6">
      {error && (
        <div className="rounded-md bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {isMarketOpen && (
        <div className="rounded-md bg-amber-50 border border-amber-200 px-4 py-3 text-sm text-amber-800">
          Market currently open — data is mid-session and will not reflect today's close.
        </div>
      )}
      {isClosedDay && (
        <div className="rounded-md bg-gray-50 border border-gray-200 px-4 py-3 text-sm text-gray-600">
          Markets closed today — any trades queued now will execute at the next market open.
        </div>
      )}

      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900">Dashboard</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={handleTrigger}
            disabled={triggerBusy}
            className="bg-primary text-white text-sm px-4 py-1.5 rounded hover:bg-primary-dark disabled:opacity-50 transition-colors"
          >
            {buttonLabel}
          </button>
          {triggerData && (
            <Badge variant={triggerData.status === 'completed' ? 'green' : triggerData.status === 'failed' ? 'red' : 'yellow'}>
              {triggerData.status}
            </Badge>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card title="Portfolio Value" value={fmt(portfolio?.total_value || 0)} />
        <Card title="Cash Balance" value={fmt(portfolio?.cash || 0)} subtitle={pct(cashPct) + ' of portfolio'} />
        <Card title="Holdings" value={holdingsCount} subtitle={holdingsCount === 0 ? 'No positions yet' : `${holdingsCount} positions`} />
        <Card title="Cash Allocation" value={pct(cashPct)} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-sm font-medium text-gray-700 mb-3">Allocation</h2>
          {holdingsCount > 0 ? (
            <AllocationChart weights={portfolio.weights} />
          ) : (
            <EmptyState message="No holdings to display" />
          )}
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-sm font-medium text-gray-700 mb-3">Constraints</h2>
          {constraints ? (
            <div className="space-y-2">
              {constraints.map((c) => (
                <div key={c.constraint_name} className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0">
                  <span className="text-sm text-gray-600">{c.description || c.constraint_name}</span>
                  <Badge variant="blue">{pct(c.value)}</Badge>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState message="No constraints loaded" />
          )}
        </div>
      </div>

      {holdingsCount > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-sm font-medium text-gray-700 mb-3">Current Holdings</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-2 font-medium text-gray-600">Ticker</th>
                  <th className="text-right py-2 px-2 font-medium text-gray-600">Shares</th>
                  <th className="text-right py-2 px-2 font-medium text-gray-600">Avg Cost</th>
                  <th className="text-right py-2 px-2 font-medium text-gray-600">Weight</th>
                </tr>
              </thead>
              <tbody>
                {portfolio.holdings.map((h) => (
                  <tr key={h.ticker} className="border-b border-gray-50">
                    <td className="py-2 px-2 font-medium text-gray-900">{h.ticker}</td>
                    <td className="py-2 px-2 text-right text-gray-700">{h.shares}</td>
                    <td className="py-2 px-2 text-right text-gray-700">{fmt(h.avg_cost_basis)}</td>
                    <td className="py-2 px-2 text-right text-gray-700">
                      {portfolio.weights[h.ticker] !== undefined ? pct(portfolio.weights[h.ticker]) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <TradesSection />
    </div>
  );
}
