import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import Badge from '../shared/Badge';
import EmptyState from '../shared/EmptyState';
import LoadingSpinner from '../shared/LoadingSpinner';
import { getTrades, getLatestRunTrades } from '../../api/trades';
import { ACTION_VARIANT, TRADE_STATUS_VARIANT } from '../../lib/badgeVariants';

function fmt(n) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n);
}

const PAGE_SIZE = 20;

function LatestRunTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getLatestRunTrades()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner />;
  if (!data?.run) return <EmptyState message="No runs yet" />;
  if (data.trades.length === 0) return <EmptyState message="No trades in the latest run" />;

  const { context, trades } = data;

  return (
    <div className="space-y-3">
      <div className="flex gap-4 text-sm text-gray-600">
        <span>Queued: <strong>{context.queued_count}</strong></span>
        <span>Executed: <strong>{context.executed_count}</strong></span>
        {context.overwritten_count > 0 && (
          <span>Overwritten: <strong>{context.overwritten_count}</strong></span>
        )}
        <span className="ml-auto text-gray-400">
          Run{' '}
          <Link to={`/runs/${data.run.run_id}`} className="text-primary hover:underline">
            #{data.run.run_id}
          </Link>
        </span>
      </div>
      <TradesTable trades={trades} showRunLinks />
    </div>
  );
}

function AllTradesTab() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [statusFilter, setStatusFilter] = useState('');
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    getTrades({ status: statusFilter || undefined, skip: page * PAGE_SIZE, limit: PAGE_SIZE })
      .then((d) => { setItems(d.items); setTotal(d.total); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [statusFilter, page]);

  useEffect(() => { load(); }, [load]);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <select
          value={statusFilter}
          onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
          className="border border-gray-300 rounded px-2 py-1 text-sm"
        >
          <option value="">All statuses</option>
          <option value="queued">Queued</option>
          <option value="executed">Executed</option>
          <option value="overwritten">Overwritten</option>
        </select>
        <span className="text-sm text-gray-500">{total} total</span>
      </div>

      {loading ? (
        <LoadingSpinner />
      ) : items.length === 0 ? (
        <EmptyState message="No trades found" />
      ) : (
        <TradesTable trades={items} showRunLinks />
      )}

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted">{total} trades</span>
          <div className="flex gap-2">
            <button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0} className="px-3 py-1 text-sm border rounded disabled:opacity-40">Prev</button>
            <span className="text-sm text-gray-600 py-1">Page {page + 1} of {totalPages}</span>
            <button onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1} className="px-3 py-1 text-sm border rounded disabled:opacity-40">Next</button>
          </div>
        </div>
      )}
    </div>
  );
}

function TradesTable({ trades, showRunLinks }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-2 px-3 font-medium text-gray-600">Ticker</th>
            <th className="text-left py-2 px-3 font-medium text-gray-600">Action</th>
            <th className="text-left py-2 px-3 font-medium text-gray-600">Status</th>
            <th className="text-right py-2 px-3 font-medium text-gray-600">Shares</th>
            <th className="text-right py-2 px-3 font-medium text-gray-600">Fill Price</th>
            {showRunLinks && <th className="text-left py-2 px-3 font-medium text-gray-600">Queued Run</th>}
            {showRunLinks && <th className="text-left py-2 px-3 font-medium text-gray-600">Exec Run</th>}
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr key={t.trade_id} className="border-b border-gray-50">
              <td className="py-2 px-3 font-medium text-gray-900">{t.ticker}</td>
              <td className="py-2 px-3"><Badge variant={ACTION_VARIANT[t.action] || 'gray'}>{t.action}</Badge></td>
              <td className="py-2 px-3"><Badge variant={TRADE_STATUS_VARIANT[t.status] || 'gray'}>{t.status}</Badge></td>
              <td className="py-2 px-3 text-right text-gray-700">{t.shares}</td>
              <td className="py-2 px-3 text-right text-gray-700">
                {t.execution_fill_price != null ? fmt(t.execution_fill_price) : '—'}
              </td>
              {showRunLinks && (
                <td className="py-2 px-3 text-gray-400 text-xs">
                  {t.queued_run_id != null ? (
                    <Link to={`/runs/${t.queued_run_id}`} className="text-primary hover:underline">#{t.queued_run_id}</Link>
                  ) : '—'}
                </td>
              )}
              {showRunLinks && (
                <td className="py-2 px-3 text-gray-400 text-xs">
                  {t.execution_run_id != null ? (
                    <Link to={`/runs/${t.execution_run_id}`} className="text-primary hover:underline">#{t.execution_run_id}</Link>
                  ) : '—'}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const TABS = ['Latest Run', 'All Trades'];

export default function TradesSection() {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-gray-700">Trades</h2>
        <div className="flex gap-1">
          {TABS.map((tab, i) => (
            <button
              key={tab}
              onClick={() => setActiveTab(i)}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                activeTab === i
                  ? 'bg-primary text-white'
                  : 'text-gray-500 hover:bg-gray-100'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>
      {activeTab === 0 ? <LatestRunTab /> : <AllTradesTab />}
    </div>
  );
}
