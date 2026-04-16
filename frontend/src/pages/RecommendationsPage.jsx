import { useState, useEffect } from 'react';
import LoadingSpinner from '../components/shared/LoadingSpinner';
import EmptyState from '../components/shared/EmptyState';
import Badge from '../components/shared/Badge';
import ConvictionScores from '../components/conviction/ConvictionScores';
import { getRecommendations } from '../api/recommendations';

const PAGE_SIZE = 20;

const ACTION_VARIANT = {
  Buy: 'green',
  Hold: 'blue',
  Trim: 'yellow',
  Exit: 'red',
  assessment: 'gray',
};

export default function RecommendationsPage() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [filterTicker, setFilterTicker] = useState('');
  const [filterAction, setFilterAction] = useState('');
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    setLoading(true);
    getRecommendations(page * PAGE_SIZE, PAGE_SIZE)
      .then((data) => { setItems(data.items); setTotal(data.total); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [page]);

  const filtered = items.filter((r) => {
    if (filterTicker && !r.ticker?.toLowerCase().includes(filterTicker.toLowerCase())) return false;
    if (filterAction && r.action !== filterAction) return false;
    return true;
  });

  const totalPages = Math.ceil(total / PAGE_SIZE);

  if (loading && items.length === 0) return <LoadingSpinner />;

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-xl font-bold text-gray-900">Recommendation History</h1>

      <div className="flex gap-3 items-center">
        <input
          type="text"
          placeholder="Filter by ticker..."
          value={filterTicker}
          onChange={(e) => setFilterTicker(e.target.value)}
          className="border border-gray-300 rounded px-3 py-1.5 text-sm w-40"
        />
        <select
          value={filterAction}
          onChange={(e) => setFilterAction(e.target.value)}
          className="border border-gray-300 rounded px-3 py-1.5 text-sm"
        >
          <option value="">All Actions</option>
          <option value="Buy">Buy</option>
          <option value="Hold">Hold</option>
          <option value="Trim">Trim</option>
          <option value="Exit">Exit</option>
          <option value="assessment">Assessment</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <EmptyState message="No recommendations found" />
      ) : (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Run</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Ticker</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Action</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Conviction</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Weight</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Details</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.recommendation_id} className="border-b border-gray-50">
                    <td className="py-2 px-3 text-gray-500">#{r.run_id}</td>
                    <td className="py-2 px-3 font-medium text-gray-900">{r.ticker}</td>
                    <td className="py-2 px-3">
                      <Badge variant={ACTION_VARIANT[r.action] || 'gray'}>{r.action}</Badge>
                    </td>
                    <td className="py-2 px-3">
                      <ConvictionScores scores={r.conviction_scores} />
                    </td>
                    <td className="py-2 px-3 text-right text-gray-700">
                      {r.conviction_weight != null ? `${(r.conviction_weight * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-2 px-3">
                      <button
                        onClick={() => setExpanded(expanded === r.recommendation_id ? null : r.recommendation_id)}
                        className="text-primary text-xs hover:underline"
                      >
                        {expanded === r.recommendation_id ? 'Hide' : 'Show'}
                      </button>
                    </td>
                  </tr>
                ))}
                {filtered.map((r) =>
                  expanded === r.recommendation_id ? (
                    <tr key={`${r.recommendation_id}-detail`}>
                      <td colSpan={6} className="px-3 py-3 bg-gray-50">
                        <div className="text-sm space-y-2">
                          {r.rationale && (
                            <div>
                              <span className="font-medium text-gray-700">Rationale: </span>
                              <span className="text-gray-600">{r.rationale}</span>
                            </div>
                          )}
                          {r.key_quant_metrics && Object.keys(r.key_quant_metrics).length > 0 && (
                            <div>
                              <span className="font-medium text-gray-700 block mb-1">Quant Metrics:</span>
                              <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-gray-600 pl-2">
                                {r.key_quant_metrics.drift != null && (
                                  <div>
                                    <span className="text-gray-500">Drift: </span>
                                    <span className={r.key_quant_metrics.drift > 0 ? 'text-green-600' : r.key_quant_metrics.drift < 0 ? 'text-red-600' : ''}>
                                      {(r.key_quant_metrics.drift * 100).toFixed(2)}%
                                    </span>
                                  </div>
                                )}
                                {r.key_quant_metrics.volatility_30d != null && (
                                  <div><span className="text-gray-500">30d Vol: </span>{(r.key_quant_metrics.volatility_30d * 100).toFixed(1)}%</div>
                                )}
                                {r.key_quant_metrics.current_weight != null && (
                                  <div><span className="text-gray-500">Weight: </span>{(r.key_quant_metrics.current_weight * 100).toFixed(2)}%</div>
                                )}
                                {r.key_quant_metrics.health_score && (
                                  <div>
                                    <span className="text-gray-500">Health: </span>
                                    <span className={
                                      r.key_quant_metrics.health_score === 'normal' ? 'text-green-600' :
                                      r.key_quant_metrics.health_score === 'warning' ? 'text-yellow-600' : 'text-red-600'
                                    }>{r.key_quant_metrics.health_score}</span>
                                  </div>
                                )}
                                {r.key_quant_metrics.flags?.length > 0 && (
                                  <div className="col-span-2"><span className="text-gray-500">Flags: </span>{r.key_quant_metrics.flags.join(', ')}</div>
                                )}
                              </div>
                            </div>
                          )}
                          {r.key_risk_factors?.length > 0 && (
                            <div>
                              <span className="font-medium text-gray-700 block mb-1">Risk Factors:</span>
                              <ul className="list-disc list-inside pl-2 text-gray-600 space-y-0.5">
                                {r.key_risk_factors.map((rf, i) => <li key={i}>{rf}</li>)}
                              </ul>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  ) : null,
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted">{total} total recommendations</span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-3 py-1 text-sm border rounded disabled:opacity-40"
            >
              Prev
            </button>
            <span className="text-sm text-gray-600 py-1">
              Page {page + 1} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-3 py-1 text-sm border rounded disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
