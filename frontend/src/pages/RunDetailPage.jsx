import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import LoadingSpinner from '../components/shared/LoadingSpinner';
import Badge from '../components/shared/Badge';
import JsonViewer from '../components/shared/JsonViewer';
import ConvictionScores from '../components/conviction/ConvictionScores';
import { getRunDetail } from '../api/runs';

function fmt(n) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n);
}

const ACTION_VARIANT = { Buy: 'green', Hold: 'blue', Trim: 'yellow', Exit: 'red', assessment: 'gray' };

export default function RunDetailPage() {
  const { runId } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getRunDetail(runId)
      .then(setData)
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load run'))
      .finally(() => setLoading(false));
  }, [runId]);

  if (loading) return <LoadingSpinner />;
  if (error) return <div className="p-6 text-danger">{error}</div>;

  const { run, agent_outputs, recommendations, trades } = data;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center gap-3">
        <Link to="/runs" className="text-primary text-sm hover:underline">&larr; Back</Link>
        <h1 className="text-xl font-bold text-gray-900">Run #{run.run_id}</h1>
        <Badge variant={run.run_type === 'pre_open' ? 'blue' : 'gray'}>{run.run_type}</Badge>
        <Badge variant={run.status === 'completed' ? 'green' : run.status === 'failed' ? 'red' : 'yellow'}>
          {run.status || 'unknown'}
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-3">
          <p className="text-xs text-muted">Started</p>
          <p className="text-sm font-medium">{run.started_at || '—'}</p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-3">
          <p className="text-xs text-muted">Completed</p>
          <p className="text-sm font-medium">{run.completed_at || '—'}</p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-3">
          <p className="text-xs text-muted">Re-query</p>
          <p className="text-sm font-medium">{run.requery_count ?? 0} time(s)</p>
        </div>
      </div>

      {run.source_status && (
        <div>
          <h2 className="text-sm font-medium text-gray-700 mb-2">Source Health</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {(Array.isArray(run.source_status) ? run.source_status : Object.values(run.source_status)).map((s, i) => (
              <div key={i} className="bg-white rounded-lg border border-gray-200 p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium">{s.source}</span>
                  <Badge variant={s.status === 'success' ? 'green' : s.status === 'timeout' ? 'yellow' : 'red'}>
                    {s.status}
                  </Badge>
                </div>
                <p className="text-xs text-muted">{s.items_fetched ?? 0} items, {s.duration_ms ?? 0}ms</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <h2 className="text-sm font-medium text-gray-700 mb-2">Agent Outputs</h2>
        <div className="space-y-2">
          {['A', 'B', 'C'].map((agent) => (
            <JsonViewer key={agent} label={`Agent ${agent}`} data={agent_outputs?.[agent]} />
          ))}
        </div>
      </div>

      {recommendations?.length > 0 && (
        <div>
          <h2 className="text-sm font-medium text-gray-700 mb-2">Recommendations</h2>
          <div className="bg-white rounded-lg border border-gray-200 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Ticker</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Action</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Conviction</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Weight</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Rationale</th>
                </tr>
              </thead>
              <tbody>
                {recommendations.map((r) => (
                  <tr key={r.recommendation_id} className="border-b border-gray-50">
                    <td className="py-2 px-3 font-medium text-gray-900">{r.ticker}</td>
                    <td className="py-2 px-3"><Badge variant={ACTION_VARIANT[r.action] || 'gray'}>{r.action}</Badge></td>
                    <td className="py-2 px-3"><ConvictionScores scores={r.conviction_scores} /></td>
                    <td className="py-2 px-3 text-right text-gray-700">
                      {r.conviction_weight != null ? `${(r.conviction_weight * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-2 px-3 text-gray-600 max-w-xs truncate">{r.rationale || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {trades?.length > 0 && (
        <div>
          <h2 className="text-sm font-medium text-gray-700 mb-2">Trades</h2>
          <div className="bg-white rounded-lg border border-gray-200 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Ticker</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Action</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Shares</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Fill Price</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Realized P&L</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t) => (
                  <tr key={t.trade_id} className="border-b border-gray-50">
                    <td className="py-2 px-3 font-medium text-gray-900">{t.ticker}</td>
                    <td className="py-2 px-3"><Badge variant={ACTION_VARIANT[t.action] || 'gray'}>{t.action}</Badge></td>
                    <td className="py-2 px-3 text-right text-gray-700">{t.shares}</td>
                    <td className="py-2 px-3 text-right text-gray-700">{fmt(t.simulated_fill_price)}</td>
                    <td className="py-2 px-3 text-right text-gray-700">
                      {t.realized_pnl != null ? fmt(t.realized_pnl) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
