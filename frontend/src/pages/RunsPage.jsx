import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import LoadingSpinner from '../components/shared/LoadingSpinner';
import EmptyState from '../components/shared/EmptyState';
import Badge from '../components/shared/Badge';
import { getRuns } from '../api/runs';

const PAGE_SIZE = 20;

export default function RunsPage() {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getRuns(page * PAGE_SIZE, PAGE_SIZE)
      .then((data) => { setItems(data.items); setTotal(data.total); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [page]);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  if (loading && items.length === 0) return <LoadingSpinner />;

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-xl font-bold text-gray-900">Run History</h1>

      {items.length === 0 ? (
        <EmptyState message="No runs yet" />
      ) : (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Date</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Started</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Status</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Queued</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Executed</th>
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Run ID</th>
                </tr>
              </thead>
              <tbody>
                {items.map((run) => (
                  <tr
                    key={run.run_id}
                    onClick={() => navigate(`/runs/${run.run_id}`)}
                    className="border-b border-gray-50 cursor-pointer hover:bg-gray-50"
                  >
                    <td className="py-2 px-3 font-medium text-gray-900">
                      {run.session_date || run.timestamp?.slice(0, 10) || '—'}
                    </td>
                    <td className="py-2 px-3 text-gray-600">
                      {run.timestamp ? run.timestamp.replace('T', ' ').slice(0, 19) + ' UTC' : '—'}
                    </td>
                    <td className="py-2 px-3">
                      <Badge variant={run.wall_clock_seconds != null ? 'green' : 'yellow'}>
                        {run.wall_clock_seconds != null ? 'completed' : 'incomplete'}
                      </Badge>
                    </td>
                    <td className="py-2 px-3 text-right text-gray-600">{run.queued_count ?? 0}</td>
                    <td className="py-2 px-3 text-right text-gray-600">{run.executed_count ?? 0}</td>
                    <td className="py-2 px-3 text-gray-400 text-xs">#{run.run_id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted">{total} total runs</span>
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
