import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import LoadingSpinner from '../components/shared/LoadingSpinner';
import EmptyState from '../components/shared/EmptyState';
import Badge from '../components/shared/Badge';
import ConvictionScores from '../components/conviction/ConvictionScores';
import { getLatestRunRecommendations, getRecommendationsForDate, getRecommendationDates } from '../api/recommendations';
import { ACTION_VARIANT } from '../lib/badgeVariants';

function SortIcon({ active, dir }) {
  if (!active) return <span className="ml-1 text-gray-300">↕</span>;
  return <span className="ml-1">{dir === 'desc' ? '↓' : '↑'}</span>;
}

export default function RecommendationsPage() {
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState('');
  const [run, setRun] = useState(null);
  const [runGroups, setRunGroups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(null);
  const [isLatest, setIsLatest] = useState(true);
  const [sortConfig, setSortConfig] = useState({ key: 'conviction_weight', dir: 'desc' });

  function groupByRun(items) {
    const map = new Map();
    for (const item of items) {
      if (!map.has(item.run_id)) map.set(item.run_id, []);
      map.get(item.run_id).push(item);
    }
    return [...map.entries()].map(([runId, recs]) => ({ runId, recs }));
  }

  function sortRecs(recs) {
    const { key, dir } = sortConfig;
    return [...recs].sort((a, b) => {
      const av = a[key] ?? -Infinity;
      const bv = b[key] ?? -Infinity;
      return dir === 'desc' ? bv - av : av - bv;
    });
  }

  function handleSort(key) {
    setSortConfig((prev) =>
      prev.key === key
        ? { key, dir: prev.dir === 'desc' ? 'asc' : 'desc' }
        : { key, dir: 'desc' }
    );
  }

  async function loadLatest() {
    setLoading(true);
    setExpanded(null);
    setIsLatest(true);
    setSelectedDate('');
    try {
      const data = await getLatestRunRecommendations();
      setRun(data.run);
      setRunGroups(data.run ? [{ runId: data.run.run_id, recs: data.items || [] }] : []);
    } catch {
      setRun(null);
      setRunGroups([]);
    } finally {
      setLoading(false);
    }
  }

  async function loadDate(date) {
    if (!date) return;
    setLoading(true);
    setExpanded(null);
    setIsLatest(false);
    try {
      const data = await getRecommendationsForDate(date);
      const groups = groupByRun(data.items || []);
      setRunGroups(groups);
      setRun(null);
    } catch {
      setRunGroups([]);
      setRun(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([getLatestRunRecommendations(), getRecommendationDates()])
      .then(([latest, datesData]) => {
        if (cancelled) return;
        const dates = datesData.dates || [];
        setAvailableDates(dates);
        setRun(latest.run);
        setRunGroups(latest.run ? [{ runId: latest.run.run_id, recs: latest.items || [] }] : []);
        if (latest.run?.session_date) setSelectedDate(latest.run.session_date);
        else if (dates.length > 0) setSelectedDate(dates[0]);
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const goToPrev = () => {
    const idx = availableDates.indexOf(selectedDate);
    if (idx < availableDates.length - 1) {
      const d = availableDates[idx + 1];
      setSelectedDate(d);
      loadDate(d);
    }
  };

  const goToNext = () => {
    const idx = availableDates.indexOf(selectedDate);
    if (idx > 0) {
      const d = availableDates[idx - 1];
      setSelectedDate(d);
      loadDate(d);
    }
  };

  const handleDateChange = (e) => {
    const d = e.target.value;
    setSelectedDate(d);
    loadDate(d);
  };

  const prevDisabled = availableDates.indexOf(selectedDate) >= availableDates.length - 1;
  const nextDisabled = availableDates.indexOf(selectedDate) <= 0;
  const allItems = runGroups.flatMap((g) => g.recs);

  function SortableTh({ label, sortKey, className = '' }) {
    const active = sortConfig.key === sortKey;
    return (
      <th
        className={`py-2 px-3 font-medium text-gray-600 cursor-pointer select-none hover:text-gray-900 ${className}`}
        onClick={() => handleSort(sortKey)}
      >
        {label}
        <SortIcon active={active} dir={sortConfig.dir} />
      </th>
    );
  }

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-xl font-bold text-gray-900">Recommendations</h1>

      {/* Date navigator */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={goToPrev}
          disabled={prevDisabled}
          className="px-2 py-1 text-sm border rounded disabled:opacity-40 hover:bg-gray-50"
        >
          &#8592;
        </button>
        <input
          type="date"
          value={selectedDate}
          onChange={handleDateChange}
          className="border border-gray-300 rounded px-2 py-1 text-sm"
        />
        <button
          onClick={goToNext}
          disabled={nextDisabled}
          className="px-2 py-1 text-sm border rounded disabled:opacity-40 hover:bg-gray-50"
        >
          &#8594;
        </button>
        <button
          onClick={loadLatest}
          className="px-3 py-1 text-sm border rounded hover:bg-gray-50 text-gray-600"
        >
          Latest
        </button>
      </div>

      {/* Run context header */}
      {!loading && isLatest && run && (
        <div className="flex flex-wrap items-center gap-2 text-sm text-gray-500">
          <Link to={`/runs/${run.run_id}`} className="font-medium text-primary hover:underline">
            Run #{run.run_id}
          </Link>
          <span>·</span>
          <span>{run.session_date}</span>
          {run.started_at && (
            <>
              <span>·</span>
              <span>started {run.started_at}</span>
            </>
          )}
          <span>·</span>
          <Badge variant="gray">{allItems.length} recommendations</Badge>
        </div>
      )}

      {loading ? (
        <LoadingSpinner />
      ) : allItems.length === 0 ? (
        <EmptyState
          message={
            isLatest
              ? 'No runs yet.'
              : `No recommendations recorded for ${selectedDate || 'this date'}.`
          }
        />
      ) : (
        <div className="space-y-6">
          {runGroups.map(({ runId, recs }) => {
            const sorted = sortRecs(recs);
            return (
              <div key={runId} className="bg-white rounded-lg border border-gray-200">
                {!isLatest && (
                  <div className="px-3 py-2 border-b border-gray-100 text-xs text-gray-400">
                    <Link to={`/runs/${runId}`} className="hover:underline text-primary font-medium">
                      Run #{runId}
                    </Link>
                  </div>
                )}
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 px-3 font-medium text-gray-600">Ticker</th>
                        <th className="text-left py-2 px-3 font-medium text-gray-600">Action</th>
                        <th className="text-left py-2 px-3 font-medium text-gray-600">Conviction</th>
                        <SortableTh label="Score" sortKey="conviction_weight" className="text-right" />
                        <SortableTh label="Allocation" sortKey="target_weight" className="text-right" />
                        <th className="text-left py-2 px-3 font-medium text-gray-600">Details</th>
                        <th className="text-left py-2 px-3 font-medium text-gray-600">Run</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sorted.map((r) => (
                        <tr key={r.recommendation_id} className="border-b border-gray-50">
                          <td className="py-2 px-3 font-medium text-gray-900">{r.ticker}</td>
                          <td className="py-2 px-3">
                            <Badge variant={ACTION_VARIANT[r.action] || 'gray'}>{r.action}</Badge>
                          </td>
                          <td className="py-2 px-3">
                            <ConvictionScores scores={r.conviction_scores} />
                          </td>
                          <td className="py-2 px-3 text-right text-gray-700 tabular-nums">
                            {r.conviction_weight != null ? r.conviction_weight.toFixed(2) : '—'}
                          </td>
                          <td className="py-2 px-3 text-right text-gray-700 tabular-nums">
                            {r.target_weight != null ? `${(r.target_weight * 100).toFixed(1)}%` : '—'}
                          </td>
                          <td className="py-2 px-3">
                            <button
                              onClick={() => setExpanded(expanded === r.recommendation_id ? null : r.recommendation_id)}
                              className="text-primary text-xs hover:underline"
                            >
                              {expanded === r.recommendation_id ? 'Hide' : 'Show'}
                            </button>
                          </td>
                          <td className="py-2 px-3 text-gray-400 text-xs">
                            <Link to={`/runs/${r.run_id}`} className="hover:underline text-primary">
                              #{r.run_id}
                            </Link>
                          </td>
                        </tr>
                      ))}
                      {sorted.map((r) =>
                        expanded === r.recommendation_id ? (
                          <tr key={`${r.recommendation_id}-detail`}>
                            <td colSpan={7} className="px-3 py-3 bg-gray-50">
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
            );
          })}
        </div>
      )}
    </div>
  );
}
