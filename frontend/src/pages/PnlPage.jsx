import { useState, useEffect, useMemo } from 'react';
import LoadingSpinner from '../components/shared/LoadingSpinner';
import EmptyState from '../components/shared/EmptyState';
import Card from '../components/shared/Card';
import PerformanceChart from '../components/charts/PerformanceChart';
import { getBenchmark } from '../api/portfolio';

function fmt(n) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n);
}

function pctReturn(initial, current) {
  if (!initial) return '—';
  return `${(((current - initial) / initial) * 100).toFixed(2)}%`;
}

export default function PnlPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    getBenchmark()
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const { portfolioFiltered, benchmarkFiltered } = useMemo(() => {
    if (!data) return { portfolioFiltered: [], benchmarkFiltered: [] };
    const filterByDate = (arr) =>
      arr.filter((p) => {
        const d = p.timestamp?.slice(0, 10);
        if (startDate && d < startDate) return false;
        if (endDate && d > endDate) return false;
        return true;
      });
    return {
      portfolioFiltered: filterByDate(data.portfolio_series),
      benchmarkFiltered: filterByDate(data.benchmark_series),
    };
  }, [data, startDate, endDate]);

  if (loading) return <LoadingSpinner />;
  if (!data || data.portfolio_series.length === 0) {
    return (
      <div className="p-6">
        <h1 className="text-xl font-bold text-gray-900 mb-4">Simulated P&L</h1>
        <EmptyState message="No snapshot data yet. Run the pipeline to generate performance data." />
      </div>
    );
  }

  const initial = data.initial_cash;
  const latestPortfolio = data.portfolio_series[data.portfolio_series.length - 1]?.value || initial;
  const latestBenchmark = data.benchmark_series[data.benchmark_series.length - 1]?.value || initial;

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-bold text-gray-900">Simulated P&L</h1>

      <div className="flex gap-3 items-center">
        <label className="text-sm text-gray-600">From:</label>
        <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="border border-gray-300 rounded px-2 py-1 text-sm" />
        <label className="text-sm text-gray-600">To:</label>
        <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="border border-gray-300 rounded px-2 py-1 text-sm" />
        {(startDate || endDate) && (
          <button onClick={() => { setStartDate(''); setEndDate(''); }} className="text-sm text-primary hover:underline">Clear</button>
        )}
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <PerformanceChart
          portfolioSeries={portfolioFiltered}
          benchmarkSeries={benchmarkFiltered}
          initialCash={initial}
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card title="Portfolio Return" value={pctReturn(initial, latestPortfolio)} subtitle={`Current: ${fmt(latestPortfolio)}`} />
        <Card title="Benchmark Return" value={pctReturn(initial, latestBenchmark)} subtitle={`SPY: ${fmt(latestBenchmark)}`} />
        <Card
          title="Alpha"
          value={`${(((latestPortfolio - initial) / initial - (latestBenchmark - initial) / initial) * 100).toFixed(2)}%`}
          subtitle="vs SPY buy-and-hold"
        />
      </div>
    </div>
  );
}
